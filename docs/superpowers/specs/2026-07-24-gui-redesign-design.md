# GUI 交互重构设计文档

> 日期: 2026-07-24
> 范围: `scripts/stroboscopic_gui.py` 全面交互重构
> 版本: 2.0

---

## 1. 目标

将当前"单次线性工作流"的 GUI 改造为"灵活增量编辑"的交互模型。用户可以在任意时刻添加/删除物体、增量追踪、调整合成参数、实时预览。

---

## 2. 状态机重构

### 2.1 旧状态机（v1）

```
SETUP ──▶ TRACKING ──▶ SELECTION ──▶ SAVE
  │          │              │
  └── restart（清空一切）───┘
```

- 单向不可逆
- SELECTION 下无法修改物体
- restart 清空所有数据（包括 inference_state → 重新加载视频帧）

### 2.2 新状态机（v2）

```
EDIT ◀──▶ TRACKING (模态)
  │
  ├── 始终可执行: 加点/删点/增删物体/标记帧/调可见性/单帧预览/保存
  │
  └── [▶ 开始跟踪] → 模态进度条 → 仅追踪脏物体 → 自动回 EDIT
```

**规则：**

| 操作 | EDIT | TRACKING |
|------|------|----------|
| 鼠标加点 | ✅ | ❌ |
| 键盘 N/K/B/V/S/P | ✅ | ❌ |
| tkinter 面板控件 | ✅ | 仅进度条+中止按钮 |
| 退出 | Esc | Esc（中止后退出） |

**`restart` 重定义为"重置全部"**：清 objects、masks、标记帧、背景帧、inference_state。放在面板底部角落，颜色用灰色避免醒目。

**`GUIState` 枚举简化为**：`EDIT = auto()`, `TRACKING = auto()`。移除 `SETUP`, `SELECTION`, `SAVE`。

---

## 3. 物体生命周期

### 3.1 状态流转

```
[N 新建物体] → EMPTY(空心按钮)
                  │
                  ├── 加点 → HAS_POINTS(实心按钮) ── 跟踪 → CLEAN(实心+✓)
                  │                                                │
                  ├── 删物体 → 移除对象+所有关联 mask               │
                  │                                                │
                  └── 再加点 → DIRTY(实心+)── 增量跟踪 → CLEAN
```

### 3.2 规则

1. **启动不自动创建 Obj1**。面板显示 `[+ 新建物体]` 引导按钮 + OpenCV 窗口中央半透明引导文字
2. **seed_frame = 第一个跟踪点的帧号**。后续加点不改变 seed_frame
3. **删除物体**：每个物体按钮右侧 `✕`，点击弹出确认对话框 → 清理 `objects[i]` + `masks[*][obj_id]`
4. **脏标记**：`TrackObject._dirty: bool = True`（新增字段）。未跟踪过的物体 dirty=True；跟踪完成后 dirty=False；再加点后 dirty=True
5. **物体按钮布局**：Grid 2 列，防止超 8 个溢出
6. **颜色**：12 种预定义色 + 超 12 复用颜色但加编号后缀
7. **空物体自动清理**：当空物体切换到另一物体时 → 自动删除空物体（避免累积僵尸）

### 3.3 数据结构变更

```python
@dataclass
class TrackObject:
    obj_id: int
    color: tuple[int, int, int]
    color_hex: str
    name: str
    seed_frame: int = 0
    points: list[tuple[int, int]] = field(default_factory=list)
    vis_start: int | None = None
    vis_end: int | None = None
    _dirty: bool = True   # 新增：是否需要(重新)追踪
```

---

## 4. 增量追踪

### 4.1 追踪逻辑

```python
def _start_tracking(self):
    dirty = [o for o in self.objects if o._dirty and o.points]
    if not dirty:
        self.status_message = "所有物体均已追踪，无需重复。"
        return

    # 首次加载帧（仅一次）
    if self.inference_state is None:
        self.inference_state = self.predictor.init_state(...)
    else:
        self.predictor.reset_state(self.inference_state)

    # 注册所有有点的物体（SAM2 要求传播前注册所有 obj_id）
    for obj in self.objects:
        if obj.points:
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=obj.seed_frame,
                obj_id=obj.obj_id,
                points=np.array(obj.points, dtype=np.float32),
                labels=np.ones(len(obj.points), dtype=np.int32),
            )

    # 传播
    min_seed = min(o.seed_frame for o in self.objects if o.points)
    # ... forward + backward + late-object backward ...

    # 完成后标记 clean
    for obj in dirty:
        obj._dirty = False
```

### 4.2 中止保护

追踪前 snapshot `self.masks` 的深拷贝。Esc 中止时恢复 snapshot → 仅丢弃本次增量产生的 mask，保留已有的。

```
中止时: self.masks = snapshot_masks  # 回滚
```

### 4.3 OOM 恢复

try/except 捕获 → 保留已产生的 mask（不清空）→ 弹窗建议降参数 → 回到 EDIT。与 v1 不同：**不清空已有 mask**，用户可以先保存当前结果。

---

## 5. 单帧预览

### 5.1 行为

- 按 `P` 或点击 `[👁 单帧预览]` → 对 **活跃物体** 在 **当前帧** 做即时分割
- 结果以半透明 mask（0.5 alpha）叠加显示
- 预览 mask **不存入 self.masks**
- 再次按 `P` 或移动帧 → 预览消失
- 如果 inference_state 不存在 → 首次预览自动 init_state

### 5.2 实现方式

利用 SAM2 predictor 的 `_single_frame_predict()` 或最小范围的 `propagate_in_video(start=当前帧, max_frame_num_to_track=1)`，仅追踪当前帧的 1 步。

---

## 6. 帧选择增强

### 6.1 标记帧列表（控制面板新增）

```
┌─ 合成帧 (N) ─────────────────────────────────┐
│  帧42  [✕] [▶]   ●Obj1 ●Obj2 ○Obj3           │
│  帧87  [✕] [▶]   ○Obj1 ●Obj2 ●Obj3           │
│  ...  (可滚动，最多显示 6 行)                   │
│  [一键清除全部]  [◀ 上一标记帧] [下一标记帧 ▶]  │
└────────────────────────────────────────────────┘
```

- 每行：帧号 + ✕ 删除 + ▶ 跳转 + per-object toggle
- 物体 toggle：实心(●)=参与叠加，空心(○)=不参与（改写 `frame_overrides`）
- 双击帧号 → 跳到该帧
- 新标记帧默认所有物体参与叠加

### 6.2 快速跳转

| 操作 | 效果 |
|------|------|
| `Ctrl+←` / `Ctrl+→` | 跳到上一个/下一个标记帧 |
| 列表点击 `[▶]` | 跳到该帧 |

### 6.3 时间线增强

- 高度 40px → 60px
- 分三层：物体可见范围色条（上）+ 标记帧绿条（中）+ 当前位置/背景线（下）
- 标记帧加绿色小三角标记便于肉眼识别
- 图例字体 scale 0.3 → 0.45

---

## 7. Bug 修复清单

| # | 问题 | 修复 |
|---|------|------|
| B1 | vis_range 滑块被 sync 反向覆盖 | Scale 加 `command` 回调实时写 obj；sync 只在 object/state 变更时设初值 |
| B2 | `self.state = GUIState.SETUP` 重复赋值 | 删除冗余行 |
| B3 | restart 不清 `self.objects` | 加 `self.objects.clear()` |
| B4 | interval Scale 改变时不更新帧数标签 | Scale 加 `command` 回调 |
| B5 | 合成预览缓存在拖动可见范围时未失效 | vis 滑块 `command` 直接设 `_preview_dirty = True` |
| B6 | `_render_composite` 无帧缓存 | 增加 `_frame_cache: dict[int, np.ndarray]` |
| B7 | 物体按钮超 8 个溢出 | Grid 2 列布局 |
| B8 | 面板尺寸硬编码 | `geometry` 根据实际控件高度动态调整 |

---

## 8. 用户引导

### 8.1 首次启动覆盖层

无物体时 OpenCV 窗口中央显示半透明引导文字（中文），创建第一个物体并加点后自动消失。

### 8.2 状态消息增强

- 跟踪完成：消息持久 5 秒（`status_timer = 150`）
- 警告类消息用黄色文字
- 错误类消息用红色文字
- 成功类消息用绿色文字

### 8.3 对话框

- 删除物体：`messagebox.askyesno` 确认
- 重置全部：`messagebox.askyesno` 确认
- 保存成功：`messagebox.showinfo` 显示路径+文件大小+帧数
- OOM 错误：`messagebox.showerror` 显示建议参数

---

## 9. 控制面板最终布局

```
┌─ 状态 ─────────────────────────────────────────┐
│  编辑 | 帧 42/300 | 活跃: Obj1 (3点)            │
├─ 物体 ─────────────────────────────────────────┤
│  ■ Obj1 (3点) [✕]    □ Obj2 (空) [✕]          │
│  [+ 新建物体]                                   │
├─ 操作 ─────────────────────────────────────────┤
│  [👁 单帧预览]  [✕ 清除选点]                    │
│  [▶ 开始跟踪]  (灰/绿按脏物体状态)              │
├─ 合成帧 (N) ───────────────────────────────────┤
│  帧42 [✕][▶] ●O1 ●O2 ○O3  ← 滚动区域          │
│  [清空全部] [◀ 上一个] [下一个 ▶]               │
├─ 帧选取 ───────────────────────────────────────┤
│  [✓ 已标记 / K 标记]  [R 范围选择]              │
│  间隔: [===|===] 1.5s  ~30帧  [应用]           │
├─ 可见范围 (Obj1) ──────────────────────────────┤
│  起始: [===|===] 0    结束: [===|===] 299      │
│  [应用范围]  [重置全部帧]                        │
├─ 视图 ─────────────────────────────────────────┤
│  ○ Mask覆盖  ○ 合成预览  ○ 原始帧               │
├─ 快捷键 ───────────────────────────────────────┤
│  ←→ 导航 | Ctrl+←→ 跳标记帧 | P 预览           │
│  N 新建 | 1-9 选物体 | Del 删 | K 标记帧       │
│  B 背景 | V 视图 | O 物体叠加 | S 保存          │
├─ 底部 ─────────────────────────────────────────┤
│  [↺ 重置全部 (灰色)]   [退出 (Esc)]              │
└─────────────────────────────────────────────────┘
```

---

## 10. YAGNI — 明确不做

| 不做 | 理由 |
|------|------|
| 跟踪中后台运行（不阻塞 UI） | SAM2 propagate 是同步 C++ 调用，Python GIL 下无真正异步 |
| 逐帧手动修正 mask（画笔） | 复杂度极高，属于不同产品方向 |
| 导出视频/GIF | 当前目标定位为单张合成图 |
| 撤销/重做（Ctrl+Z） | 30%+ 代码增量，边际收益有限 |
| 保存/加载项目文件 | 属于后续版本考虑 |
| 深色主题 | tkinter clam 主题已是浅色，改深色需大量样式覆盖 |
| 多语言支持 | 当前用户中文即可 |
| 鼠标悬停时间线显示帧号 | OpenCV 无原生 hover 事件，实现复杂 |

---

## 11. 验证方法

```bash
# 基础功能
uv run python scripts/stroboscopic_gui.py --video video/test.MP4 --process-fps 15

# 低显存场景
uv run python scripts/stroboscopic_gui.py --video video/test.MP4 --process-fps 5 --max-dim 720
```

**验证清单：**
- [ ] 首次启动显示引导覆盖层，创建物体+加点后消失
- [ ] N 新建物体，Del 删除物体，物体按钮状态正确
- [ ] P 单帧预览显示半透明 mask，移动帧后消失
- [ ] 增量追踪只处理脏物体，已有 mask 保持不变
- [ ] Esc 中止追踪保留已有 mask
- [ ] 标记帧列表实时更新，per-object toggle 保存到合成图
- [ ] Ctrl+←/→ 跳转标记帧
- [ ] vis_range 滑块可正常拖动，实时更新预览
- [ ] K/I/R 三种选帧方式正常
- [ ] B 设置背景，V 切换视图，S 保存
- [ ] 重启清空所有状态
- [ ] OOM 时保留已产生 mask 并回到 EDIT
