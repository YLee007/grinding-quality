# 打磨质量视觉评估

已提供可直接运行的离线分析、USB 相机预览、涂画标注、随机森林训练、MobileNetV2 U-Net 训练、PyTorch/ONNX 推理和 ONNX 导出。首轮已分析用户提供的四张照片。当前有效输出是**颜色与纹理规则的候选结果**；尚未用人工标注训练现场质量模型，不能当作验收结论。

## 原理与技术路线

本项目把打磨质量评估建模为像素级三分类：`裸露金属`、`残锈/氧化层`和`不确定`。相机先在限定 ROI 内观察钢板，算法再生成分割掩膜、残留面积率、网格热力图和待返工区域坐标。这里的“金属光泽”不是只看亮度：规则基线联合 Lab/HSV 颜色、局部对比度、纹理与有效曝光判断，避免把阴影简单当成锈蚀、把高光简单当成合格。

技术路线分为三个层级：

1. **规则基线**：无需训练，快速验证固定光源、曝光、ROI 和评价指标是否合理，适合采集早期发现问题。
2. **随机森林**：用少量人工涂画像素学习现场颜色与纹理分布，训练成本低，可作为小数据基线。
3. **MobileNetV2 U-Net**：用人工语义分割标签学习复杂边界和光照变化；支持滑窗推理、ONNX 导出，并可进一步在 Jetson 上转换 TensorRT。

最终判定不是单张图的平均亮度，而是综合残留面积率、可评价面积、局部连续残留区域及网格一致性。所有阈值都必须用同材质、同照明、同工艺下的合格/不合格样板标定。相机应尽量固定距离、角度、曝光和色温；在自然光强反射条件下，建议增加漫射照明或交叉偏振。当前输出属于机器视觉候选结论，尚不能替代粗糙度、除锈等级等正式工业验收。

## 1. 先看结果，再调配置

直接打开 `outputs/baseline/report.html`。每张图提供原图缩略、区域分割、网格热力图及 JSON 数据。

配色：绿色=裸露金属候选；红色=残锈/深色残留候选；黄色=不确定；暗灰=检测范围外。红色里可能包含阴影等误判。绿色也需要人工核对。

所有配置都在 `config/quality.yaml`，相对路径均以本工程目录为基准，不依赖终端当前路径。

```powershell
git clone https://github.com/YLee007/grinding-quality.git
cd grinding-quality
python bootstrap.py
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.cmd analyze
```

已提供 `run.cmd` 入口，本机 PowerShell 禁止执行 `.ps1` 也可以用。还可直接执行下列等效命令，无需修改系统执行策略：

```powershell
.\.venv\Scripts\python.exe quality.py analyze --config config/quality.yaml
```

只测一张图片：

```powershell
.\.venv\Scripts\python.exe quality.py analyze --input data/images/sample01.jpg
```

离线 `analyze` 永远不打开相机。只有显式执行 `camera` 才打开。

### 关键参数

| 参数                                            | 用途                                                                          |
| ----------------------------------------------- | ----------------------------------------------------------------------------- |
| `method`                                      | `baseline`、`rf`、`unet` 或 `onnx`                                    |
| `input` / `output`                          | 输入图片/目录、输出目录；同目录再次运行会更新派生结果，想保留比较请换输出目录 |
| `max_side`                                    | 分析时最长边；默认 1280，保留原图长宽比；纹理特征受这个值影响                 |
| `roi.default`                                 | 默认`[左,上,右,下]`，坐标归一化到 0～1                                      |
| `roi.overrides`                               | 按文件名设定检测范围；示例已排除明显的焊缝边缘/地板                           |
| `roi.exclude_rects`                           | 排除异物等；图4对可见异物设有示例排除框                                       |
| `baseline.metal_l_min`                        | 越高，对金属候选的亮度要求越严格；Lab 的 L 为 OpenCV 0～255 尺度              |
| `baseline.metal_chroma_max`                   | 越低，越严格地要求接近无彩色                                                  |
| `baseline.dark_l_max`                         | 深色残留的亮度阈值；调高会增加暗部残留候选，也可能增加阴影误判                |
| `baseline.min_evidence` / `min_margin`      | 越高，不确定区域越多；这是规则分值，不是统计概率                              |
| `assessment.grid_px`                          | 分析图上的网格边长，默认 80 像素；未标定前不是毫米                            |
| `assessment.max_residual_fraction`            | 示例为 0.05，**不是工业标准**；需要用认可样板确定                       |
| `assessment.min_evaluable_fraction`           | 默认 0.85；可评价面积不足时不能输出清洁候选                                   |
| `assessment.min_component_area_px`            | 只控制 JSON 中列出的连通域大小；小残留仍计入面积率                            |
| `assessment.mm_per_pixel_original`            | 默认 null；仅在实际标定后的近似正视平面上填写，不可猜测                       |
| `rf.min_confidence` / `unet.min_confidence` | 分类分数不足时标黄；分数尚未做概率校准                                        |

ROI 目前是“可见钢板内部”的探索范围，并不代表生产中确实需要磨到的范围。若只需打磨焊缝两侧某宽度，必须先换成对应工艺区域。

## 2. 现有结果的含义

`summary.csv` 可用 Excel 打开；单张 `.json` 保存细节：

- `residual_fraction_roi_lower_bound`：已识别残留面积 / 全部 ROI，未知区域不计为干净；它不是残留真值。
- `residual_fraction_evaluable`：已识别残留 /（金属候选＋残留候选）。未知很多时不可单看这个比例。
- `evaluable_fraction`、`unknown_fraction`：可评价比例及待复核比例。
- `cells`：各网格指标和候选判断。
- `residual_components`：达到报告面积阈值的残留连通域、原图包围框、质心、面积。
- `box_xyxy_original`、`centroid_xy_original`：**原图像素坐标，不是机器人坐标**。

掩膜 `*_mask.png` 的值为 `0=金属、1=残留、2=不确定、255=ROI外`，尺寸是分析图尺寸，需结合 JSON 中的原图尺寸映射。不要把预测掩膜直接作为人工真值训练。

照片首轮有较多黄色区，这是自然光、反射方向和深色表面歧义造成的。首轮不能凭某张图的残留率较低，就说它打磨得更好。程序目前只自动屏蔽极暗/过曝像素；模糊、粉尘、复杂遮挡还需要固定成像条件及人工复核，未宣称具备完整图像可用性检测。

## 3. 用少量涂画训练随机森林（建议先做）

```powershell
.\run.cmd annotate
```

打开图片后，左键涂画；鼠标绘制结果保存为原始图片尺寸的单通道 PNG。

| 操作            | 功能                                 |
| --------------- | ------------------------------------ |
| `1`           | 涂裸露金属（绿色，标签 0）           |
| `2`           | 涂确认的残锈/氧化层（红色，标签 1）  |
| `0`           | 橡皮/忽略（标签 255）                |
| `[` / `]`   | 缩小/增大画笔                        |
| `Ctrl+Z`      | 撤销最近一次笔画，最多10次           |
| `S`           | 保存；覆盖旧标注前保留带时间戳的备份 |
| `N` / `P`   | 保存并切换下一张/上一张              |
| `Q` / `Esc` | 保存并退出                           |

未涂画区域是忽略，**不是默认金属**。对于不确定的深色表面先不涂。要同时选明暗不同的金属、颜色不同的残锈，避免分类器只学亮度。

标签输出在 `data/annotations`。参考 `data/split.example.csv` 新建 `data/split.csv`：

```csv
image,mask,split,group
data/images/plate_A_01.jpg,data/annotations/plate_A_01.png,train,plate_A
data/images/plate_A_02.jpg,data/annotations/plate_A_02.png,train,plate_A
data/images/plate_B_01.jpg,data/annotations/plate_B_01.png,val,plate_B
```

`group` 是真实试件/采集批次标识。**同一块板及其相邻帧必须处于同一划分**。原来的四张照片可能有相同试件的不同视角，因此没有替你伪造四个独立组。需要确认真实归属，补充独立试件作为验证集。代码会拒绝已知同组或完全相同图片跨训练/验证集。

```powershell
.\run.cmd train-rf
```

得到 `weights/quality_rf.joblib` 和验证统计 JSON。然后在 YAML 设置 `method: rf`、`output: outputs/rf`，再次运行 `analyze`。

RF 使用颜色和局部纹理，训练计算量小，可先验证是否已满足固定光照下的需求。模型与训练时的 `max_side` 绑定；改变分析尺度后应重新训练。只加载本工程自己生成或可信来源的 joblib 文件。

## 4. 训练轻量 U-Net

当固定成像条件＋RF 对现场变化仍不够稳健时，使用更完整、覆盖多样工况的标注，保留独立试件验证。稀疏涂画可用于初步试验，正式评价最好完整标注独立验证图。

```powershell
.\run.cmd train-unet
```

网络为 MobileNetV2 编码器＋轻量 U-Net 解码器，共约 413 万参数，输出金属/残留两类；低分值在后处理中转为不确定。输入使用 RGB、ImageNet mean/std 归一化。默认训练 512×512 图块，随机翻转和轻微亮度增强，忽略标签255。

训练参数仍在 YAML：`training.epochs`、`batch_size`、`learning_rate`。默认使用本机 CUDA；可将 `unet.device` 改为 `cpu`。

训练保存独立验证集 mIoU 最佳权重 `weights/quality_unet.pt`，以及 `.history.json`。换 `method: unet`、`output: outputs/unet` 后可推理。大图按重叠图块推理，不会把整张长钢板强行压成正方形。

训练入口遇到已存在的目标模型会拒绝覆盖；继续新实验请在 YAML 更换 checkpoint 路径。训练没有自动复训/后台启动，标注不足时会明确报错。

**`mobilenet_v2-b0353104.pth` 仅为 ImageNet 预训练编码器，不具备现成的打磨质量判断能力。当前没有把随机解码器或合成测试模型冒充可用现场权重。**

## 5. 导出 ONNX / 后续 Jetson

真实质量模型训练完成后：

```powershell
.\run.cmd export
```

导出固定输入 `1×3×512×512`、opset17 的 ONNX，并使用 ONNX Runtime 对比 PyTorch 数值。产生 `quality_unet.onnx` 和对应元数据 JSON。修改 `method: onnx` 后可在当前电脑 CPU 上验证，输入/输出处理与 PyTorch 一致。

后续在 Jetson 本机用与该设备 TensorRT 版本相匹配的 `trtexec` 构建引擎，示意命令：

```bash
trtexec --onnx=weights/quality_unet.onnx --saveEngine=weights/quality_unet_fp16.engine --fp16
```

这里没有连接 Jetson，也没有宣称 TensorRT 引擎或 C++ 实时质量检测已经完成。迁移时需测导出算子、FP16误差、显存与端到端时延，并接入现有 C++ 图像流；最终可将本工程 YAML 的质量参数合并到既有 `weld.yaml`。

## 6. 相机预览

在 YAML 调整 `camera.id`、宽高，以及 `show_window: true`：

```powershell
.\run.cmd camera
```

`camera.analyze_every_n_frames` 控制每多少采集帧分析一次；`save_every_n_results` 控制落盘间隔。默认每5采集帧分析一次，每10次分析保存一组，避免无限速写图。循环在同一线程执行，处理期间可能存在相机缓冲延迟；目前不是经时延校准的实时机器人闭环。

终端输出候选状态、面积率、处理耗时及主机接收帧后的时间戳。这里的时间不是硬件曝光同步时间。相机模式只做预览记录，不发送任何运动/继电器指令。程序默认未打开相机窗口，需显式配置。

曝光、增益、白平衡先用相机工具锁定；不同厂商参数接口不一致，当前程序不会声称已成功锁定。检测相机应观察滚刷后的表面。位姿配准、跨帧地图融合和回位补磨属于后续集成步骤，目前不具备。

## 7. 源码、环境和验证

- 上游：[segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)，v0.5.0，commit `420ce84b0c2df0286fa9bb2bd1499eea625c9b33`，MIT许可证在源码目录。
- 权重：[PyTorch官方 MobileNetV2](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)，SHA256 `b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889`。
- 数据来源/哈希：`data/provenance.json`，保留原文件且没有修改原始照片。
- `.venv` 继承本机 `D:\python\envs\vggt` 已安装的 Torch 2.3.1 / torchvision 0.18.1，在自己的目录安装固定版本的 NumPy/OpenCV/SMP/sklearn 等。没有卸载或更改原环境。此环境不是可直接搬到另一电脑或 Jetson 的便携环境。
- 项目依赖版本见 `requirements.txt`。其他机器应先安装匹配设备的 PyTorch/torchvision，再创建本工程环境并安装依赖。
- `bootstrap.py` 可校验/下载通用权重和复制示例原图；原图来源路径写在脚本中。

```powershell
.\run.cmd self-test
```

自检覆盖无有效区域/全不确定不能判清洁、连通域尺度换算、小残留统计、训练验证分组、随机森林训练推理、GPU前向反向、U-Net完整训练入口和ONNX数值一致性。使用合成数据的流程测试**不是**现场精度。测试模型只存在临时目录，不会写入正式 `weights`。

文件分工：`quality.py` 命令/报告/相机；`quality_core.py` 特征/基线/指标；`quality_models.py` RF/U-Net/ONNX；`quality_annotate.py` 标注；`quality_selftest.py` 流程自检。

## 8. 下一阶段的顺序

1. 检查当前四图的绿色误判、红色误判，以及 ROI 是否符合真实工艺范围。
2. 固定相机/补光，采集未磨、欠磨、认可合格样板，保留独立试件。
3. 涂画标注，先训练 RF；统计残留漏检而不只看总体准确率。
4. 必要时完整标注更多图，训练 U-Net，测试不同试件与不同工况。
5. Jetson TensorRT/C++ 部署，建立相机到工件坐标映射和时序对齐。
6. 最后加入缺陷地图、返程定位、补磨与复检；毫米定位不能由像素质心直接替代。
