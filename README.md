# 打磨质量视觉评估 · OpenCV C++

生产运行只需 OpenCV、yaml-cpp 和 nlohmann-json，不需要 Python、PyTorch 或训练模型。当前实现是颜色/纹理/曝光规则基线，不是迁移或训练了一个深度模型。结果是候选判断，不是工业验收结论。

## 编译与运行

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/grinding_quality analyze --config config/quality.yaml
./build/grinding_quality analyze --config config/quality.yaml --input data/images/sample01.jpg --output outputs/single
./build/grinding_quality camera --config config/quality.yaml
```

Ubuntu 开发依赖为 libopencv-dev、libyaml-cpp-dev、nlohmann-json3-dev。当前 Jetson 同时有多个 OpenCV，已用 `/usr/lib/cmake/opencv4` 下的 4.8.0 编译；若默认选择错误，配置时添加 `-DOpenCV_DIR=/usr/lib/cmake/opencv4`。不要无故替换系统的 Python OpenCV。

## 目录与参数

```text
src/main.cpp           离线/相机/常驻 worker 三种入口
src/quality.cpp        规则分割、统计、热力图和结果保存
include/quality.hpp    可供其他 C++ 项目调用的接口
config/quality.yaml    图像、ROI、规则及评价阈值
data/images/           四张原始测试图
tests/compare_python.py 迁移一致性验证，非生产运行依赖
tools/python_reference/ 原 Python 参考实现，仅用于比较验证
outputs/               图片、逐网格 JSON、summary.csv、report.html
```

独立运行时，默认 input/output 路径相对于仓库根目录（配置放在 config/ 下）；显式 CLI --input/--output 相对于当前终端目录。配置参数含义延续原项目，阈值没有改动。`assessment.grid_px` 是分析图上的网格边长，越小越细，目前保留 40；`max_side` 控制分析分辨率。`roi` 为归一化坐标，离线四张样本可用各自 override。实时总调度覆盖 ROI，避免带入离线样本规则。

三类掩膜：0=裸露金属候选、1=残留候选、2=不确定，255=ROI 外。未知区域不按干净处理。JSON 中残留连通区域坐标仍是原图像素；仅在平面标定后设置 mm_per_pixel_original，不能直接当作机器人回位坐标。

颜色规则联合 Lab 中亮度/色度、HSV 锈色特征、结构张量方向一致性与曝光屏蔽。C++ 省略规则法不使用的 RF 特征堆叠，保留实际判定公式。仅凭金属光泽不能测去除厚度、粗糙度或过磨。

## 统一调度

```bash
cd /home/tjy/grinding_robot_suite
python3 run.py --demo --cycles 4
python3 run.py
```

总入口依然是轻量 Python 调度，图像分析使用 C++ 常驻子进程：`build/grinding_quality worker --worker-config <JSON>`。总入口写入 JSON 行任务，C++ 返回相同任务 ID 的 done.json；每次任务有独立结果目录。不会打开第二个相机，也不写机器人串口。

原质量项目 Python 标注/训练/ONNX 等工具已退出生产目录并归档。原 baseline 代码仅放 tools/python_reference 供回归测试，生产不导入它。归档位置与逐文件清单见 CLEANUP.md；Git 历史、数据集及各 TensorRT 运行引擎保留。

## 验证

`python3 tests/compare_python.py` 需要 Python 的 OpenCV、NumPy、PyYAML（只作测试），在四张样本的 640/1280 两种分辨率比较掩膜与统计，还检查常驻任务、读图失败及过曝/暗帧/空有效 ROI。测试会更新 outputs/cpp_validation。

原始四张实拍样本未提交到 Git；在新设备上运行该测试或总调度四图 demo 前，需要自行放入 `data/images/sample01.jpg` 至 `sample04.jpg`。只安装 C++ 生产依赖即可运行评估，Python 参考依赖不是生产必需。

2026-11-06 Jetson 首轮八组比较掩膜差异均为零，指标一致。640 分辨率下第二轮常驻任务的分析中位耗时：C++ 45.75 ms，Python 103.26 ms；该值不含读图/可视化/保存，且仅代表这台设备和这四张图片，不代表算法准确率提高。
