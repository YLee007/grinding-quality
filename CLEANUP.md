# C++ 迁移与清理说明

本仓库现采用 Jetson 原生 C++ 实现；正式部署位于 `/home/tjy/grinding-quality`。本轮以 Jetson 为发布源，未同步覆盖 Windows 原业务仓库。

质量评估现以 C++/OpenCV 运行。原 Python 规则实现放在 Jetson 的 `tools/python_reference/` 供回归比较，旧训练/标注工具等已从活动项目移至可恢复备份。

两个项目的清理记录与恢复方式：`/home/tjy/grinding_robot_suite/CLEANUP.md`；逐项清单：`/home/tjy/grinding_robot_suite/cleanup_manifest.json`。可恢复归档：`/home/tjy/backups/robot_cleanup_20261106_140730`。

共归档 22 项约 5.78 GB，没有永久删除，也未释放磁盘空间。保留当前运行引擎、重建用 ONNX、原始样本、配置及 Git 历史。导出模型前如需旧 Python 环境，应从归档按清单恢复。
