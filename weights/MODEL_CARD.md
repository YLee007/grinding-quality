# 权重说明

`mobilenet_v2-b0353104.pth` 为 PyTorch 官方 ImageNet 预训练 MobileNetV2 编码器权重，已下载并校验 SHA256。

- 下载源：https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
- SHA256：`b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889`
- 用途：初始化轻量 U-Net 编码器；解码器需用项目数据训练。
- 当前状态：**没有经过打磨质量任务训练，不可直接用于判断打磨合格。**

`quality_rf.joblib`、`quality_unet.pt`、`quality_unet.onnx` 需要完成实际标注、划分和训练后生成，目前没有提供虚假的现场模型。

软件自检使用临时合成数据，测试结束后临时权重会自动清理。`outputs/self_test.json` 中的通过状态代表软件流程检查，不代表质量识别精度。
