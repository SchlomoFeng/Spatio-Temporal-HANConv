# S4蒸汽管网智能异常检测与定位系统

基于时空异构图神经网络（Spatio-Temporal HANConv）的S4蒸汽管网智能异常检测与定位系统，能够实时监控管网运行状态、自动检测异常并精确定位异常源头。

## 🎯 项目特点

- **实时异常检测**: 基于时间序列数据的实时异常监控
- **根本原因分析**: 精确定位引起异常的"流股"节点
- **异构图建模**: 考虑不同类型节点（流股、阀门、混合器、三通）的复杂关系
- **时空融合**: 结合LSTM时序建模和图神经网络空间建模
- **可视化分析**: 直观展示异常位置和影响范围

## 🏗️ 系统架构

### 数据层
- **管网拓扑**: 从`blueprint/0708YTS4.txt`解析管网结构
- **传感器数据**: 从`data/0708YTS4.csv`读取实时传感器数据
- **数据预处理**: 清洗、归一化、时间窗口构建

### 模型层
- **LSTM编码器**: 处理流股节点的时间序列数据
- **线性编码器**: 处理静态节点（阀门、混合器、三通）特征
- **异构图卷积**: HANConv/HGTConv处理节点间复杂关系
- **MLP解码器**: 重构传感器读数

### 应用层
- **训练模块**: 模型训练和验证
- **推理模块**: 实时/批量异常检测
- **可视化模块**: 异常定位和报告生成

## 📦 环境配置

### 系统要求
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (可选，GPU加速)

### 快速安装

**CPU版本（默认）：**
```bash
# 克隆项目
git clone https://github.com/SchlomoFeng/Spatio-Temporal-HANConv.git
cd Spatio-Temporal-HANConv

# 安装依赖
pip install -r requirements.txt
```

**GPU版本（推荐）：**
```bash
# 安装CUDA支持的PyTorch
pip install -r requirements_cuda.txt
```

### 详细安装指南

查看 [INSTALLATION.md](INSTALLATION.md) 获取：
- 完整的安装说明
- GPU环境配置
- 故障排除指南
- 性能优化建议

### 环境验证
```bash
# 验证安装
python main.py --mode validate

# 检查CUDA支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 依赖包说明
- `torch`: 深度学习框架
- `torch-geometric`: 图神经网络库
- `pandas`: 数据处理
- `numpy`: 数值计算
- `scikit-learn`: 数据预处理和评估
- `matplotlib`, `seaborn`: 可视化
- `networkx`: 图结构处理
- `tqdm`: 进度条
- `tensorboard`: 训练监控

## 🚀 快速开始

### 1. 数据验证
```bash
# 验证数据完整性和格式
python main.py --mode validate
```

### 2. 模型训练
```bash
# 开始训练
python main.py --mode train

# 从检查点恢复训练
python main.py --mode train --resume checkpoints/last_checkpoint.pth
```

### 3. 异常检测
```bash
# 批量检测（推荐）
python main.py --mode detect --detection-mode batch

# 实时检测
python main.py --mode detect --detection-mode real-time
```

### 4. 网络可视化
```bash
# 可视化管网拓扑
python main.py --mode visualize
```

## 📁 项目结构

```
Spatio-Temporal-HANConv/
├── main.py                    # 主入口脚本
├── requirements.txt           # 依赖包列表
├── config/
│   └── config.yaml           # 配置文件
├── src/
│   ├── data/
│   │   ├── preprocessing.py  # 数据预处理
│   │   └── dataset.py        # 数据集和数据加载器
│   ├── models/
│   │   └── han_autoencoder.py # 模型定义
│   ├── training/
│   │   └── train.py          # 训练脚本
│   ├── inference/
│   │   └── anomaly_detection.py # 异常检测
│   ├── utils/                # 工具函数
│   └── visualization/        # 可视化工具
├── blueprint/
│   ├── 0708YTS4.txt         # 管网拓扑数据（JSON格式）
│   └── GraphPlot_0708YTS4.py # 拓扑可视化脚本
├── data/
│   └── 0708YTS4.csv         # 传感器时间序列数据
├── checkpoints/             # 模型检查点
├── logs/                    # 训练日志
└── visualizations/          # 可视化输出
```

## ⚙️ 配置说明

### 数据配置 (`config/config.yaml`)
```yaml
data:
  blueprint_path: "blueprint/0708YTS4.txt"  # 拓扑文件路径
  sensor_data_path: "data/0708YTS4.csv"    # 传感器数据路径
  window_size: 60                          # LSTM时间窗口大小
  stride: 10                               # 滑动窗口步长
  train_ratio: 0.7                         # 训练集比例
  val_ratio: 0.15                          # 验证集比例
  test_ratio: 0.15                         # 测试集比例
```

### 模型配置
```yaml
model:
  stream_input_dim: 36      # 传感器数量
  static_input_dim: 10      # 静态节点特征维度
  hidden_dim: 128           # 隐藏层维度
  lstm:
    hidden_size: 64         # LSTM隐藏大小
    num_layers: 2           # LSTM层数
    bidirectional: true     # 双向LSTM
  hetero_conv:
    type: "HANConv"         # 图卷积类型
    num_layers: 3           # 图卷积层数
    heads: 4                # 注意力头数
```

### 训练配置
```yaml
training:
  batch_size: 32
  epochs: 200
  learning_rate: 0.001
  patience: 20              # 早停耐心值
  loss_function: "MSELoss"
```

### 异常检测配置
```yaml
anomaly:
  threshold_method: "percentile"  # 阈值计算方法
  threshold_percentile: 95        # 百分位阈值
  top_k_anomalies: 5             # 报告前K个异常节点
```

## 🔍 数据格式

### 管网拓扑数据 (`blueprint/0708YTS4.txt`)
JSON格式，包含节点和边信息：
```json
{
  "nodelist": [
    {
      "id": "node_uuid",
      "name": "节点名称",
      "parameter": "{...节点参数...}"
    }
  ],
  "linklist": [
    {
      "id": "edge_uuid",
      "sourceid": "源节点ID",
      "targetid": "目标节点ID",
      "parameter": "{...边参数...}"
    }
  ]
}
```

### 传感器数据 (`data/0708YTS4.csv`)
时间序列格式：
```csv
timestamp,YT.63PI_00406.PV,YT.63FI_00406.PV,...
2025/7/1 12:00:00,452.97,21.72,...
2025/7/1 12:00:10,453.58,21.84,...
```

## 📊 模型性能

### 网络规模
- **节点数量**: 209个（流股84个，三通68个，阀门36个，混合器21个）
- **边数量**: 206条连接
- **传感器数量**: 36个
- **时间跨度**: 约6天，51,841条记录

### 模型参数
- **总参数量**: ~565K
- **训练时间**: 约2-4小时（GPU）
- **推理速度**: <1秒/样本

## 🎨 可视化功能

### 1. 管网拓扑可视化
- 节点类型区分（颜色和形状）
- 真实坐标系统
- 连接关系展示

### 2. 异常检测可视化
- 异常节点高亮标注
- 异常分数热力图
- 传感器重构误差分析
- Top-K异常节点定位

### 3. 训练监控
- 损失函数曲线
- 学习率变化
- 验证指标跟踪

## 🔧 高级用法

### 自定义模型配置
```python
# 修改模型架构
config['model']['hetero_conv']['type'] = 'HGTConv'  # 使用HGT卷积
config['model']['lstm']['num_layers'] = 3           # 增加LSTM层数
```

### 实时监控部署
```python
from src.inference.anomaly_detection import AnomalyDetector

# 初始化检测器
detector = AnomalyDetector(config, model_path, threshold_path)

# 实时检测
while True:
    sensor_data = get_real_time_data()  # 获取实时数据
    is_anomaly, score, details = detector.detect_anomaly(sensor_data)
    
    if is_anomaly:
        alert_operators(details)  # 发送警报
```

### 批量历史分析
```python
from src.inference.anomaly_detection import BatchAnomalyDetector

# 批量分析
batch_detector = BatchAnomalyDetector(config, model_path, threshold_path)
results = batch_detector.detect_anomalies_in_dataset(historical_dataset)
```

## 🐛 故障排除

### 常见问题

**1. CUDA支持问题**
```bash
# 错误：Torch not compiled with CUDA support
# 解决：安装CUDA版本的PyTorch
pip install -r requirements_cuda.txt

# 检查环境
python src/utils/device_utils.py
```

**2. 设备配置错误**
```yaml
# 在config/config.yaml中设置设备
system:
  device: "auto"  # "auto", "cpu", "cuda", "cuda:0"
```

**3. 内存不足**
```bash
# 检查文件路径和格式
python main.py --mode validate
```

**4. 内存不足**
```yaml
# 减小批量大小和窗口大小
training:
  batch_size: 16
data:
  window_size: 30
```

**5. GPU内存不足**
```yaml
# 使用CPU训练
system:
  device: "cpu"
```

**6. 收敛慢或不收敛**
```yaml
# 调整学习率和优化器
training:
  learning_rate: 0.0001
  optimizer: "AdamW"
```

### 环境检查工具
```bash
# 详细环境信息
python src/utils/device_utils.py

# 配置验证
python src/utils/config_validator.py config/config.yaml
```

### 日志查看
```bash
# 查看训练日志
tail -f logs/training_*.log

# 启动Tensorboard
tensorboard --logdir logs/tensorboard
```

## 📈 性能优化

### 训练优化
- 使用GPU加速：设置`device: "cuda"`
- 混合精度训练：减少内存使用
- 数据并行：多GPU训练支持
- 梯度累积：模拟大批量训练

### 推理优化
- 模型量化：减少模型大小
- 批量推理：提高吞吐量
- 异步处理：实时系统响应
- 缓存机制：减少重复计算

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python -m pytest tests/

# 代码格式化
black src/
isort src/
```

### 提交规范
- feat: 新功能
- fix: 修复
- docs: 文档
- test: 测试
- refactor: 重构

## 📄 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。

## 📞 联系方式

- 项目维护者: SchlomoFeng
- 邮箱: [your-email@example.com]
- 问题反馈: [GitHub Issues](https://github.com/SchlomoFeng/Spatio-Temporal-HANConv/issues)

## 🎯 技术路线图

### v1.0 (当前版本)
- [x] 基础异常检测功能
- [x] 图神经网络建模
- [x] 实时推理支持
- [x] 可视化系统

### v1.1 (计划中)
- [ ] 模型解释性增强
- [ ] 多模态数据融合
- [ ] 在线学习支持
- [ ] Web界面开发

### v2.0 (未来版本)
- [ ] 分布式训练
- [ ] 模型压缩优化
- [ ] 边缘计算部署
- [ ] 预测性维护

---

**如果这个项目对您有帮助，请给我们一个⭐️！**
