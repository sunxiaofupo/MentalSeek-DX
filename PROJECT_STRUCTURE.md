# Project Structure

```
MentalSeek-Dx/
├── README.md                                    # 项目主文档
├── requirements.txt                             # Python 依赖包列表
├── .gitignore                                   # Git 忽略文件配置
├── run_MentalSeek-Dx.py                         # 主运行脚本
│
├── Reasoning_Trajectory_Building/               # 推理轨迹构建模块
│   └── prompts.py                               # 提示词模板和推理轨迹构建逻辑
│
├── Structured_Psychiatric_Knowledge_Base/       # 结构化精神病学知识库
│   ├── Criteria_part.jsonl                      # 诊断标准部分数据
│   └── Diagnostic_category_part.jsonl           # 诊断类别部分数据
│
├── MentalDx-Bench/                             # 基准测试数据集和评估工具
│   ├── Ground_Truth.jsonl                       # 真实标签数据
│   ├── MentalDx-Bench.jsonl                     # 基准测试数据集
│   └── src/                                     # 评估源代码
│       ├── extract_and_evaluate.py              # 结果提取和评估脚本
│       └── statistics_category_accuracy.py     # 分类准确率统计脚本
│
├── Models/                                      # 模型配置和说明
│   ├── MentalSeek-Dx-14B/
│   │   └── models.md                            # 14B 模型说明文档
│   └── MentalSeek-Dx-7B/
│       └── models.md                            # 7B 模型说明文档
│
├── eval_results/                                # 评估结果目录
│   ├── extracted_result/                        # 提取的评估结果
│   │   ├── MentalSeek-Dx-14B.jsonl             # 14B 模型评估结果
│   │   └── MentalSeek-Dx-7B.jsonl              # 7B 模型评估结果
│   ├── prediction_output/                      # 预测输出结果
│   │   ├── MentalSeek-Dx-14B.jsonl             # 14B 模型预测输出
│   │   └── MentalSeek-Dx-7B.jsonl              # 7B 模型预测输出
│   └── result_statistics.txt                    # 结果统计汇总文件
│
└── verl/                                        # VERL 框架（第三方强化学习框架）
    ├── verl/                                    # VERL 核心代码
    │   ├── checkpoint_engine/                   # 检查点引擎
    │   ├── experimental/                        # 实验性功能
    │   ├── interactions/                        # 交互模块
    │   ├── model_merger/                        # 模型合并工具
    │   ├── models/                              # 模型定义
    │   ├── single_controller/                   # 单控制器
    │   ├── third_party/                         # 第三方依赖
    │   ├── tools/                               # 工具函数
    │   ├── trainer/                             # 训练器
    │   ├── utils/                               # 工具模块
    │   ├── version/                             # 版本信息
    │   ├── workers/                             # 工作节点
    │   └── ...
    ├── examples/                                 # 示例脚本
    │   ├── cispo_trainer/                       # CISPO 训练器示例
    │   ├── data_preprocess/                     # 数据预处理示例
    │   ├── generation/                          # 生成示例
    │   ├── gmpo_trainer/                        # GMPO 训练器示例
    │   ├── gpg_trainer/                         # GPG 训练器示例
    │   ├── grpo_trainer/                        # GRPO 训练器示例
    │   ├── gspo_trainer/                         # GSPO 训练器示例
    │   ├── mtp_trainer/                         # MTP 训练器示例
    │   ├── otb_trainer/                          # OTB 训练器示例
    │   ├── ppo_trainer/                         # PPO 训练器示例
    │   └── ...
    ├── docs/                                     # 文档目录
    ├── docker/                                   # Docker 配置文件
    ├── scripts/                                  # 工具脚本
    ├── tests/                                    # 测试代码
    ├── recipe/                                   # 配方/配置
    │   └── MentalSeek-Dx/                        # MentalSeek-Dx 相关配置
    ├── requirements.txt                         # VERL 依赖
    ├── requirements-cuda.txt                    # CUDA 相关依赖
    ├── requirements-npu.txt                     # NPU 相关依赖
    ├── setup.py                                 # 安装脚本
    └── README.md                                # VERL 框架说明
```

## 目录说明

### 核心模块

- **`run_MentalSeek-Dx.py`**: 项目主入口脚本，用于运行 MentalSeek-Dx 模型
- **`Reasoning_Trajectory_Building/`**: 包含推理轨迹构建的核心逻辑和提示词模板
- **`Structured_Psychiatric_Knowledge_Base/`**: 存储结构化的精神病学知识库数据（JSONL 格式）

### 评估模块

- **`MentalDx-Bench/`**: 包含基准测试数据集和评估工具
  - `Ground_Truth.jsonl`: 真实诊断标签
  - `MentalDx-Bench.jsonl`: 完整的基准测试数据集（712 条记录）
  - `src/`: 评估脚本源代码

### 结果输出

- **`eval_results/`**: 存储模型评估结果
  - `extracted_result/`: 提取的评估结果（JSONL 格式）
  - `prediction_output/`: 模型预测输出
  - `result_statistics.txt`: 统计汇总

### 模型配置

- **`Models/`**: 包含不同规模模型的配置和说明文档
  - `MentalSeek-Dx-14B/`: 14B 参数模型
  - `MentalSeek-Dx-7B/`: 7B 参数模型

### 第三方框架

- **`verl/`**: VERL（Versatile Reinforcement Learning）框架
  - 用于强化学习训练的基础框架
  - 包含训练器、工作节点、工具函数等完整实现
  - 支持多种训练算法（PPO, GRPO, GPG 等）

## 主要文件类型

- **`.py`**: Python 源代码文件
- **`.jsonl`**: JSON Lines 格式数据文件（每行一个 JSON 对象）
- **`.md`**: Markdown 文档文件
- **`.txt`**: 文本文件（依赖列表、统计结果等）
- **`.yaml/.yml`**: YAML 配置文件
- **`.sh`**: Shell 脚本文件
