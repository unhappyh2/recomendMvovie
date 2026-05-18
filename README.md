# Breeze — 基于 DiffuRec 的电影推荐系统

基于 DiffuRec 思路的电影推荐系统，RecBole 框架实现训练与评估，Flask Web 交互保持不变。当前主模型已替换为序列扩散推荐模型，类名保留为 `LightGCNDiffusion` 以兼容原训练入口和 Web 侧 checkpoint 读取。

## 项目结构

```
recomendMvovie/
├── config.yaml                    # 模型/训练超参数配置
├── setup.bat / setup.sh           # 一键环境安装（Windows / Linux）
├── train.py                       # 训练入口
├── models/
│   ├── diffusion_layers.py        # 历史预留 DDPM 组件
│   └── lightgcn_diffusion.py      # 主模型：DiffuRec 序列扩散推荐
├── app/
│   ├── main.py                    # Flask Web 服务（用户端 + 管理端）
│   ├── templates/                 # 页面模板（11个）
│   └── static/style.css           # 样式
├── utils/metrics.py               # 评估指标
└── saved/                         # 训练后模型输出
    └── model_checkpoint.pt        # 模型参数、id映射、基础用户序列
```

## 模型架构

```
高评分交互 → 用户历史序列 → DiffuRec 扩散去噪 → 物品内积排序
             按时间截断填充   Transformer 近似器     Recall@K 评估
```

| 组件 | 说明 | 参数 |
|------|------|------|
| Item Embedding | 学习物品向量并作为分类候选空间 | 128维 |
| DiffuRec Core | 对用户历史序列做扩散采样和去噪表征 | 32步, 4层 Transformer |
| Predictor | 用户表征与全量物品向量内积排序 | Recall@10/20/50 |

## 环境安装

### Windows
双击 `setup.bat` 或命令行运行：
```cmd
setup.bat
```

### Linux / macOS
```bash
bash setup.sh
```

### 手动安装
```bash
conda create -n recbole python=3.9 -y
conda activate recbole
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install recbole==1.2.1
pip install numpy pandas scipy flask pyyaml
```

## 使用方法

### 1. 训练模型
```bash
conda activate recbole
python train.py --device cuda --epochs 100
```
训练完成后会将模型参数、物品映射和基础用户历史统一保存到 `saved/model_checkpoint.pt`。

### 2. 启动 Web 服务
```bash
python app/main.py
```
浏览器打开 http://localhost:5000

### 3. 登录
| 身份 | 账号 | 密码 | 说明 |
|------|------|------|------|
| 管理员 | admin | admin123 | 用户/电影/评论管理 |
| 数据集用户 | user_1 ~ user_943 | 123456 | 对应 ml-100k 用户，个性化推荐 |
| 新用户 | 自行注册 | — | 冷启动，评分后推荐逐渐个性化 |

## 推荐策略

```
用户名映射到 MovieLens 用户？  →  读取基础历史序列  →  DiffuRec 在线推理
            ↓ 否
      有正反馈评分？         →  本地评分序列      →  DiffuRec 在线推理
            ↓ 否
           冷启动            →  物品均值向量兜底   →  热门/相似内容探索
```

## 配置说明 (`config.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| embedding_size | 128 | 嵌入维度 |
| val_interval.rating | [4,inf) | 仅保留 4 分及以上交互作为正反馈 |
| epochs | 100 | 训练轮数 |
| train_batch_size | 2048 | 训练批大小 |
| learning_rate | 1e-3 | 学习率 |
| MAX_ITEM_LIST_LENGTH | 50 | 每个用户最多保留的历史序列长度 |
| diffurec_num_blocks | 4 | Transformer 去噪层数 |
| diffusion_steps | 32 | 扩散反采样步数 |
| diffusion_schedule | trunc_lin | 噪声调度策略 |

训练脚本通过 `Config(model='SASRec')` 触发 RecBole 的顺序数据集与原生 `Trainer` 流程，但实际训练的模型实现来自 `models/lightgcn_diffusion.py` 中的 DiffuRec。

## 命令行参数

```bash
python train.py --help

  --config PATH      配置文件路径 (默认 config.yaml)
  --epochs N         训练轮数
  --lr FLOAT         学习率
  --device DEVICE    设备 (cpu / cuda / cuda:0)
  --no_progress      关闭进度条
```
