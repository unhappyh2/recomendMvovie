# Breeze — 基于 LightGCN + Diffusion 的电影推荐系统

融合图神经网络与扩散模型的推荐系统，RecBole 框架实现，Flask Web 交互。

## 项目结构

```
recomendMvovie/
├── config.yaml                    # 模型/训练超参数配置
├── setup.bat / setup.sh           # 一键环境安装（Windows / Linux）
├── train.py                       # 训练入口
├── models/
│   ├── diffusion_layers.py        # DDPM 组件（调度器、时间嵌入、去噪MLP）
│   └── lightgcn_diffusion.py      # 主模型：LightGCN + Diffusion + Predictor
├── app/
│   ├── main.py                    # Flask Web 服务（用户端 + 管理端）
│   ├── templates/                 # 页面模板（11个）
│   └── static/style.css           # 样式
├── utils/metrics.py               # 评估指标
└── saved/                         # 训练后模型输出
    ├── user_emb.npy               # 用户嵌入 (944, 64)
    ├── item_emb.npy               # 物品嵌入 (1683, 64)
    └── model_checkpoint.pt        # 完整 checkpoint
```

## 模型架构

```
交互数据 → 图嵌入层(LightGCN) → 特征增强层(DDPM) → 推荐预测层(Inner Product)
            3层图卷积、BPR损失      20步扩散、噪声预测       内积评分排序
```

| 组件 | 说明 | 参数 |
|------|------|------|
| LightGCN | 轻量图卷积，学习用户-物品交互特征 | 3层, 64维嵌入 |
| DDPM | 扩散概率模型，对嵌入加噪再去噪，增强鲁棒性 | 20步扩散 |
| Predictor | 内积法计算用户-物品匹配度 | — |

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
python train.py --device cuda --epochs 200
```
训练完成后嵌入自动保存到 `saved/` 目录。

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
recbole_user_id 匹配？  →  使用预训练嵌入  → 完全个性化
        ↓ 否
    有评分记录？  →  加权合成嵌入  → 你的口味
        ↓ 否
    冷启动  →  全局平均嵌入  → 积累评分后变个性
```

## 配置说明 (`config.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| embedding_size | 64 | 嵌入维度 |
| n_layers | 3 | LightGCN 层数 |
| diffusion_steps | 20 | DDPM 扩散步数 |
| diffusion_beta_start | 1e-4 | 噪声调度起始值 |
| diffusion_beta_end | 0.02 | 噪声调度结束值 |
| epochs | 200 | 训练轮数 |
| train_batch_size | 2048 | 训练批大小 |
| learning_rate | 1e-3 | 学习率 |
| reg_weight | 1e-4 | 正则化系数 |

## 命令行参数

```bash
python train.py --help

  --config PATH      配置文件路径 (默认 config.yaml)
  --epochs N         训练轮数
  --lr FLOAT         学习率
  --device DEVICE    设备 (cpu / cuda / cuda:0)
  --no_progress      关闭进度条
```
