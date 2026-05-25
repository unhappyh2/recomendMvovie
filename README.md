# Breeze — 基于 SASRec 的电影推荐系统

`Breeze` 是一个基于 MovieLens-100k 的电影推荐系统。训练侧使用 **RecBole 官方内置 `SASRec`**，Web 侧使用 Flask，保持用户登录、评分、评论、管理后台和在线推荐流程不变。

## 项目结构

```text
recomendMvovie/
├── config.yaml              # RecBole 和 SASRec 超参数
├── train.py                 # 训练入口，保存 checkpoint
├── data/sequence_data.py    # 原始 id 映射和用户历史序列导出
├── models/                  # 历史实验模型代码，当前主流程不依赖
├── app/
│   ├── main.py              # Flask 应用与在线推理
│   ├── templates/           # 页面模板
│   └── static/style.css     # 样式
└── saved/
    └── model_checkpoint.pt  # 训练后模型、id 映射、基础用户历史
```

## 当前模型

- 训练模型：RecBole 官方 `SASRec`
- 训练框架：`Config -> create_dataset -> data_preparation -> Trainer.fit/evaluate`
- 线上推理：Flask 读取 `saved/model_checkpoint.pt`，按用户历史实时生成序列表示

当前轻量配置见 [config.yaml](/Users/huahua/Documents/WorkStudio/recomendMvovie/config.yaml:19)：
- `hidden_size: 64`
- `n_layers: 2`
- `n_heads: 2`
- `MAX_ITEM_LIST_LENGTH: 50`
- `loss_type: CE`

## 训练与启动

```bash
conda activate recbole
python train.py --device cuda
python app/main.py
```

训练完成后会生成 `saved/model_checkpoint.pt`。Web 服务默认监听 `http://localhost:5000`。

## 业务接口与推荐流程

Web 侧不再读取 `user_emb.npy / item_emb.npy`，统一读取 `saved/model_checkpoint.pt`。

推荐流程：
1. 识别当前用户是否是 `user_1 ~ user_943`
2. 读取 checkpoint 中保存的基础历史序列
3. 追加当前站内评分里 `rating >= 4.0` 的正反馈电影
4. 截断到 `MAX_ITEM_LIST_LENGTH`
5. 调用 `SASRec.forward(item_seq, item_seq_len)` 生成用户表示
6. 与全量 `item_embedding` 做内积排序，过滤已看电影

冷启动用户没有历史时，系统退化为物品向量均值兜底。

## 默认账号

| 身份 | 账号 | 密码 |
|------|------|------|
| 管理员 | `admin` | `admin123` |
| 数据集用户 | `user_1 ~ user_943` | `123456` |

## 说明

- `models/official_diffurec.py`、`models/lightgcn_diffusion.py` 是历史实验代码，当前主训练和主推理链路不使用。
- `saved/model_checkpoint.pt`、`log_tensorboard/`、`__pycache__/` 都属于生成物，不建议提交。
