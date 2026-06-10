CONTROLLER_HEART_BEAT_EXPIRATION = 30#控制器心跳过期时间（30秒），用于分布式训练时检测控制器是否还活跃
WORKER_HEART_BEAT_INTERVAL = 15# 工作节点心跳间隔（15秒），工作节点每隔15秒向控制器发送心跳信号

LOGDIR = "."#日志文件保存的默认目录（当前目录）

# Model Constants
IGNORE_INDEX = -100# 在计算损失时忽略的标签索引，这是 PyTorch 中 CrossEntropyLoss 的默认忽略索引，用于不计算某些 token 的损失（如 padding tokens）
GRAPH_TOKEN_INDEX = -200#图 token 的特殊索引标识
DEFAULT_GRAPH_TOKEN = "<graph>"#图数据在文本中的占位符 token
DEFAULT_GRAPH_START_TOKEN = "<GH>"
DEFAULT_GRAPH_END_TOKEN = "</GH>"
DEFAULT_GRAPH_PAD_ID = -500#图数据填充（padding）时使用的 ID