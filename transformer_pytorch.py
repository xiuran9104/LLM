"""最小的 Transformer 实现（PyTorch 版）：GPT 结构的字符级语言模型。

结构和 transformer_numpy.py 一一对应（可对照着看），但前向/反向都交给 PyTorch，
有 NVIDIA 显卡时自动用 GPU（RTX 50 系 Blackwell 需要 cu128 及以上的轮子）。

模型结构总览（自上而下）：
    token id (B,T)
      → wte/wpe 查表得到"词义+位置"向量 (B,T,d)
      → n_layer 个 Block（自注意力横向交流 + FFN 纵向加工，残差连接）
      → ln_f 收尾归一化 → lm_head 投影到词表 (B,T,vocab)

用法：python transformer_pytorch.py             训练并存档 model.pt
      python transformer_pytorch.py 开头文字     读取存档，续写 200 字符
"""
import math                         # 只用到 sqrt：注意力分数的缩放因子
import sys                          # 读命令行参数：python transformer_pytorch.py [提示文字|train]
from dataclasses import dataclass   # 给配置类自动生成 __init__ / __repr__ 等

import torch                        # 张量库：整个脚本的数值计算底座
import torch.nn as nn               # 神经网络积木：Linear / Embedding / LayerNorm 等
import torch.nn.functional as F     # 函数式接口：softmax / cross_entropy 等不带参数的操作

# 有 NVIDIA 显卡就用 cuda，否则退回 cpu；全脚本所有张量/模型存档都放这个设备上
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GPTConfig:
    """模型超参数；默认值和 NumPy 版保持一致，方便两个实现互相验证。

    参数量粗估公式（不含 embedding）：≈ 12 × n_layer × n_embd²
    """
    vocab_size: int          # 必填：词表大小 = 训练语料去重后的字符数，决定输出 logits 的维度
    block_size: int = 32     # 上下文窗口：模型一次最多能看的序列长度，注意力矩阵就是 (T,T) 大小
    n_embd: int = 64         # 特征维度 d：每个 token 被表示成多少维向量
    n_head: int = 4          # 注意力头数 h：把 d 拆成 h 份并行做注意力（必须整除 d）
    n_layer: int = 2         # Block 堆叠层数：越深表达力越强、也越难训
    dropout: float = 0.0     # 训练时随机丢弃一部分激活做正则化；玩具模型不需要，正经训练设 0.1
    use_sdpa: bool = False   # False=注意力走教学手写版（默认）；True=PyTorch 融合内核（FlashAttention）加速


class Attention(nn.Module):
    """多头因果自注意力：每个 token 只能看自己和之前的 token（GPT 是 decoder-only，全靠它）。

    数学：Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) + mask ) V
    use_sdpa=False（默认）教学版：上式每一步手写，逐行看 shape 流动
    use_sdpa=True ：同一个数学交给 PyTorch 融合内核（FlashAttention），训练快、省显存
    """

    def __init__(self, d, n_head, use_sdpa=False):
        super().__init__()                # 初始化父类 nn.Module，参数才能被注册和追踪
        self.n_head = n_head              # 记下头数 h，forward 里拆头要用
        self.use_sdpa = use_sdpa          # True=融合内核加速版，False=教学手写版
        self.qkv = nn.Linear(d, 3 * d)    # 融合 QKV 投影：Wq、Wk、Wv 三个 (d,d) 矩阵横向拼接成一个 (d,3d)，一次矩阵乘全算完
        self.proj = nn.Linear(d, d)       # 输出投影 Wo：多头结果拼接后再投回残差流

    def forward(self, x):
        B, T, C = x.shape                 # 批大小 B、序列长 T、特征维 C=d，例如 (2, 16, 64)
        q, k, v = self.qkv(x).split(C, dim=-1)  # (B,T,d)@(d,3d)->(B,T,3d)，按最后一维三等分成 Q/K/V，各 (B,T,d)
        # 拆多头：(B,T,d) 先 view 成 (B,T,h,d/h) 再 transpose(1,2) 成 (B,h,T,d/h)
        # 必须先 view 后 transpose：view 是"按内存顺序重切"，这样才能让每个头分到属于自己的那一段特征
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if self.use_sdpa:
            # 加速版：一行 = 下面手写的 ①~⑤（数学相同，内核融合了缩放/mask/softmax/加权和）
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # 输出 (B,h,T,d/h)
        else:
            # ---------- 教学版：softmax(QK^T/sqrt(d_k))V 逐步手写 ----------
            head_dim = q.size(-1)                # 每个头的维度 d_k = d/h，例如 64/4=16
            scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)  # ① 原始分数：(B,h,T,d_k)@(B,h,d_k,T)->(B,h,T,T)，第 i 行 = 第 i 个 token 对所有 token 的相关性；÷sqrt(d_k) 防点积过大把 softmax 掐进饱和区导致没梯度
            future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)  # ② 因果 mask：上三角（不含对角线）标记"未来"位置为 True
            scores = scores.masked_fill(future, float("-inf"))      # ③ 未来位置的分数置 -inf；softmax 中 exp(-inf)=0 → "每个 token 看不到未来"
            att = F.softmax(scores, dim=-1)      # ④ 按最后一维归一化：每行变成权重分布，和为 1
            att = att @ v                        # ⑤ 加权求和 V：(B,h,T,T)@(B,h,T,d_k)->(B,h,T,d_k)，每个位置的新表示 = 按相关性混合过去信息
        return self.proj(att.transpose(1, 2).reshape(B, T, C))  # ⑥ (B,h,T,d_k) 先 transpose 回 (B,T,h,d_k) 再 reshape 拼回 (B,T,d)，经 Wo 投影输出


class Block(nn.Module):
    """一个 Transformer Block（Pre-LN）：x = x + Attn(LN(x))，再 x = x + FFN(LN(x))

    GPT 的基本积木，内部就两组子层：
    ① 自注意力 Attention：token 之间"横向"交流，每个位置去看上下文里相关的信息
    ② 前馈网络 FFN：每个 token "纵向"独立加工，两层 MLP + 非线性，占模型大部分参数
    Pre-LN（GPT-2 起的主流）：LayerNorm 放在子层前面而不是后面，深堆叠时训练更稳；
    两条子层都是残差写法 x = x + 子层(...)：主干 x 全程直通，每层只往主干上加增量，
    梯度可以沿主干无衰减地回传，这是深层网络能训得动的前提。
    """

    def __init__(self, d, n_head, use_sdpa=False):
        super().__init__()                          # 初始化父类，参数才会被注册
        self.ln1 = nn.LayerNorm(d)                  # 注意力前的归一化：按最后一维拉成均值0方差1，再学一对缩放/平移参数，稳定每层输入分布
        self.attn = Attention(d, n_head, use_sdpa)  # 多头自注意力（上一个类逐行注解的实现）：token 之间横向交换信息
        self.ln2 = nn.LayerNorm(d)                  # FFN 前的第二个归一化，作用同 ln1
        self.fc1 = nn.Linear(d, 4 * d)              # FFN 升维：d -> 4d，先扩张，给非线性留出表达空间
        self.fc2 = nn.Linear(4 * d, d)              # FFN 降维：4d -> d，压回残差流维度（否则形状对不上无法相加）；4 倍是原论文沿用的经验值

    def forward(self, x):
        # ① 注意力子层：先归一化再注意力，输出与原 x 残差相加；全程 (B,T,d) 形状不变
        x = x + self.attn(self.ln1(x))                   # token 间交流完成：每个位置混入了上下文信息
        # ② FFN 子层：归一化 -> 升维 -> ReLU（负数截零，引入非线性）-> 降维 -> 残差相加
        x = x + self.fc2(F.relu(self.fc1(self.ln2(x))))  # 每个 token 独立加工（所有位置共用同一组线性层），形状仍 (B,T,d)
        return x                                    # 输出形状 = 输入形状，所以 Block 可以无脑堆 n_layer 层


class GPT(nn.Module):
    """GPT 结构的 Transformer：输入 token id，输出每个位置预测"下一个 token"的 logits。

    训练目标（语言建模的标准做法）：每个位置都去预测它的下一个字符，
    一条序列同时产出 T 个训练信号，语料利用率高。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()                                         # 初始化父类，参数才会被注册
        assert config.vocab_size is not None                       # 两个必填项防呆：没传就尽早报错
        assert config.block_size is not None
        self.config = config                                       # 存下配置：存档/读档/forward 都要用
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embedding：词 id -> 词义向量，本质是可学习的查表 (vocab, d)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # 可学习位置编码：位置号 -> 位置向量；注意力本身对顺序无感，语序信息全靠它注入
        self.drop = nn.Dropout(config.dropout)                     # embedding 之后的 dropout（=0 时等价于直通）
        self.blocks = nn.ModuleList(Block(config.n_embd, config.n_head, config.use_sdpa) for _ in range(config.n_layer))  # 堆 n_layer 个 Block；必须用 ModuleList，普通 list 里的参数不会被注册
        self.ln_f = nn.LayerNorm(config.n_embd)                    # 最后一个 Block 之后的收尾归一化
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 语言模型头：d -> vocab，给词表里每个字符打分；GPT 惯例不带 bias
        self.apply(self._init_weights)                             # 递归遍历所有子模块，统一按 N(0, 0.02) 初始化权重

    @staticmethod
    def _init_weights(module):
        """被上面的 apply() 对每个子模块调用一次：Linear/Embedding 权重 ~ N(0,0.02)，bias 归零。"""
        if isinstance(module, (nn.Linear, nn.Embedding)):          # 只初始化带权重矩阵的两种层
            nn.init.normal_(module.weight, mean=0.0, std=0.02)     # 小方差正态：初始输出不至于爆炸或消失
            if isinstance(module, nn.Linear) and module.bias is not None:  # Linear 的 bias（有的层没有，如 lm_head）
                nn.init.zeros_(module.bias)                        # bias 从 0 开始

    def forward(self, idx, targets=None):
        """idx: (B, T) 的 token id；targets: 右移一位的目标，给了就顺便算交叉熵 loss"""
        B, T = idx.shape                           # 批大小、序列长度
        assert T <= self.config.block_size, f"序列长 {T} 超过最大 {self.config.block_size}"  # 位置编码表只有 block_size 行，超了会越界
        pos = torch.arange(T, device=idx.device)   # 位置号 0..T-1，形状 (T,)
        x = self.drop(self.wte(idx) + self.wpe(pos))  # (B,T) -> (B,T,d)：词义向量 + 位置向量广播相加，再 dropout
        for block in self.blocks:                  # 依次过 n_layer 个 Block（内部 = 注意力 + FFN，残差连接）
            x = block(x)                           # 形状始终保持 (B, T, d)
        logits = self.lm_head(self.ln_f(x))  # (B,T,vocab)：每个位置对词表里每个字符的打分（还没归一化）
        loss = None
        if targets is not None:
            # 交叉熵要求 (B, T, V) 摊平成 (B*T, V) 去和 (B*T,) 的标签比
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))  # 切片出的 targets 非连续，用 reshape；loss = 所有位置"预测下一个字符"的平均误差
        return logits, loss                        # 推理只用 logits；训练两个都用

    @torch.no_grad()  # 推理不需要梯度：不建计算图，省显存也更快
    def generate(self, prompt_ids, n_new=200):
        """推理（自回归生成）：取最后位置的预测接到末尾，循环 n_new 次"""
        idx = prompt_ids                           # (1, T0) 的提示词 id
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.config.block_size:])  # 序列超过 block_size 就只喂末尾窗口（位置编码长度有限）
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdims=True)], dim=1)  # 贪心解码：取打分最高的字符接到末尾，它又成为下一步的输入
        return idx                                 # 原提示 + 新生成的 n_new 个字符


def save_model(model, chars, path="model.pt"):
    """训练完把全部参数 + 词表 + 超参存进一个文件，推理时读档用。

    词表必须一起存：模型输出的是词表下标，没有词表就无法还原成字符。
    存档里只有张量、字符串、数字——不含代码，读档时配合 weights_only=True 才安全。
    """
    cfg = model.config
    torch.save({"model": model.state_dict(),       # 所有参数张量（权重 + 偏置）
                "chars": chars,                    # 词表：id <-> 字符 映射的唯一依据
                "config": {"block_size": cfg.block_size, "n_embd": cfg.n_embd,
                           "n_head": cfg.n_head, "n_layer": cfg.n_layer}}, path)  # 结构超参：读档时按它重建同结构模型


def load_model(path="model.pt", device=DEVICE):
    """读档：按存的超参重建一个同结构的模型，再把参数装回去。

    词表以存档为准（而不是当前文件里的语料），保证和训练时严格一致。
    """
    # weights_only=True：只反序列化张量和基础类型（str/int/dict 等），不执行任何代码。
    # torch.load 底层是 pickle，加载来路不明的 .pt 可能触发任意代码执行，必须显式加这个参数。
    z = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(vocab_size=len(z["chars"]), **z["config"])).to(device)  # 用存档超参重建同结构空模型，搬上设备
    model.load_state_dict(z["model"])              # 把训练好的参数逐张量装入（键名和形状必须严丝合缝）
    model.eval()                                   # 关闭 dropout 等训练期行为；推理前必须调用
    return model, z["chars"]                       # 连词表一起返回：调用方用它做 字符 <-> id 映射


def main():
    # ---------- 语料与词表 ----------
    # 玩具语料：想换自己的数据，把 text 换成 open("你的文本.txt", encoding="utf-8").read() 即可
    text = ("the quick brown fox jumps over the lazy dog. "
            "pack my box with five dozen liquor jugs. "
            "how vexingly quick daft zebras jump. "
            "sphinx of black quartz, judge my vow. ")
    chars = "".join(sorted(set(text)))          # 词表：语料去重再排序，得到固定的 字符->下标 顺序
    stoi = {c: i for i, c in enumerate(chars)}  # 字符 -> id：喂给模型用
    itos = {i: c for c, i in stoi.items()}      # id -> 字符：模型输出还原用

    if len(sys.argv) > 1 and sys.argv[1] != "train":
        # ---------- 推理分支：python transformer_pytorch.py 开头的几个字 ----------
        model, chars = load_model()  # 词表以存档为准，和训练时严格一致
        stoi = {c: i for i, c in enumerate(chars)}  # 字符 -> id（按存档词表重建）
        itos = {i: c for c, i in stoi.items()}      # id -> 字符
        prompt = " ".join(sys.argv[1:])             # 命令行传的多个词拼成一个提示
        bad = set(prompt) - set(chars)              # 提示里有词表外的字符就没法编码
        if bad:
            sys.exit(f"prompt 里有词表外的字符: {bad}，词表只包含训练语料出现过的字符")
        idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=DEVICE)  # 提示转成 (1, len) 的 id 张量，放上 GPU
        out = model.generate(idx)                   # 自回归续写 n_new=200 个字符
        print("".join(itos[i] for i in out[0].tolist()))  # id 序列还原成字符串打印
    else:
        # ---------- 训练分支：python transformer_pytorch.py（或加参数 train）----------
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=DEVICE)  # 整个语料转成一条 id 长链，放上 GPU
        model = GPT(GPTConfig(vocab_size=len(chars))).to(DEVICE)  # .to(DEVICE)：参数搬到 GPU
        n_params = sum(p.numel() for p in model.parameters())     # 参数总量：每个参数张量的元素数求和
        print(f"设备: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
        print(f"参数量: {n_params}")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)       # Adam 优化器：按梯度更新所有参数，lr = 学习率

        def get_batch(batch_size):
            """随机截取 batch_size 段文本：x 是输入，y 是右移一位的预测目标"""
            ix = torch.randint(0, len(data) - model.config.block_size, (batch_size,), device=DEVICE)  # 随机抽 batch_size 个起点
            chunks = torch.stack([data[i:i + model.config.block_size + 1] for i in ix])  # 每个起点截 block_size+1 个字符，摞成 (B, T+1)
            return chunks[:, :-1], chunks[:, 1:]  # 前 T 个做输入 x，后 T 个做目标 y：y 恰好是 x 右移一位

        import time
        t0 = time.time()
        for step in range(1, 1001):                # 训练 1000 步
            x, y = get_batch(32)                   # 每步现抽一批 (B=32, T) 样本
            _, loss = model(x, y)                  # 前向：所有位置预测下一个字符的平均交叉熵
            opt.zero_grad()   # 清掉上一步的梯度：PyTorch 梯度是累加的，不清会越叠越大
            loss.backward()   # 反向传播：PyTorch 自动算出所有参数的梯度
            opt.step()        # 优化器按梯度更新一次参数
            if step % 100 == 0:
                print(f"step {step:4d}  loss {loss.item():.3f}")  # loss 越小 = 越会预测下一个字符
        print(f"训练耗时 {time.time() - t0:.1f}s")

        save_model(model, chars)  # 存档，之后 python transformer_pytorch.py 提示文本 即可推理
        print("已存档到 model.pt")
        out = model.generate(torch.tensor([[stoi[text[0]]]], device=DEVICE))  # 用语料首字符做提示，保证在词表内
        print("".join(itos[i] for i in out[0].tolist()))  # 存完顺手推理一段看效果


if __name__ == "__main__":  # 直接运行本文件才执行；被 import 时不执行（别处可以复用 Attention 类）
    main()
