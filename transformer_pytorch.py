"""最小的 Transformer 实现（PyTorch 版）：GPT 结构的字符级语言模型。

结构和 transformer_numpy.py 一一对应（可对照着看），但前向/反向都交给 PyTorch，
有 NVIDIA 显卡时自动用 GPU（RTX 50 系 Blackwell 需要 cu128 及以上的轮子）。
用法：python test.py             训练并存档 model.pt
      python test.py 开头文字     读取存档，续写 200 字符
"""
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# 有 NVIDIA 显卡就用 cuda，否则退回 cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GPTConfig:
    """超参数；默认值和 NumPy 版保持一致"""
    vocab_size: int          # 词表大小 = 语料中出现过的字符数
    block_size: int = 32     # 最长序列长度
    n_embd: int = 64         # embedding 维度
    n_head: int = 4          # 注意力头数
    n_layer: int = 2         # Block 堆叠层数
    dropout: float = 0.0     # 想做正则化可以调成 0.1
    use_sdpa: bool = False   # 默认教学版手写注意力；True 切换 PyTorch 融合内核（FlashAttention）加速


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
            scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)  # ① 原始分数：(B,h,T,d_k)@(B,h,d_k,T)->(B,h,T,T)，第 i 行 = 第 i 个 token 对所有 token 的相关性；÷sqrt(d_k) 防点积过大把 softmax 推进饱和区导致没梯度
            future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)  # ② 因果 mask：上三角（不含对角线）标记"未来"位置为 True
            scores = scores.masked_fill(future, float("-inf"))      # ③ 未来位置的分数置 -inf；softmax 中 exp(-inf)=0 → "每个 token 看不到未来"
            att = F.softmax(scores, dim=-1)      # ④ 按最后一维归一化：每行变成权重分布，和为 1
            att = att @ v                        # ⑤ 加权求和 V：(B,h,T,T)@(B,h,T,d_k)->(B,h,T,d_k)，每个位置的新表示 = 按相关性混合过去信息
        return self.proj(att.transpose(1, 2).reshape(B, T, C))  # ⑥ (B,h,T,d_k) 先 transpose 回 (B,T,h,d_k) 再 reshape 拼回 (B,T,d)，经 Wo 投影输出


class Block(nn.Module):
    """一个 Transformer Block（Pre-LN）：x = x + Attn(LN(x))，再 x = x + FFN(LN(x))"""

    def __init__(self, d, n_head, use_sdpa=False):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(d), Attention(d, n_head, use_sdpa)
        self.ln2 = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)  # 前馈网络，中间层放大 4 倍

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.relu(self.fc1(self.ln2(x))))
        return x


class GPT(nn.Module):
    """GPT 结构的 Transformer：输入 token id，输出每个位置下一个 token 的 logits"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # 可学习位置编码
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config.n_embd, config.n_head, config.use_sdpa) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)  # 所有权重初始化为 N(0, 0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """idx: (B, T) 的 token id；targets: 右移一位的目标，给了就顺便算交叉熵 loss"""
        B, T = idx.shape
        assert T <= self.config.block_size, f"序列长 {T} 超过最大 {self.config.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))  # (B, T) -> (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # 交叉熵要求 (B, T, V) 摊平成 (B*T, V) 去和 (B*T,) 的标签比
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))  # 切片出的 targets 非连续，用 reshape
        return logits, loss

    @torch.no_grad()  # 推理不需要梯度，省显存也更快
    def generate(self, prompt_ids, n_new=200):
        """推理（自回归生成）：取最后位置的预测接到末尾，循环 n_new 次"""
        idx = prompt_ids
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.config.block_size:])  # 超过 block_size 就截取末尾
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdims=True)], dim=1)
        return idx


def save_model(model, chars, path="model.pt"):
    """训练完把全部参数 + 词表 + 超参存进一个文件，推理时读档用"""
    cfg = model.config
    torch.save({"model": model.state_dict(), "chars": chars,
                "config": {"block_size": cfg.block_size, "n_embd": cfg.n_embd,
                           "n_head": cfg.n_head, "n_layer": cfg.n_layer}}, path)


def load_model(path="model.pt", device=DEVICE):
    """读档：按存的超参重建一个同结构的模型，再把参数装回去"""
    z = torch.load(path, map_location=device)  # map_location：存档在别的机器存的也能读
    model = GPT(GPTConfig(vocab_size=len(z["chars"]), **z["config"])).to(device)
    model.load_state_dict(z["model"])
    model.eval()  # 关闭 dropout 等训练行为
    return model, z["chars"]


def main():
    # 玩具语料：想换自己的数据，把 text 换成 open("你的文本.txt", encoding="utf-8").read() 即可
    text = ("the quick brown fox jumps over the lazy dog. "
            "pack my box with five dozen liquor jugs. "
            "how vexingly quick daft zebras jump. "
            "sphinx of black quartz, judge my vow. ")
    chars = "".join(sorted(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}  # 字符 -> id
    itos = {i: c for c, i in stoi.items()}      # id -> 字符

    if len(sys.argv) > 1 and sys.argv[1] != "train":
        # 推理：python test.py 开头的几个字  —— 读取 model.pt 续写
        model, chars = load_model()  # 词表以存档为准，和训练时严格一致
        stoi = {c: i for i, c in enumerate(chars)}  # 字符 -> id
        itos = {i: c for c, i in stoi.items()}      # id -> 字符
        prompt = " ".join(sys.argv[1:])
        bad = set(prompt) - set(chars)
        if bad:
            sys.exit(f"prompt 里有词表外的字符: {bad}，词表只包含训练语料出现过的字符")
        idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=DEVICE)
        out = model.generate(idx)
        print("".join(itos[i] for i in out[0].tolist()))
    else:
        # 训练：python test.py 或 python test.py train —— 训练完存档到 model.pt
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=DEVICE)
        model = GPT(GPTConfig(vocab_size=len(chars))).to(DEVICE)  # .to(DEVICE)：参数搬到 GPU
        n_params = sum(p.numel() for p in model.parameters())
        print(f"设备: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
        print(f"参数量: {n_params}")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def get_batch(batch_size):
            """随机截取 batch_size 段文本：x 是输入，y 是右移一位的预测目标"""
            ix = torch.randint(0, len(data) - model.config.block_size, (batch_size,), device=DEVICE)
            chunks = torch.stack([data[i:i + model.config.block_size + 1] for i in ix])
            return chunks[:, :-1], chunks[:, 1:]

        import time
        t0 = time.time()
        for step in range(1, 1001):
            x, y = get_batch(32)
            _, loss = model(x, y)
            opt.zero_grad()   # 清掉上一步的梯度
            loss.backward()   # 反向传播：PyTorch 自动算出所有参数的梯度
            opt.step()        # 优化器按梯度更新参数
            if step % 100 == 0:
                print(f"step {step:4d}  loss {loss.item():.3f}")
        print(f"训练耗时 {time.time() - t0:.1f}s")

        save_model(model, chars)  # 存档，之后 python test.py 提示文本 即可推理
        print("已存档到 model.pt")
        out = model.generate(torch.tensor([[stoi[text[0]]]], device=DEVICE))  # 用语料首字符做提示，保证在词表内
        print("".join(itos[i] for i in out[0].tolist()))  # 存完顺手推理一段看效果


if __name__ == "__main__":
    main()
