"""最小的 Transformer 实现（只依赖 NumPy）：GPT 结构的字符级语言模型。

token embedding + 位置编码 -> N x [Transformer Block] -> LayerNorm -> 输出层
每个 Block = 多头因果自注意力 + 前馈网络，都带残差连接和 Pre-LN。
前向、反向传播、Adam 全部手写。运行：python test.py
"""
import sys

import numpy as np

rng = np.random.default_rng(0)


def softmax(x):
    """沿最后一维做 softmax（先减最大值，防止 exp 溢出）"""
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def split_heads(x, n_head):
    """(B, T, C) -> (B, n_head, T, C // n_head)：把 C 维切成 n_head 个头"""
    B, T, C = x.shape
    return x.reshape(B, T, n_head, C // n_head).transpose(0, 2, 1, 3)


def merge_heads(x):
    """(B, n_head, T, hs) -> (B, T, n_head * hs)：把各个头拼回去"""
    B, H, T, hs = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * hs)


class Linear:
    """y = x @ W + b"""

    def __init__(self, n_in, n_out):
        self.W = rng.normal(0, 0.02, (n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW, self.db = np.zeros_like(self.W), np.zeros_like(self.b)  # 梯度

    def __call__(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        """g 是 dL/dy：算出 dW、db，返回 dL/dx"""
        x2 = self.x.reshape(-1, self.W.shape[0])
        g2 = g.reshape(-1, self.W.shape[1])
        np.matmul(x2.T, g2, out=self.dW)
        self.db[:] = g2.sum(axis=0)
        return g @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class LayerNorm:
    """对最后一维归一化，再乘 g、加 b"""

    def __init__(self, d, eps=1e-5):
        self.g, self.b, self.eps = np.ones(d), np.zeros(d), eps
        self.dg, self.db = np.zeros_like(self.g), np.zeros_like(self.b)

    def __call__(self, x):
        self.sd = np.sqrt(x.var(-1, keepdims=True) + self.eps)  # 标准差 (..., 1)
        self.xhat = (x - x.mean(-1, keepdims=True)) / self.sd   # 归一化结果
        return self.g * self.xhat + self.b

    def backward(self, g):
        dxhat = g * self.g
        axes = tuple(range(g.ndim - 1))
        self.dg[:] = (g * self.xhat).sum(axis=axes)
        self.db[:] = g.sum(axis=axes)
        m1 = dxhat.mean(-1, keepdims=True)
        m2 = (dxhat * self.xhat).mean(-1, keepdims=True)
        return (dxhat - m1 - self.xhat * m2) / self.sd

    def params(self):
        return [(self.g, self.dg), (self.b, self.db)]


class Attention:
    """多头因果自注意力：每个 token 只能看自己和之前的 token"""

    def __init__(self, d, n_head):
        self.n_head = n_head
        self.qkv = Linear(d, 3 * d)  # 一次算出 Q、K、V
        self.proj = Linear(d, d)

    def __call__(self, x):
        q, k, v = (split_heads(t, self.n_head) for t in np.split(self.qkv(x), 3, axis=-1))
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1])     # (B, H, T, T) 注意力分数
        causal = np.tril(np.ones((x.shape[1], x.shape[1]), dtype=bool))  # 只有下三角可见
        att = softmax(np.where(causal, scores, -np.inf))                 # 每一行是一个注意力分布
        self.q, self.k, self.v, self.att = q, k, v, att
        return self.proj(merge_heads(att @ v))                           # (B, T, C)

    def backward(self, g):
        q, k, v, att = self.q, self.k, self.v, self.att
        dy = split_heads(self.proj.backward(g), self.n_head)
        datt = dy @ v.transpose(0, 1, 3, 2)                             # y = att @ v
        dv = att.transpose(0, 1, 3, 2) @ dy
        datt = att * (datt - (datt * att).sum(-1, keepdims=True))       # softmax 的反向
        scale = 1 / np.sqrt(q.shape[-1])
        dq, dk = (datt @ k) * scale, (datt.transpose(0, 1, 3, 2) @ q) * scale
        dqkv = np.concatenate([merge_heads(dq), merge_heads(dk), merge_heads(dv)], axis=-1)
        return self.qkv.backward(dqkv)

    def params(self):
        return self.qkv.params() + self.proj.params()


class Block:
    """一个 Transformer Block（Pre-LN）：x = x + Attn(LN(x))，再 x = x + FFN(LN(x))"""

    def __init__(self, d, n_head):
        self.ln1, self.attn = LayerNorm(d), Attention(d, n_head)
        self.ln2, self.fc1, self.fc2 = LayerNorm(d), Linear(d, 4 * d), Linear(4 * d, d)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        self.h = np.maximum(self.fc1(self.ln2(x)), 0)  # 前馈网络，激活函数用 ReLU
        return x + self.fc2(self.h)

    def backward(self, g):
        dh = self.fc2.backward(g) * (self.h > 0)                # ReLU 的反向
        dx = g + self.ln2.backward(self.fc1.backward(dh))       # 过第二个残差连接，得到 dL/dx1
        # 第一个残差：梯度 dx 先过 attention 的反向，再过 ln1 的反向（链式法则顺序不能反）
        return dx + self.ln1.backward(self.attn.backward(dx))

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.fc1.params() + self.fc2.params()


class Transformer:
    """GPT 结构的 Transformer：输入 token id，输出下一个 token 的概率（logits）"""

    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=2):
        self.block_size = block_size  # 最长序列
        self.n_embd, self.n_head, self.n_layer = n_embd, n_head, n_layer  # 存下超参，存档/读档时要用
        self.wte = rng.normal(0, 0.02, (vocab_size, n_embd))  # token embedding
        self.wpe = rng.normal(0, 0.02, (block_size, n_embd))  # 位置编码（可学习）
        self.dwte, self.dwpe = np.zeros_like(self.wte), np.zeros_like(self.wpe)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f, self.lm_head = LayerNorm(n_embd), Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """idx: (B, T) 的 token id；返回 (logits, loss)，没有 targets 时 loss 为 None"""
        self.idx = idx
        x = self.wte[idx] + self.wpe[: idx.shape[1]]  # (B, T) -> (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        self.probs = softmax(logits)
        p = np.take_along_axis(self.probs, targets[..., None], axis=-1)[..., 0]  # 目标 token 的概率
        return logits, -np.log(p).mean()  # 交叉熵 loss

    def backward(self, targets):
        """从 loss 反向传播到 embedding；梯度写在各层的 dW / db 里"""
        g = (self.probs - np.eye(len(self.wte))[targets]) / targets.size  # dL/dlogits
        g = self.ln_f.backward(self.lm_head.backward(g))
        for block in reversed(self.blocks):
            g = block.backward(g)
        self.dwpe[: self.idx.shape[1]] = g.sum(axis=0)
        self.dwte[:] = 0
        np.add.at(self.dwte, self.idx, g)  # 同一个 token 出现多次，梯度要累加

    def params(self):
        """所有 (参数, 梯度)，交给优化器更新"""
        ps = [(self.wte, self.dwte), (self.wpe, self.dwpe)]
        for block in self.blocks:
            ps += block.params()
        return ps + self.ln_f.params() + self.lm_head.params()


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999):
        self.params, self.lr, self.b1, self.b2 = params, lr, b1, b2
        self.m = [np.zeros_like(p) for p, _ in params]  # 梯度的一阶矩
        self.v = [np.zeros_like(p) for p, _ in params]  # 梯度的二阶矩
        self.t = 0

    def step(self):
        self.t += 1
        for (p, g), m, v in zip(self.params, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)  # 修正偏差
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def save_model(model, chars, path="model.npz"):
    """训练完把全部参数 + 词表 + 超参存进一个文件，推理时读档用"""
    arrays = {f"p{i:02d}": p for i, (p, _) in enumerate(model.params())}
    np.savez(path, chars=np.frombuffer(chars.encode("utf-8"), np.uint8),
             n_embd=model.n_embd, n_head=model.n_head, n_layer=model.n_layer,
             block_size=model.block_size, **arrays)


def load_model(path="model.npz"):
    """读档：按存的超参重建一个同结构的模型，再把参数逐个覆盖进去"""
    z = np.load(path)
    chars = z["chars"].tobytes().decode("utf-8")
    model = Transformer(vocab_size=len(chars), block_size=int(z["block_size"]),
                        n_embd=int(z["n_embd"]), n_head=int(z["n_head"]),
                        n_layer=int(z["n_layer"]))
    for i, (p, _) in enumerate(model.params()):
        p[:] = z[f"p{i:02d}"]
    return model, chars


def generate(model, stoi, itos, prompt=" ", n_new=300):
    """推理（自回归生成）：prompt 喂进去，取最后位置的预测接到末尾，循环 n_new 次"""
    idx = np.array([[stoi[c] for c in prompt]])
    for _ in range(n_new):
        logits, _ = model.forward(idx[:, -model.block_size:])  # 超过 block_size 就截取末尾
        idx = np.concatenate([idx, logits[:, -1].argmax(-1, keepdims=True)], axis=1)
    return prompt + "".join(itos[i] for i in idx[0, len(prompt):])


if __name__ == "__main__":
    # 玩具语料：想换自己的数据，把 text 换成 open("你的文本.txt", encoding="utf-8").read() 即可
    text = ("the quick brown fox jumps over the lazy dog. "
            "pack my box with five dozen liquor jugs. "
            "how vexingly quick daft zebras jump. "
            "sphinx of black quartz, judge my vow. ")
    chars = "".join(sorted(set(text)))

    if len(sys.argv) > 1 and sys.argv[1] != "train":
        # 推理：python test.py 开头的几个字  —— 读取 model.npz 续写
        model, _ = load_model()
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}
        prompt = " ".join(sys.argv[1:])
        bad = set(prompt) - set(chars)
        if bad:
            sys.exit(f"prompt 里有词表外的字符: {bad}，词表只包含训练语料出现过的字符")
        print(generate(model, stoi, itos, prompt))
    else:
        # 训练：python test.py 或 python test.py train —— 训练完存档到 model.npz
        stoi = {c: i for i, c in enumerate(chars)}  # 字符 -> id
        itos = {i: c for c, i in stoi.items()}      # id -> 字符
        data = np.array([stoi[c] for c in text])

        block_size = 32
        model = Transformer(vocab_size=len(chars), block_size=block_size)
        opt = Adam(model.params(), lr=1e-3)
        print(f"参数量: {sum(p.size for p, _ in model.params())}")

        def get_batch(batch_size):
            """随机截取 batch_size 段文本：x 是输入，y 是右移一位的预测目标"""
            starts = rng.integers(0, len(data) - block_size, batch_size)
            chunks = np.stack([data[s:s + block_size + 1] for s in starts])
            return chunks[:, :-1], chunks[:, 1:]

        for step in range(1, 1001):
            x, y = get_batch(32)
            _, loss = model.forward(x, y)
            model.backward(y)
            opt.step()
            if step % 100 == 0:
                print(f"step {step:4d}  loss {loss:.3f}")

        save_model(model, chars)  # 存档，之后 python test.py 提示文本 即可推理
        print("已存档到 model.npz")
        print(generate(model, stoi, itos, " "))  # 存完顺手推理一段看效果
