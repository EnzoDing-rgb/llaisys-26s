# 任务 2.7：SwiGLU 讲义

> 用一组假但好算的数字，把 FFN + SwiGLU 每张矩阵、每个中间张量的形状钉死。然后再说它为什么在这儿。

---

## 1. 约定一个具体规模（贯穿全文）

假装：

| 名字 | 取值 | 含义 |
|------|------|------|
| `seqlen` | 3 | 句子里 3 个 token |
| `hidden` | 64 | 每个 token 的隐层宽度 |
| `intermediate` | 512 | FFN 升维后的宽度 |

本层进 FFN 之前，残差后的激活长这样：

```text
x : [3, 64]
```

也就是 3 行，每行 64 个数。

---

## 2. 整层里 FFN 站在哪（只保留必要上下文）

```text
x [3, 64]
  → Attention（略）→ 残差
  → RMSNorm
  → ★ FFN（下面展开）★
  → 残差
```

Attention 混完上下文之后，每个位置仍是 **64 维**。FFN 对这 3 个位置 **各自** 做升维 → 门控合并 → 降维，位置之间互不看。

---

## 3. FFN 里到底有几块权重？形状是什么？

LLaMA 式 SwiGLU-FFN 有 **三块线性权重**（偏置可忽略）：

### 3.1 两路升维（多出来的就是「多一路矩阵」）

相对「只升维一次再激活」的老 FFN，这里 **升维做成两路**，所以多了一张与 `W_up` 同形状的 `W_gate`：

```text
W_gate : [512, 64]    // 线性：y = x @ W_gate^T 时，出 512
W_up   : [512, 64]    // 同样从 64 升到 512
```

对输入 `x [3, 64]`：

```text
gate = x @ W_gate^T  →  [3, 512]
up   = x @ W_up^T    →  [3, 512]
```

每一行（每个 token）都得到 512 个数的 `gate` 和 512 个数的 `up`。

### 3.2 本作业的 SwiGLU（没有新矩阵）

```text
hidden_ff = SwiGLU(gate, up)  →  [3, 512]
```

逐元素：

$$
\mathrm{hidden\_ff}_{t,j} = \mathrm{up}_{t,j} \cdot \mathrm{SiLU}(\mathrm{gate}_{t,j})
$$

$$
\mathrm{SiLU}(z) = \frac{z}{1 + e^{-z}}
$$

**这里不再乘任何权重**，只是把两个 `[3, 512]` 揉成一个 `[3, 512]`。

### 3.3 一路降维

```text
W_down : [64, 512]    // 从 512 压回 64
```

```text
ffn_out = hidden_ff @ W_down^T  →  [3, 64]
```

和进 FFN 前的 `x` 同形状，好加残差。

---

## 4. 一张表看完「64 → 512 → 64」

| 东西 | 形状 | 谁算的 |
|------|------|--------|
| `x` | `[3, 64]` | 上一层 / Attention 残差后 |
| `W_gate` | `[512, 64]` | 参数 |
| `W_up` | `[512, 64]` | 参数 |
| `gate` | `[3, 512]` | `x` 与 `W_gate` |
| `up` | `[3, 512]` | `x` 与 `W_up` |
| `hidden_ff` | `[3, 512]` | **SwiGLU(`gate`,`up`)** ← 作业 |
| `W_down` | `[64, 512]` | 参数 |
| `ffn_out` | `[3, 64]` | `hidden_ff` 与 `W_down` |

所以：

- **多出来的矩阵**：相对「单路升维 FFN」，多的是整张 **`W_gate`（与 `W_up` 同大）**。  
- **`swiglu` 算子本身**：零张新矩阵，只吃已经算好的 `gate`/`up`。

---

## 5. 用一个「单 token、单维」把数算透

只看 token 0 的 intermediate 第 0 维，设：

```text
gate[0,0] = 2
up[0,0]   = 3
```

$$
\mathrm{SiLU}(2) = \frac{2}{1+e^{-2}} \approx 1.7616
$$

$$
\mathrm{hidden\_ff}[0,0] \approx 3 \times 1.7616 \approx 5.2848
$$

其余 511 维各自同样算；3 个 token 互不干涉。算完 512 维后，再用 `W_down` 变回 64 维。

---

## 6. Why we want 这一套？

1. **FFN 要干嘛：** Attention 之后仍在 `hidden=64`；需要按位置做一次非线性加工。做法是先拉到 `intermediate=512` 的宽空间里折腾，再压回 64。  
2. **为啥两路升维：** `up` 当「内容」，`gate` 经 SiLU 后当「逐维增益」，`up ⊙ SiLU(gate)` 比「单路升维再激活」更灵活。代价就是 **多存、多算一张 `W_gate`**。  
3. **作业边界：** 实现 `out = up ⊙ SiLU(gate)`；形状与测试一致，例如 `[2,3]`、`[512,4096]`（那里的 `4096` 就是 intermediate）。

---

## 7. 实现与自检

```text
out[i] = up[i] * (gate[i] / (1 + exp(-gate[i])))
```

`gate` / `up` / `out` 同形状、连续；F16/BF16 在 float 里算 `exp`。

```bash
python test/ops/swiglu.py
```
