# 3 THE PROPOSED METHOD

本节提出一个由心理账户理论驱动的两阶段价格感知推荐模型。3.1 节给出问题定义、记号与时间约定；3.2 节从 Thaler (1985) 的购买判定规则出发，推导出含“获得效用—价格支出压力—交易效用”三项的潜在净价值，并由随机效用假设得到选择概率，其中获得效用的非价格部分进一步分解为稳定匹配与随使用经验变化的状态依赖；3.3 节以因果图刻画价格、品类、用户状态与购买之间的关系，导出价格效应的识别条件与两阶段结构；3.4–3.9 节依次给出统一打分函数、共享编码、第一阶段（品类偏好）、价格参数头、第二阶段（兴趣偏好）与匹配项；3.10 节给出训练目标、估计与阶段衔接；3.11 节分析计算复杂度。

---

## 3.1 问题定义、记号与时间约定

设用户集合为 $\mathcal U$，商品集合为 $\mathcal I$，每个商品 $i$ 属于唯一品类 $c(i)\in\mathcal C$。一条交易记录为 $o=(u,i,p,t)$，表示用户 $u$ 在时刻 $t$ 以价格 $p$ 购买了商品 $i$。所有记录按其发生顺序排列。给定预测时刻 $t$ 之前的全部记录，任务是为用户 $u$ 在候选集 $\mathcal J(t)$ 上给出下一次购买的概率分布 $P(i\mid u,t)$，并同时估计价格对选择的影响及其在品类间的差异。

**时间约定。** 记 $d(t)$ 为时刻 $t$ 所在的日。价格以日为粒度公布：候选商品 $j$ 的价格取其在 $d(t)$ 之前最近一次观测到的价格，记为 $\tilde p_j(d^-)$；候选集 $\mathcal J(t)$ 由在 $d(t)$ 之前有可用价格观测的商品构成。市场层面的量，即品类状态、参考价与候选价格，统一以 $d(t)$ 开始为信息截点，只使用 $d(t)$ 之前各日的训练记录，记为 $\cdot\,(d^-)$，因此它们同属一个信息集，且相对当期价格是预先确定的量。用户层面的量使用该用户在 $t$ 之前的记录。因果图（图 1）中市场层面节点的时期下标取为日。

价格在对数尺度上按品类标准化：

$$
\tilde p_{i}(t)=\frac{\log p_i(t)-\mu_{c(i)}}{\sigma_{c(i)}},
\tag{1}
$$

其中 $\mu_c,\sigma_c$ 为品类 $c$ 在训练期的对数价格均值与标准差。$\tilde p$ 刻画商品在本品类中的相对价位。参考价 $\tilde p^{*}_c(d^-)$ 与 $\tilde p$ 处于同一尺度，其构造见 3.3.4 节。主要记号汇总如下。

| 记号 | 含义 |
|---|---|
| $\tilde p_i(t)$ | 商品 $i$ 的品类内标准化对数价格 |
| $\tilde p^{*}_c(d^-)$ | 品类 $c$ 在预测日 $d$ 的参考价格（同一尺度） |
| $\mathrm{AU}_{u,i}(t)$ | 获得效用中与价格无关的部分（用户—商品匹配与经验） |
| $S_{u,i}(t)$ | 获得效用中的状态依赖项（用户对该商品的购买经验） |
| $\mathrm{TU}_{u,i}(t)=\tilde p^{*}_c(d^-)-\tilde p_i(d^-)$ | 交易效用（相对参考价的划算程度） |
| $\beta_{u,c}(t)\ge 0$ | 价格支出压力系数 |
| $\gamma_{u,c}(t)\ge 0$ | 交易效用系数 |
| $k_c(t)$ | 品类购买阈值 |
| $U_t,\ P_t,\ C,\ Y_t$ | 用户状态、价格、品类、购买（因果图节点） |
| $d(t)$ | 时刻 $t$ 所在的日；$T$ 为训练期天数 |
| $n_{c,d}$ | 品类 $c$ 在第 $d$ 日的训练交易数 |
| $n_{uj}(t),\ t^{\mathrm{last}}_{uj}$ | 用户 $u$ 在 $t$ 之前购买 $j$ 的次数与最近一次购买时间 |
| $W$ | 用户序列的块长 |

---

## 3.2 理论基础：从心理账户到选择概率

### 3.2.1 Thaler 的购买判定规则

Thaler (1985) 将一次购买带来的总价值分解为两部分：

$$
w(z,p,p^{*})=\underbrace{v(\hat p,\,-p)}_{\text{获得效用 AU}}+\underbrace{v(-p:\,-p^{*})}_{\text{交易效用 TU}},
\tag{2}
$$

其中 $z$ 为商品，$\hat p$ 为消费者对商品的价值评估，$p$ 为支付价格，$p^{*}$ 为参考价格。获得效用衡量“以价格 $p$ 得到价值为 $\hat p$ 的商品”带来的收益；交易效用衡量“支付 $p$ 而非预期的 $p^{*}$”带来的得失感。消费者购买商品 $z$，当且仅当单位价格的价值超过其为该类支出设定的预算阈值：

$$
\frac{w(z,p,p^{*})}{p}>k_{it}.
\tag{3}
$$

式 (2) 表明价格 $p$ 同时出现在两个位置：它以绝对水平进入获得效用，又以与参考价的比较进入交易效用。因此，价格对购买的作用天然具有两条通道。

### 3.2.2 潜在净价值的推导

**第一步：取对数。** 式 (3) 两端为正，取对数得

$$
\log w(z,p,p^{*})-\log p>\log k_{it}.
\tag{4}
$$

**第二步：分离价格的两种作用。** 记 $\ell=\log p$，$\ell^{*}=\log p^{*}$。获得效用依赖商品价值与绝对价格，交易效用只依赖价格相对参考价的偏离 $\ell-\ell^{*}$，因此可写 $\log w=g\big(\hat w_{u,i},\,\ell,\,\ell-\ell^{*}\big)$，其中 $\hat w_{u,i}$ 概括用户 $u$ 对商品 $i$ 的价值评估。在品类典型价位处（$\ell=\mu_c$，且价格等于参考价 $\ell=\ell^{*}$）做一阶展开：

$$
\log w\approx g_0(\hat w_{u,i})+g_\ell\,(\ell-\mu_c)+g_d\,(\ell-\ell^{*}),
\qquad g_\ell\le 0,\ \ g_d\le 0,
\tag{5}
$$

其中 $g_\ell=\partial\log w/\partial\ell$ 反映价格升高对获得效用的削弱，$g_d=\partial\log w/\partial(\ell-\ell^{*})$ 反映价格高于参考价对交易效用的削弱。

**价值评估的经验依赖。** 式 (5) 中价值评估 $\hat w_{u,i}$ 并非静态。消费者对已购商品的评价随使用经验更新（Erdem & Keane, 1996），刚购入后的存货又会暂时降低再次购买的获得效用（Erdem, Imai & Keane, 2003）。记 $\hat w_{u,i}(t)=\hat w^{0}_{u,i}+\Delta\hat w_{u,i}(t)$，其中 $\Delta\hat w_{u,i}(t)$ 只依赖用户 $u$ 在 $t$ 之前对 $i$ 的购买记录。在 $\hat w^{0}_{u,i}$ 处一阶展开，$g_0(\hat w_{u,i}(t))\approx g_0(\hat w^{0}_{u,i})+S_{u,i}(t)$，$S_{u,i}(t)\equiv g_0'\,\Delta\hat w_{u,i}(t)$。因此获得效用中与价格无关的部分由“稳定匹配”与“状态依赖”两部分组成，后者即离散选择文献中的状态依赖（Heckman, 1981；Guadagni & Little, 1983；Keane, 1997），在推荐中对应复购与补货建模（Bhagat et al., 2018；Ren et al., 2019；Wang et al., 2019）。它不含价格，式 (8) 的价格两通道结构不变。

**第三步：代回判定规则。** 将式 (5) 代入式 (4)，并把 $-\log p=-(\ell-\mu_c)-\mu_c$ 并入：

$$
g_0-(1-g_\ell)(\ell-\mu_c)+g_d(\ell-\ell^{*})>\log k_{it}+\mu_c .
\tag{6}
$$

**第四步：标准化并定义各项。** 由式 (1)，$\ell-\mu_c=\sigma_c\tilde p$，$\ell-\ell^{*}=\sigma_c(\tilde p-\tilde p^{*})$。定义

$$
\mathrm{AU}_{u,i}(t)\equiv g_0,\qquad
\beta_{u,c}(t)\equiv(1-g_\ell)\,\sigma_c>0,\qquad
\gamma_{u,c}(t)\equiv -g_d\,\sigma_c\ge 0,\qquad
\mathrm{TU}_{u,i}(t)\equiv\tilde p^{*}_c-\tilde p_i,
\tag{7}
$$

即得**潜在净价值**

$$
W_{uit}=\underbrace{\mathrm{AU}_{u,i}(t)}_{\text{获得效用}}
-\underbrace{\beta_{u,c}(t)\cdot\tilde p_{i}}_{\text{价格支出压力}}
+\underbrace{\gamma_{u,c}(t)\cdot\mathrm{TU}_{u,i}(t)}_{\text{交易效用}} .
\tag{8}
$$

右端 $\log k_{it}+\mu_c$ 记为品类阈值 $k_c(t)$。展开的近似误差与决策中不可观测的成分（时机、情绪、未观测的替代品等）记为随机项 $\epsilon_{uit}$，购买规则成为

$$
Y_{uit}=\mathbf 1\big[\,W_{uit}>k_c(t)+\epsilon_{uit}\,\big].
\tag{9}
$$

推导给出三点对建模有直接指导意义的结论：

1. **价格支出压力恒为正。** 即使获得效用对价格不敏感（$g_\ell=0$），式 (3) 的“单位价格价值”形式仍使 $\beta\ge\sigma_c>0$。
2. **系数随用户与时间变化。** $g_\ell,g_d$ 是在用户当时的状态下求得的偏导数，故 $\beta,\gamma$ 带有下标 $u,c,t$，为个性化与时变建模提供了依据；获得效用的状态依赖 $S_{u,i}(t)$ 同样随用户经验变化。
3. **阈值包含品类价格水平。** $k_c(t)=\log k_{it}+\mu_c$：价格水平越高的品类越难触发购买。标准化价格 $\tilde p$ 去掉的品类间价格差异，在阈值中得到恢复（见 3.7 节）。

### 3.2.3 随机效用与选择概率

**二元形式。** 假设 $\epsilon_{uit}\sim\mathrm{Logistic}(0,1)$，则

$$
\Pr(Y_{uit}=1)=\Pr\big(\epsilon<W-k_c\big)=F_{\text{logistic}}(W-k_c)=\sigma(W-k_c),
\tag{10}
$$

代入式 (8)：

$$
\boxed{\ \Pr(Y_{uit}=1)=\sigma\Big(\mathrm{AU}_{u,i}(t)-\beta_{u,c}(t)\,\tilde p_{i}+\gamma_{u,c}(t)\,\mathrm{TU}_{u,i}(t)-k_c(t)\Big)\ }
\tag{11}
$$

sigmoid 因而不是任意选取的激活函数，而是“阈值判定规则 + logistic 扰动”的直接结果，$k_c$ 对应 Thaler 的预算阈值。

**多项形式。** 实际决策中，用户在候选集 $\mathcal J(t)$ 中选择一件商品。令每个候选的净效用为 $s_{uj}(t)=W_{ujt}-k_{c(j)}(t)$，扰动独立同分布于标准 Gumbel 分布，则由随机效用最大化（McFadden, 1974）得

$$
P(j\mid u,t)=\frac{\exp s_{uj}(t)}{\sum_{l\in\mathcal J(t)}\exp s_{ul}(t)} .
\tag{12}
$$

两个独立 Gumbel 变量之差服从 logistic 分布，因此式 (11) 正是式 (12) 在“购买 vs. 不购买”二选一时的特例；二者描述同一决策机制。本文以式 (12) 作为估计形式。对所有候选取值相同的项（如全局时间冲击、用户整体购买倾向）在式 (12) 中相互抵消，无需单独估计。

### 3.2.4 双通道的系数含义

由于 $\tilde p$ 同时出现在支出压力与交易效用中，将价格相关项改写为只含当期价格与参考价的形式：

$$
-\beta\,\tilde p+\gamma(\tilde p^{*}-\tilde p)
=-\underbrace{(\beta+\gamma)}_{\text{当期价格的总系数}}\tilde p+\underbrace{\gamma}_{\text{参考价的系数}}\tilde p^{*},
\tag{13}
$$

从而

$$
\frac{\partial W}{\partial\tilde p}\bigg|_{\tilde p^{*}\text{固定}}=-(\beta+\gamma),
\qquad
\frac{\partial W}{\partial\tilde p^{*}}\bigg|_{\tilde p\text{固定}}=\gamma .
\tag{14}
$$

| 系数 | 含义 | 理论对应 |
|---|---|---|
| $\beta+\gamma$ | 参考价固定时，价格变动一个单位引起的效用变化 | 短期价格敏感度 |
| $\gamma$ | 价格额外偏离参考价一个单位引起的效用变化 | 交易效用强度 |
| $\beta$ | 价格与参考价同步移动（偏离不变）时的效用变化 | 获得效用（支出压力）敏感度 |

由式 (12)，对选择概率的影响为

$$
\frac{\partial\log P(j)}{\partial\tilde p_j}=-(\beta+\gamma)\big(1-P(j)\big),
\qquad
\frac{\partial\log P(j)}{\partial\log p_j}=-\frac{\beta+\gamma}{\sigma_c}\big(1-P(j)\big).
\tag{15}
$$

前者为每个品类标准差的半弹性，后者为对数价格弹性，用于跨品类比较。品类层面的价格敏感度定义为品类内选择事件上式 (15) 的平均。

### 3.2.5 短期与长期效应

双通道结构蕴含一个可检验的动态性质。设品类 $c$ 的整体价格水平永久上移 $\Delta$（对数），则该品类商品的 $\tilde p$ 立即上移 $d=\Delta/\sigma_c$，参考价随后逐步追上：

$$
\text{短期：}\ \Delta W=-(\beta+\gamma)\,d,
\qquad
\text{长期（}\tilde p^{*}\to\tilde p^{*}+d\text{）：}\ \Delta W=-\beta\,d .
\tag{16}
$$

交易效用通道随参考价调整而消退，消退速度由参考价的记忆长度决定。这解释了一次性促销的效果强于持续低价，以及涨价冲击随时间减弱的现象。

---

## 3.3 因果结构与识别

### 3.3.1 因果图

图 1 给出两期展开的因果图。每期包含用户状态 $U$、价格 $P$、品类 $C$ 与购买 $Y$ 四个节点。期内，品类决定价格量级（$C\to P$）与基础购买倾向（$C\to Y$），价格与用户状态共同影响购买（$P\to Y$，$U\to Y$）。跨期，上期购买改变用户状态（$Y_{t-1}\to U_t$），上期价格经参考价影响当期购买（$P_{t-1}\to Y_t$）。

![图 1 因果图（两期展开）](fig_causal_graph.png)

对应的结构方程为

$$
\begin{aligned}
U_t&=f_U\big(Y_{<t},\ \varepsilon_{U,t}\big),\\
P_t&=f_P\big(C,\ \varepsilon_{P,t}\big),\\
Y_t&=f_Y\big(P_t,\ \tilde p^{*}(P_{<t}),\ C,\ U_t,\ \varepsilon_{Y,t}\big),
\end{aligned}
\tag{17}
$$

联合分布据此分解为

$$
p\big(C,\{U_t,P_t,Y_t\}_t\big)=p(C)\prod_t
\underbrace{p(U_t\mid Y_{<t})}_{\text{状态更新}}\,
\underbrace{p(P_t\mid C)}_{\text{价格生成}}\,
\underbrace{p(Y_t\mid P_t,\tilde p^{*}_t,C,U_t)}_{\text{选择机制}} .
\tag{18}
$$

需要估计的是第三个因子中价格的作用；前两个因子决定了估计时必须控制的变量。各条边与模型组件的对应关系如下。

| 边 | 含义 | 理论对应 | 模型组件 |
|---|---|---|---|
| $C\to P$ | 品类决定价格量级 | — | 品类内标准化 $\tilde p$，式 (1) |
| $C\to Y$ | 品类基础购买倾向 | 预算阈值 $k_{it}$ 的品类维度 | 品类阈值 $k_c(t)$，3.7 节 |
| $P\to Y$ | 当期价格的双通道作用 | 支出压力 + 交易效用 | $-\beta\tilde p+\gamma\,\mathrm{TU}$ |
| $U\to Y$ | 用户—商品匹配 | 获得效用的非价格部分 | 匹配项 $\mathrm{AU}$，3.9 节 |
| $P_{t-1}\to Y_t$ | 历史价格形成参考价 | $p^{*}$ 由过去价格形成 | 参考价 $\tilde p^{*}_c(d^-)$，3.3.4 节 |
| $Y_{t-1}\to U_t$（序列层） | 购买行为更新用户状态 | 阈值随时间变化 | 用户序列编码器，3.8 节 |
| $Y_{<t}\to U_t$（商品层） | 用户对具体商品的购买经验 | 价值评估的经验依赖 | 状态依赖项 $S_{uj}$，式 (41b) |

### 3.3.2 后门调整

价格的同期原因只有品类，从 $P_t$ 出发的后门路径 $P_t\leftarrow C\to Y_t$、$P_t\leftarrow C\to P_{t-1}\to Y_t$、$P_t\leftarrow C\to Y_{t-1}\to U_t\to Y_t$ 均经过 $C$。因此 $\{C\}$ 满足后门准则，对 $C$ 调整即可阻断 $P\leftarrow C\to Y$。在用户状态 $U=u$ 下（$U$ 不是 $P$ 的后代，条件化它不引入偏误），由后门调整定理（Pearl, 2009）：

$$
P\big(Y\mid do(P=p),\,U=u\big)=\sum_{c}P\big(Y\mid P=p,\,C=c,\,U=u\big)\,P\big(C=c\mid U=u\big).
\tag{19}
$$

在品类内部，干预分布等于条件分布：

$$
\mathbb E\big[Y\mid do(P=p),\,C=c\big]=\mathbb E\big[Y\mid P=p,\,C=c\big],
\qquad
\mathbb E\big[Y\mid do(P=p),\,C=c,\,U=u\big]=\mathbb E\big[Y\mid P=p,\,C=c,\,U=u\big].
\tag{20}
$$

左式只需品类即可识别品类层的价格效应，这是第一阶段无需用户信息的依据；右式在其上引入用户状态，刻画个体差异，对应第二阶段。研究目标为品类条件价格效应

$$
\tau(c)=\frac{\partial}{\partial p}\,\mathbb E\big[\log Y\mid do(P=p),\,C=c\big].
\tag{21}
$$

**状态依赖作为条件变量。** 第二阶段引入的条件变量包含用户对具体商品的购买经验 $S_{uj}(t)$，它是 $Y_{<t}$ 的确定函数，对应因果图中已有的商品层边 $Y_{<t}\to U_t$，只是把用户状态细化到商品层。由结构方程 (17)，$P_t$ 的唯一父节点是 $C$，$S_{uj}(t)$ 不是 $P_t$ 的后代，因而 $\{C,U_t,S_{uj}\}$ 仍满足后门准则，式 (19)、(20) 的识别结论不变。若放宽 (17) 中价格扰动独立的假设、允许商品层价格存在持续成分 $\xi_j$（$\xi_j\to P_{t-1}$，$\xi_j\to P_t$），则路径 $P_t\leftarrow\xi_j\to P_{t-1}\to Y_{t-1}\to S_{uj}\to Y_t$ 是一条后门路径；$S_{uj}$ 在其上是链节点，条件化它恰好阻断该路径，因此该条件变量使识别对价格持续性更稳健。

### 3.3.3 第一阶段的结构方程

用户状态维度高，直接估计式 (19) 不可行。假设结构方程在对数尺度上可拆为价格块与其余部分：

$$
\mathbb E\big[\log Y_{it}\mid P,C,U\big]
=\underbrace{\beta_A\,\tilde p_{it}+\beta_T\,\mathrm{Dev}_{it}}_{\text{价格块 }g}
+\underbrace{m\big(c(i),U_{it}\big)}_{\text{其余部分}}+\lambda_t,
\qquad \mathrm{Dev}_{it}=\tilde p_{it}-\tilde p^{*}_{c,t},
\tag{22}
$$

其中 $\beta_A=-\beta$，$\beta_T=-\gamma$ 为对数需求对价格的斜率。由式 (19)，

$$
\tau(c)=\frac{\partial}{\partial\tilde p}\int\big[g+m(c,u)\big]\,p(u\mid c)\,\mathrm du
=\int\frac{\partial g}{\partial\tilde p}\,p(u\mid c)\,\mathrm du ,
\tag{23}
$$

求导消去了 $m$，因此 $m$ 内部品类与用户如何交互均不影响价格效应。$m$ 的品类部分与时不变部分由商品与品类层面的项吸收，得到第一阶段的估计方程

$$
\mathbb E[\log Y_{it}]=\beta_A\,\tilde p_{it}+\beta_T\,\mathrm{Dev}_{it}+m_{i}+\lambda_t .
\tag{24}
$$

式 (24) 与式 (12) 是同一模型的两种写法。取时段为日，设第 $d$ 日共有 $N_d$ 次选择，商品 $j$ 的期望购买量为 $N_dP_d(j)$，由式 (12)：

$$
\log\mathbb E[Y_{jd}]
=\underbrace{-\beta\,\tilde p_{jd}-\gamma\big(\tilde p_{jd}-\tilde p^{*}_{c,d}\big)}_{\beta_A\tilde p+\beta_T\mathrm{Dev}}
+\underbrace{\mathrm{AU}_{jd}-k_{c,d}}_{m}
+\underbrace{\log N_d-\log\textstyle\sum_{l}e^{s_{ld}}}_{\lambda_d}.
\tag{25}
$$

时间效应 $\lambda_d$ 对应 softmax 的归一化项。带时间效应的泊松对数线性模型与多项 logit 的似然等价（Baker, 1994）。第一阶段中同一日的全部记录共享同一组品类状态、参考价与候选价格（3.1 节的时间约定），逐记录的多项似然与按日计数的似然因此严格相等（见式 (33)），第一阶段直接在日计数上估计式 (24)。

### 3.3.4 参考价的形成

参考价对应跨期边 $P_{t-1}\to Y_t$，由品类过去的交易价格指数衰减加权得到。预测日 $d$ 的参考价为

$$
\tilde p^{*}_c(d^-)=\frac{\sum_{m:\,t_m<d,\ c_m=c}w_m\,\tilde p_m}{\sum_{m:\,t_m<d,\ c_m=c}w_m},
\qquad w_m=2^{-(d-t_m)/h},
\tag{26}
$$

$h$ 为半衰期。求和范围只含 $d$ 之前的训练记录，与候选价格使用同一信息截点。权重中的评估时刻只改变分子分母共同的比例因子，对比值没有影响，因此参考价可以在该品类最近一条记录处递推计算，结果与式 (26) 相同。式 (26) 是自适应期望 $\tilde p^{*}_t=\alpha\,\tilde p^{*}_{t-1}+(1-\alpha)\bar p_{t-1}$ 的连续时间形式，$\alpha$ 越大参考价记忆越长。

由式 (13)，两条通道可以分离的条件是参考价随时间变动：若 $\tilde p^{*}$ 为常数，$\mathrm{TU}$ 与 $\tilde p$ 完全共线。$\beta+\gamma$ 由同一商品价格的时间变动识别，$\gamma$ 由品类参考价的变动识别。

### 3.3.5 从理论到两阶段模型

上述推导确定了模型的结构：

- **第一阶段（品类偏好）。** 依据式 (20) 左式，从去除用户标识的品类日度交易面板中学习品类状态，输出品类层的 $\beta_c(d)$、$\gamma_c(d)$、$k_c(d)$ 与参考价 $\tilde p^{*}_c(d^-)$，估计式 (24)。
- **第二阶段（兴趣偏好）。** 依据式 (20) 右式，读入用户的行为序列以表示 $U_t$（对应 $Y_{t-1}\to U_t$），在品类参数上叠加个体残差，并以用户对具体商品的经验构成状态依赖项 $S_{uj}$，得到式 (11) 中的 $\mathrm{AU}_{u,i}(t)$、$\beta_{u,c}(t)$、$\gamma_{u,c}(t)$。

两阶段共用由式 (8)、(9)、(12) 确定的打分函数，只在各项的计算方式上不同。

---

## 3.4 统一打分函数

对用户 $u$、候选商品 $j$（品类 $c=c(j)$）与预测时刻 $t$（所在日 $d=d(t)$），打分函数为

$$
s_{uj}(t)=\mathrm{AU}_{uj}(t)-\beta_{u,c}(t)\,\tilde p_j(d^-)+\gamma_{u,c}(t)\big[\tilde p^{*}_c(d^-)-\tilde p_j(d^-)\big]-k_{u,c}(t).
\tag{27}
$$

选择概率由式 (12) 给出。各项在两阶段中的实现见下表。

| 理论量 | 第一阶段（品类层） | 第二阶段（用户层） | 节 |
|---|---|---|---|
| $\mathrm{AU}$ | 商品偏置 + 群体匹配 | 再加用户相对品类先验的偏离 + 状态依赖 $S_{uj}$ | 3.9 |
| $\beta$ | $\mathrm{softplus}(a_c(d))$ | $\mathrm{softplus}(a_c(d)+\Delta a_{u,c})$ | 3.7, 3.8 |
| $\gamma$ | $\mathrm{softplus}(b_c(d))$ | $\mathrm{softplus}(b_c(d)+\Delta b_{u,c})$ | 3.7, 3.8 |
| $k$ | $\eta\tilde\mu_c+\delta_c+h_k(\mathbf g_c)$ | 再加 $\Delta k_{u,c}$ | 3.7, 3.8 |
| $\tilde p^{*}$ | 式 (26) | 同左 | 3.3.4 |
| $U_t$ | — | 用户序列分块编码与按品类读出 | 3.8 |

---

## 3.5 共享基础编码

两阶段共享商品、品类、价格与时间四类编码。商品与品类采用查表嵌入 $\mathbf e^{\mathrm{item}}_j=E_{\mathrm{item}}[i_j]$，$\mathbf e_c=E_{\mathrm{cat}}[c]$。

**价格编码。** 价格同时承载两类信息：商品在品类内的相对位置 $\tilde p_j$，以及跨品类可比的绝对价位 $\ell_j=(\log p_j-\mu)/\sigma$（$\mu,\sigma$ 为全体商品的对数价格统计量）。二者分别以分段线性基 $\mathbf b(\cdot)$ 展开（结点取训练集分位数；Gorishniy et al., 2022），并按品类做特征调制（FiLM；Perez et al., 2018），使“同样高出一个标准差”在不同品类中可以具有不同含义：

$$
\mathbf e^{\mathrm{price}}_j=\big(\mathbf 1+\mathbf W_\gamma\mathbf e_{c_j}\big)\odot\mathbf W_p\big[\mathbf b(\tilde p_j);\ \mathbf b(\ell_j);\ m_j\big]+\mathbf W_\beta\mathbf e_{c_j},
\tag{28}
$$

其中 $m_j$ 为价格缺失指示。$\mathbf W_\gamma,\mathbf W_\beta$ 零初始化，训练起点各品类的价格编码相同，品类间的差异由数据学得。

**时间编码。** 序列顺序由注意力中的相对位置项表达；相邻记录的时间间隔编码为 $\mathbf e^{\mathrm{time}}_j=\mathbf W_t\log(1+\Delta_j)$，其中 $\Delta_j$ 在品类面板中是该记录与同品类上一条记录的间隔，在用户序列中是与该用户上一条记录的间隔。

价格编码只进入事件表示（品类面板中的交易与用户历史）。候选商品的当前价格仅通过式 (27) 的价格项进入打分，从而保持式 (22) 中价格块与其余部分的可分离性。

---

## 3.6 第一阶段：品类偏好提取

第一阶段从每个品类的日度交易面板中提取随日历时间漂移的群体偏好，输出品类状态 $\mathbf g_c(d^-)$、品类层价格参数与参考价。

**事件表示。** 记 $\mathcal E_{c,d}$ 为品类 $c$ 在第 $d$ 日的训练记录（去除用户字段），$n_{c,d}=\lvert\mathcal E_{c,d}\rvert$。每条记录表示为

$$
\mathbf x^{(1)}_{c,m}=\tanh\big(\mathbf W_1[\mathbf e^{\mathrm{item}}_m\,\Vert\,\mathbf e^{\mathrm{price}}_m\,\Vert\,\mathbf e^{\mathrm{time}}_m]+\mathbf b_1\big).
\tag{29}
$$

**日度汇总。** 同一日内来自不同匿名用户的购买，其先后顺序不携带品类状态演化的信息，日内记录可交换。因此以置换不变的均值汇总，并保留交易量（Zaheer et al., 2017）：

$$
\mathbf u_{c,d}=\tanh\Big(\mathbf W_u\Big[\tfrac{1}{\max(n_{c,d},1)}\textstyle\sum_{m\in\mathcal E_{c,d}}\mathbf x^{(1)}_{c,m}\ \Big\Vert\ \log(1+n_{c,d})\Big]+\mathbf b_u\Big),\qquad d\ge d^0_c ,
\tag{30}
$$

其中 $d^0_c$ 为品类 $c$ 首次出现交易的日；此后没有交易的日以 $n_{c,d}=0$ 进入面板。所有训练记录都进入面板，不论其标签在似然中是否可用（见式 (33)）。

**序列编码。** 品类 $c$ 的面板 $\mathbf U_c=(\mathbf u_{c,d^0_c},\dots,\mathbf u_{c,T-1})$ 上采用因果自注意力（Kang & McAuley, 2018），位置即日历日，在注意力 logit 上加入相对位置项：

$$
\mathbf H_c=\mathrm{CausalSA}\big(\mathbf U_c;\ \Omega\big),
\qquad
\Omega_{dd'}=\omega\big(\mathrm{bucket}(d-d')\big),\quad d'\le d .
\tag{31}
$$

相对距离近处逐日分桶、远处按对数分桶（Shaw et al., 2018; Raffel et al., 2020）。带相对位置偏置的注意力相当于对近期日度分布做可学习的核平滑（Tsai et al., 2019），近期权重高、远期衰减，衰减形状由数据决定。位置以日历日而非记录序号度量，距离因而不受交易量波动影响；参考价式 (26) 同样按日历时间衰减，两者的记忆长度可以直接比较。面板长度不超过 $T$，注意力的规模只由训练期天数决定，与品类的交易量无关。

**输出。** 预测日 $d$ 的品类状态读取前一日的编码输出：

$$
\mathbf g_c(d^-)=\tanh\big(\mathbf W_g[\mathbf h_{c,\min(d,T)-1}\,\Vert\,\mathbf e_c\,\Vert\,\phi_t(d-t^{\mathrm{last}}_c(d))]+\mathbf b_g\big),
\qquad
\boldsymbol\theta_c(d^-)=(a_c,\,b_c,\,k_c)=\mathrm{Head}\big(\mathbf g_c(d^-)\big),
\tag{32}
$$

其中 $t^{\mathrm{last}}_c(d)$ 为第 $d$ 日之前该品类最后一笔训练交易的时间，$\phi_t$ 编码距该时间的间隔；品类在 $d$ 之前没有交易时，状态输入与间隔项取 0。训练期之后的日读取训练期最后一日的编码输出，即市场状态在验证与测试期冻结。参数头见 3.7 节，参考价按式 (26) 计算。

**训练。** 同一日的所有记录共享品类状态、参考价与候选价格，式 (12) 的逐记录似然可按日精确改写。记 $y_{dj}$ 为第 $d$ 日商品 $j$ 的价格可得的购买次数（候选价格不可得的记录不进入似然），$n_d=\sum_jy_{dj}$，则

$$
\mathcal L_1=-\sum_m\log P_1(i_m\mid d_m)
=\sum_{d<T}\Big[n_d\log\!\!\sum_{l\in\mathcal J(d)}\!e^{s^{(1)}_l(d)}-\sum_jy_{dj}\,s^{(1)}_j(d)\Big],
\qquad
s^{(1)}_j(d)=\mathrm{AU}^{(1)}_j-\beta_c\,\tilde p_j+\gamma_c\big(\tilde p^{*}_c-\tilde p_j\big)-k_c .
\tag{33}
$$

右端正是式 (24) 以日为时段的计数似然（3.3.3 节）。训练时每一步随机抽取若干日计算上式，品类面板在每一步按当前参数整体重算。

---

## 3.7 价格参数头

价格参数头将品类状态映射为式 (27) 中的 $\beta$、$\gamma$ 与 $k$：

$$
\begin{aligned}
a_c(d)&=\bar a+\mathbf w_a^{\top}\big(\mathbf g_c(d^-)-\bar{\mathbf g}\big), &\beta_c(d)&=\mathrm{softplus}\big(a_c(d)\big),\\
b_c(d)&=\bar b+\mathbf w_b^{\top}\big(\mathbf g_c(d^-)-\bar{\mathbf g}\big), &\gamma_c(d)&=\mathrm{softplus}\big(b_c(d)\big),\\
k_c(d)&=\eta\,\tilde\mu_c+\delta_c+\mathring h_k\big(\mathbf g_c(d^-)\big), &\tilde\mu_c&=(\mu_c-\mu)/\sigma,\quad \eta\ge 0 .
\end{aligned}
\tag{34}
$$

**价格系数。** $\mathbf g_c$ 包含品类嵌入且随时间变化，因此单一线性项同时刻画系数在品类间与时间上的差异；softplus 保证系数为正，与式 (7) 一致。$\bar{\mathbf g}$ 为训练期品类状态的滑动平均（不参与反向传播），中心化后 $\bar a,\bar b$ 恰为平均品类、平均时刻的基准系数。对 $\mathbf w_a,\mathbf w_b$ 施加权重衰减，等价于以基准系数为中心的高斯先验，使样本稀疏的品类向总体收缩（James & Stein, 1961; Efron & Morris, 1975; Gelman & Hill, 2007）。$\bar a,\bar b$ 不施加衰减。

**品类阈值。** 式 (34) 的阈值由三部分组成，与 3.2.2 节 $k_c(t)=\log k_{it}+\mu_c$ 的推导一一对应：

- $\eta\tilde\mu_c$ 对应 $\mu_c$，即品类自身的价格水平，补回了品类内标准化去掉的跨品类价格差异。$\tilde\mu_c$ 为训练期的静态量。
- $\delta_c$ 为其余品类差异，施加 $\ell_2$ 惩罚，使 $\eta$ 由“品类偏置随价格水平变化的斜率”确定（Mundlak, 1978）。
- $\mathring h_k(\mathbf g_c)$ 为小型 MLP，对应 $\log k_{it}$ 的时间变化，如品类热度与季节性。其跨品类均值在式 (12) 的 softmax 中相互抵消，故对 $\mathbf g_c$ 的 MLP 输出按品类中心化 $\mathring h_k(\mathbf g_c)=h_k(\mathbf g_c)-\frac1{|\mathcal C|}\sum_{c'}h_k(\mathbf g_{c'})$，这是与商品偏置品类内中心化同类的归一化（见 3.9 节），去掉一个不改变任何打分、但在数值上可自由平移的方向。

阈值中的价格水平部分取静态值，因为参考价已承载品类价格水平的时间变化，二者若同时时变将无法区分。

**按日预计算。** 式 (34) 以及 3.8、3.9 节中只依赖品类状态的各项，对每个（日，品类）组合只需计算一次，查询时按其所在日索引。

---

## 3.8 第二阶段：兴趣偏好提取

第二阶段以用户的行为序列表示用户状态 $U_t$，按候选品类读出用户表示，并在品类参数上叠加个性化残差。

**事件表示。** 用户 $u$ 在 $t$ 之前的记录按发生顺序排列：

$$
\mathcal H_u(t)=\big((c_1,i_1,p_1,t_1),\dots,(c_{m_u},i_{m_u},p_{m_u},t_{m_u})\big),
$$

$$
\mathbf x^{(2)}_{u,j}=\tanh\big(\mathbf W_2[\mathbf e_{c_j}\Vert\mathbf e^{\mathrm{item}}_j\Vert\mathbf e^{\mathrm{price}}_j\Vert\mathbf e^{\mathrm{time}}_j]+\mathbf b_2\big).
\tag{35}
$$

**分块编码。** 用户序列按绝对位置分为长度为 $W$ 的块，块 $b$ 覆盖位置 $[bW,(b+1)W)$。块 $b$ 内的事件以因果自注意力编码，并以块 $b-1$ 的键值为记忆（Dai et al., 2019）；块 $b-1$ 的键值来自它自身的编码，不带更早的记忆，且记忆不传梯度。注意力 logit 上加入位置与时间间隔两项：

$$
\Omega^{u}_{jl}=\omega_{\mathrm{pos}}\big(\mathrm{bucket}(j-l)\big)+\omega_{\mathrm{gap}}\big(\mathrm{bucket}(t_j-t_l)\big),
\tag{36}
$$

第一项为相对位置（Zhang et al., 2022），第二项为时间间隔（Li et al., 2020）。位置 $p$（即该查询之前的记录数）处的查询可读的上下文为

$$
\mathcal R_u(p)=\big\{\,j:\ \max(\lfloor p/W\rfloor-1,\,0)\,W\le j<p\,\big\},
\qquad
W\le\lvert\mathcal R_u(p)\rvert<2W\quad(p\ge W).
\tag{37}
$$

编码器是因果的，$j\in\mathcal R_u(p)$ 的状态 $\mathbf h_{u,j}$ 只依赖位置 $\le j<p$ 的事件，因此查询不会读到自身或之后的信息。$W$ 是控制个性化上下文长度的超参数。训练以“用户 × 块”为样本，块内全部训练查询在一次前向中同时计算（Kang & McAuley, 2018）；评估时位置 $p$ 处的查询按同一规则构造上下文，训练与评估使用相同的读取方式。

**按品类读出。** 历史中哪些行为重要取决于正在打分的品类，也取决于该行为距当前查询多久。对每个候选品类 $c$，以品类嵌入与品类状态构造查询（Zhou et al., 2018），与历史状态做乘性打分，并加入与式 (36) 同构、相对当前查询的相对位置与时间间隔偏置，再加入以 $\mathbf g_c$ 为值的先验项：

$$
q_{u,j,c}=\frac{(\mathbf W_h\mathbf h_{u,j})^{\top}\big(\mathbf W_c\mathbf e_c+\mathbf W_{gc}\mathbf g_c(d^-)\big)}{\sqrt{d_a}}
+\omega^{r}_{\mathrm{pos}}\big(\mathrm{bucket}(p-j)\big)+\omega^{r}_{\mathrm{gap}}\big(\mathrm{bucket}(t-t_j)\big),
\qquad
q_{0,c}=\mathbf v_0^{\top}\mathbf g_c(d^-),
\tag{38}
$$

$$
\mathbf h_{u,c}(t)=\frac{\sum_{j\in\mathcal R_u(p)} e^{q_{u,j,c}}\,\mathbf W_V\mathbf h_{u,j}+e^{q_{0,c}}\,\mathbf W_0\,\mathbf g_c(d^-)}{\sum_{j\in\mathcal R_u(p)} e^{q_{u,j,c}}+e^{q_{0,c}}} .
\tag{39}
$$

此处的位置与时间都相对于当前查询（位置 $p$、时刻 $t$，训练与评估都取目标行的时间戳），使读出可以区分“最近的行为”与“很久以前的行为”；历史状态 $\mathbf h_{u,j}$ 自身的时间编码只表达相邻记录间隔，不含到预测时刻的距离，因而需要这两项相对查询的偏置。偏置在掩码之前加入、只作用于可读键，先验槽 $q_{0,c}$ 不变。式 (38) 的维度 $d_a$ 只用于产生标量权重，可以小于隐层维度 $d$；被读出的 $\mathbf W_V\mathbf h_{u,j}$ 与先验 $\mathbf W_0\mathbf g_c$ 保持 $d$ 维。打分取乘性形式的原因是：若取加性形式 $\mathbf v^{\top}\tanh(\mathbf a_j+\mathbf b_c)$，在未饱和区间内一阶项中的品类部分 $\mathbf v^{\top}\mathbf b_c$ 对所有历史记录取值相同，在对 $j$ 的 softmax 中抵消，品类相关的选择只来自高阶项；乘性形式在最低阶就包含历史与品类的交互 $\mathbf a_j^{\top}\mathbf b_c$（Luong et al., 2015; Vaswani et al., 2017）。

式 (39) 具有收缩估计 $(N\bar{\mathbf v}+\lambda\boldsymbol\mu)/(N+\lambda)$ 的形式，有效证据量 $N=\sum_je^{q_{u,j,c}}$，伪计数 $\lambda=e^{q_{0,c}}$：用户在品类 $c$ 上的相关历史越多，表示越依赖其本人；没有相关历史时，表示退回到品类先验 $\mathbf W_0\mathbf g_c$。先验项始终存在，所以空历史下式 (39) 有定义，且恰好等于先验。

**个性化残差。**

$$
(\Delta a,\Delta b,\Delta k)_{u,c}=f_\Delta\big([\mathbf h_{u,c}\Vert\mathbf e_c\Vert\mathbf g_c]\big),
\quad
\beta_{u,c}=\mathrm{softplus}(a_c+\Delta a),
\quad
\gamma_{u,c}=\mathrm{softplus}(b_c+\Delta b),
\quad
k_{u,c}=k_c+\mathring{\Delta k} .
\tag{40}
$$

这对应式 (7) 中系数随用户状态变化的结论。$f_\Delta$ 末层零初始化，使第二阶段从第一阶段的品类基准出发。与阈值中的 $h_k$ 同理，$\Delta k$ 对同一查询的所有候选是跨品类常数时在式 (12) 中抵消，因此按品类中心化 $\mathring{\Delta k}_{u,c}=\Delta k_{u,c}-\frac1{|\mathcal C|}\sum_{c'}\Delta k_{u,c'}$。$f_\Delta$ 的第一层对输入按块线性，作用于 $[\mathbf e_c\Vert\mathbf g_c]$ 的部分只依赖（日，品类），按 3.7 节的约定预先计算，对每个（查询，品类）只剩作用于 $\mathbf h_{u,c}$ 的一次乘法。

---

## 3.9 匹配项

匹配项 $\mathrm{AU}$ 对应获得效用中与价格无关的部分，由群体匹配、用户偏离与状态依赖三部分组成：

$$
\mathrm{AU}_{uj}(t)=\underbrace{b_j+\mathbf e_j^{\top}\mathbf W_g\,\mathbf g_c(d^-)}_{\mathrm{AU}^{(1)}_j:\ \text{第一阶段群体匹配}}
+\underbrace{\mathbf e_j^{\top}\mathbf W_u\big(\mathbf h_{u,c}(t)-\mathbf W_0\,\mathbf g_c(d^-)\big)}_{\text{用户相对品类先验的偏离}}
+\underbrace{S_{uj}(t)}_{\text{状态依赖，式 (41b)}},
\qquad c=c(j).
\tag{41}
$$

- $b_j$ 为商品偏置，刻画商品自身的热度并吸收其固定价位，因此价格系数由同一商品价格随时间的变动识别。$b_j$ 在品类内中心化（$\sum_{j\in c}b_j=0$），品类层面的水平统一由 $k_c$ 表达。
- $\mathbf e_j^{\top}\mathbf W_g\mathbf g_c$ 刻画当前市场在品类 $c$ 中偏好的商品，第一阶段只使用这一部分。
- $\mathbf h_{u,c}-\mathbf W_0\mathbf g_c$ 为用户相对品类先验的偏离。无相关历史时该差值为零，匹配项自动退回群体匹配；$\mathbf W_u$ 零初始化，阶段切换时该项为零。
- $S_{uj}(t)$ 为状态依赖项，刻画用户对该商品的购买经验（复购与补货周期），只在第二阶段出现。

**状态依赖项。** 由 3.2.2 节，获得效用中随经验变化的部分为 $S_{uj}(t)$。化妆品的复购与补货周期是强信号，且复购时间间隔的分布因品类而异。将其写为

$$
S_{uj}(t)=\mathbf 1\big[n_{uj}(t)>0\big]\Big[\kappa\big(\mathrm{bk}(t-t^{\mathrm{last}}_{uj})\big)+\kappa_{c}\big(\mathrm{bk}(t-t^{\mathrm{last}}_{uj})\big)+\nu\big(\lfloor\log_2 n_{uj}(t)\rfloor\big)+\mathbf w_\rho^{\top}\mathbf h_{u,c}(t)\Big],\qquad c=c(j),
\tag{41b}
$$

其中 $n_{uj}(t)$ 为 $u$ 在 $t$ 之前购买 $j$ 的次数，$t^{\mathrm{last}}_{uj}$ 为最近一次购买时间，$\mathrm{bk}(\cdot)$ 与式 (36) 的时间间隔分桶相同（按小时，近处逐小时、远处对数分桶，同篮记录落入 0 号桶）。$\kappa$ 为全局补货核，$\kappa_c$ 为品类偏离并施加 $\ell_2$ 收缩（与 $\delta_c$ 同理，样本稀疏的品类向全局核收缩），$\nu$ 为次数效应，$\mathbf w_\rho^{\top}\mathbf h_{u,c}$ 让“忠诚/求异”倾向随用户状态个性化。核可以取负值：短间隔的负值对应存货与餍足，同篮的负值对应“已放入购物车的商品不会再选一次”。$S_{uj}$ 是 $Y_{<t}$ 的确定函数，对应因果图中已有的商品层边 $Y_{<t}\to U_t$，相对当期价格前定（见 3.3.2 节）；其全部参数零初始化，阶段切换时 $S\equiv0$。

匹配项只做商品层匹配：用户对品类整体的偏好由 $\Delta k_{u,c}$ 表达，候选的当前价格由价格项表达，用户对具体商品的经验由 $S_{uj}$ 表达，使式 (27) 中每种含义只出现一次。$b_j$ 为商品的固定热度，点积项为品类内的偏好匹配，$S_{uj}$ 为用户对该商品的经验。采用点积形式（Koren et al., 2009; Rendle et al., 2020），候选按品类排列并切成不跨品类的定宽块后，每块只需读取一个品类的匹配向量，全候选集打分化为按块的批量矩阵乘；$S_{uj}$ 的稀疏（查询，商品）值按同一块结构写入，使 softmax 的分母与正例分子来自同一张量。

---

## 3.10 训练目标、估计与阶段衔接

两个阶段均在全部合格候选上最大化式 (12) 的对数似然：

$$
\mathcal L_k=-\sum\log P_k(\cdot)+\lambda_w\big(\|\mathbf w_a\|^2+\|\mathbf w_b\|^2\big)+\lambda_\delta\Big(\sum_c\delta_c^2+\sum_c\|\kappa_c\|^2\Big),
\qquad k\in\{1,2\},
\tag{42}
$$

第一阶段的似然项为式 (33)：求和遍历训练日，以品类状态代替用户表示，每一步随机抽取若干日，品类面板按当前参数整体重算。第二阶段的似然项为 $-\sum_{(u,i,t)}\log P_2(i\mid u,t)$，求和遍历用户块中的训练查询；每个查询的 softmax 取在其所在日的合格候选集上，仅保留目标在该候选集内的查询（这些行仍留在用户历史中，只是不作为似然项），与第一阶段“候选价格不可得的记录不进入似然”一致。正则强度由验证集选择。

**第一阶段的起点。** 第一阶段以固定效应嵌套模型（只含商品效应 $b_j$、品类阈值 $k_c$ 与日效应）的极大似然解为起点。该模型是带商品效应与日效应的泊松对数线性模型（式 (24)），日效应被 softmax 的归一化项吸收（Baker, 1994），商品效应的集中似然一阶条件给出不动点迭代（Hausman, Hall & Griliches, 1984）

$$
\rho_j\leftarrow\log(Y_j+\alpha)-\log\sum_{d}\frac{n_d\,\mathbf 1[j\in\mathcal J(d)]}{Z_d(\boldsymbol\rho)},
\qquad
Z_d(\boldsymbol\rho)=\sum_{l\in\mathcal J(d)}e^{\rho_l},
\tag{43}
$$

其中 $Y_j$ 为商品 $j$ 的合格训练购买次数，$n_d$ 为第 $d$ 日的合格标签总数，$\alpha$ 为伪计数。令 $k_c=-\overline{\rho}_c$（品类内均值取负）、$b_j=\rho_j+k_c$，使 $b_j$ 恰好在品类内中心化、初始打分 $b_j-k_c=\rho_j$；再按 3.7 节的 Mundlak 分解，以品类购买量为权重对 $k_c$ 关于 $\tilde\mu_c$ 做加权最小二乘，斜率取 $\max(\hat\eta,\eta_{\min})$ 作为 $\eta$ 的初值、残差作为 $\delta_c$ 的初值。此时 $h_k$ 末层与 $\mathbf w_a,\mathbf w_b$ 为零，价格项很小，训练起点即固定效应嵌套模型的极大似然点，第一阶段只需学习相对它的时间漂移、价格效应与群体匹配。式 (34) 的参数化、$\delta_c$ 的 $\ell_2$ 收缩与 $\eta$ 的识别方式都不变；只在无标签日合格、从而没有曝光的商品不参与估计，赋予最小值。

**优化。** 采用 Adam，参数分三组：编码器、读出层、$f_\Delta$、状态依赖与读出近因的参数为一组，全程学习率 $\mathrm{lr}$；商品嵌入 $\mathbf e_j$ 与商品偏置 $b_j$ 为一组，两阶段均为 $\mathrm{lr}$；品类嵌入、价格编码、时间编码、品类状态读出、群体匹配与价格参数头为一组，第一阶段为 $\mathrm{lr}$，第二阶段为 $\mathrm{lr}\cdot r_{\mathrm{base}}$（$r_{\mathrm{base}}<1$，以保护第一阶段学到的品类结构）。学习率采用线性 warmup 后余弦退火到一个下限（Loshchilov & Hutter, 2017）。第二阶段对权重做滑动平均（EMA；Polyak & Juditsky, 1992），验证、选模与最终评估均使用 EMA 权重，以降低选中某一轮恰好偏高或偏低的方差。每个阶段保留验证 N@10 最佳的检查点。

**阶段衔接。** 第二阶段以第一阶段的参数初始化：品类面板编码器、日度汇总层、共享嵌入与价格参数头沿用第一阶段的结果，$\bar{\mathbf g}$ 固定为第一阶段结束时的值；用户编码器复制面板编码器，个体残差 $f_\Delta$、$\mathbf W_u$ 与状态依赖、读出近因的全部参数零初始化。第二阶段每一步按当前参数重算品类面板，面板编码器的输出停止梯度，其后的状态读出层与价格参数头继续训练。因此切换瞬间第二阶段的打分与第一阶段逐位一致，此后用户信息只以残差与状态依赖的形式逐步修正品类基准。

**可解释量与嵌套检验。** 训练完成后，模型直接给出各品类、各日的 $\beta_c(d)$、$\gamma_c(d)$ 及其个体分布，由式 (15) 得到的品类价格弹性，以及阈值中的品类价格水平系数 $\eta$。各机制是否存在，由在完整模型上逐一加约束的嵌套模型比较检验：$\gamma\equiv0$（无交易效用）、$\mathbf w_a=\mathbf w_b=\mathbf 0$（价格敏感度无品类/时间异质性）、$S\equiv0$（无状态依赖）、读出近因偏置为零。每个约束项保持零初值且不训练，因而在打分中精确为零。

---

## 3.11 计算复杂度

记训练记录数为 $N$，训练查询数为 $Q$，品类数为 $C$，商品数为 $I$，训练期天数为 $T$，协议覆盖的总天数为 $D$，隐层维度为 $d$，读出打分维度为 $d_a$，$\bar R<2W$ 为查询的平均可读上下文长度，$\bar R_{\mathrm{rep}}$ 为查询的平均经验商品数（上界为复购窗口容量）。

**第一阶段。** 每一步的开销为

$$
O\big(N d^2+C\,T^2 d+D\,C\,d^2+\lvert\mathcal D_b\rvert\,I\,d\big),
$$

四项依次为事件表示、品类面板注意力、按（日，品类）预计算的价格参数头与匹配向量、所抽取的 $\lvert\mathcal D_b\rvert$ 个日的全目录打分。该开销与单个品类的交易规模无关，市场状态也不需要任何跨步缓存。固定效应初始化为一次性预处理，开销 $O(N+DI)$。

**第二阶段。** 每轮的开销为

$$
O\big(N\,W d+Q\,C\,\bar R\,(d_a+d)+Q\,C\,d^2+Q\,I\,d+Q\,\bar R_{\mathrm{rep}}\big),
$$

依次为用户分块编码、按品类读出（含读出近因偏置）、个性化残差、全目录打分与状态依赖项。同一用户的序列在每一轮中只被编码一次，查询之间共享编码；读出近因偏置与状态依赖项均为低阶附加，不改变复杂度类别。

两个阶段对交互数都是线性的。主导项 $Q\,C\,\bar R\,(d_a+d)$ 与全目录 softmax 的 $Q\,I\,d$ 同阶，与以全目录交叉熵训练的自注意力序列推荐模型（每轮约 $O(NLd+NId)$）属于同一复杂度类。

---

## 参考文献

- Baker, S. G. (1994). The multinomial-Poisson transformation. *The Statistician*, 43(4), 495–504.
- Bhagat, R., Muralidharan, S., Lobzhanidze, A., & Vishwanath, S. (2018). Buy it again: Modeling repeat purchase recommendations. *KDD*.
- Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q., & Salakhutdinov, R. (2019). Transformer-XL: Attentive language models beyond a fixed-length context. *ACL*.
- Efron, B., & Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *JASA*, 70(350), 311–319.
- Erdem, T., Imai, S., & Keane, M. P. (2003). Brand and quantity choice dynamics under price uncertainty. *Quantitative Marketing and Economics*, 1(1), 5–64.
- Erdem, T., & Keane, M. P. (1996). Decision-making under uncertainty: Capturing dynamic brand choice processes in turbulent consumer goods markets. *Marketing Science*, 15(1), 1–20.
- Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- Gorishniy, Y., Rubachev, I., & Babenko, A. (2022). On embeddings for numerical features in tabular deep learning. *NeurIPS*.
- Guadagni, P. M., & Little, J. D. C. (1983). A logit model of brand choice calibrated on scanner data. *Marketing Science*, 2(3), 203–238.
- Hausman, J., Hall, B. H., & Griliches, Z. (1984). Econometric models for count data with an application to the patents–R&D relationship. *Econometrica*, 52(4), 909–938.
- Heckman, J. J. (1981). Heterogeneity and state dependence. In S. Rosen (Ed.), *Studies in Labor Markets* (pp. 91–140). University of Chicago Press.
- James, W., & Stein, C. (1961). Estimation with quadratic loss. *Proc. Fourth Berkeley Symposium*, 1, 361–379.
- Kang, W.-C., & McAuley, J. (2018). Self-attentive sequential recommendation. *ICDM*.
- Keane, M. P. (1997). Modeling heterogeneity and state dependence in consumer choice behavior. *Journal of Business & Economic Statistics*, 15(3), 310–327.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30–37.
- Li, J., Wang, Y., & McAuley, J. (2020). Time interval aware self-attention for sequential recommendation. *WSDM*.
- Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR*.
- Luong, M.-T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. *EMNLP*.
- McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In *Frontiers in Econometrics*.
- Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69–85.
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI*.
- Polyak, B. T., & Juditsky, A. B. (1992). Acceleration of stochastic approximation by averaging. *SIAM Journal on Control and Optimization*, 30(4), 838–855.
- Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *JMLR*.
- Ren, P., Chen, Z., Li, J., Ren, Z., Ma, J., & de Rijke, M. (2019). RepeatNet: A repeat aware neural recommendation machine for session-based recommendation. *AAAI*.
- Rendle, S., Krichene, W., Zhang, L., & Anderson, J. (2020). Neural collaborative filtering vs. matrix factorization revisited. *RecSys*.
- Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-attention with relative position representations. *NAACL*.
- Thaler, R. (1985). Mental accounting and consumer choice. *Marketing Science*, 4(3), 199–214.
- Tsai, Y.-H. H., Bai, S., Yamada, M., Morency, L.-P., & Salakhutdinov, R. (2019). Transformer dissection: A unified understanding of Transformer's attention via the lens of kernel. *EMNLP*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Wang, C., Zhang, M., Ma, W., Liu, Y., & Ma, S. (2019). Modeling item-specific temporal dynamics of repeat consumption for recommender systems. *WWW*.
- Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R., & Smola, A. (2017). Deep sets. *NeurIPS*.
- Zhang, X., et al. (2022). Price DOES matter! Modeling price and interest preferences in session-based recommendation. *SIGIR*.
- Zhou, G., et al. (2018). Deep interest network for click-through rate prediction. *KDD*.
