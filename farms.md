、# 最终方案：pvlib + FARMS + 残差晴空头（Version B）

## 0. 最终闭环

图像 + 气象 → 网络预测云参数 + 气象预测 AOD → pvlib（Simplified Solis）计算晴空 `GHI/DNI/DHI` → FARMS overcast（所需晴空内部量由“残差晴空头”提供）→ 部分多云混合 → 与观测 `GHI/DNI/DHI` 做物理监督 loss。

### 关键取舍

* 晴空辐照度 `GHI_clr / DNI_clr / DHI_clr` 交给 **pvlib**。
* FARMS overcast 仍然需要晴空内部量：

  * `T_clr_du`
  * `T_clr_uu`
  * `R_clr_uu`
    pvlib 不直接提供，因此保留 **残差晴空头**，只做这些内部量的校正。
* 直达透过率 `T_clr_dd` 用 pvlib 的 `DNI_clr` 反推：

  `T_clr_dd = clip(DNI_clr_pv / F0, 0, 1)`

---

## 1. 输入与预计算

### 1.1 已有数据

* 图像：天空图像 `I_t`（单帧或序列）
* 气象：

  * `air_temp`
  * `relhum`
  * `press`
  * `windsp`
  * `winddir`
  * `max_windsp`
  * `precipitation`
* 辐照度观测：`GHI, DNI, DHI`
* 经纬度 + 时间戳（必须带时区）

### 1.2 预计算（pvlib）

* 太阳位置：`apparent_elevation / zenith`
* 地外辐照度：`dni_extra`
* `mu0 = cos(zenith)`

`pvlib.clearsky.simplified_solis` 需要：

* `apparent_elevation`
* `aod700`
* `precipitable_water`（单位 cm）
* `pressure`（单位 Pa）
* `dni_extra`

---

## 2. 网络输出参数清单

### 2.1 云参数（image-dominant）

由 `CloudHead(I, met, geo)` 输出：

1. 云量 `f ∈ [0,1]`
2. 云光学厚度 `tau_cld >= 0`
3. 云有效粒径 `De ∈ [Dmin, Dmax]`（如 5–120 μm）
4. 相态概率 `p_ice ∈ [0,1]`
5. 太阳遮挡概率 `p_occ ∈ [0,1]`（强烈建议）

### 输出映射

* `f = sigmoid(.)`
* `tau_cld = softplus(.)`
* `De = Dmin + (Dmax - Dmin) * sigmoid(.)`
* `p_ice, p_occ = sigmoid(.)`

### 2.2 大气参数（met-only）

由 `AODHead(met)` 输出：

6. `tau_aer >= 0`（AOD proxy）

### 2.3 经验/慢变项

7. `PWV = PWV_emp(T, RH, P) * exp(delta_pwv)`
8. `O3` 常数/慢变（Solis 本身不需要 O3，可只用于分解式或备用正则）

---

## 3. pvlib 计算晴空辐照度（clear-sky baseline）

### 3.1 选择模型：Simplified Solis

输入：

* `apparent_elevation`
* `aod700`（由 `tau_aer` 映射得到）
* `precipitable_water`（cm）
* `pressure`（Pa）
* `dni_extra`

输出：

* `GHI_clr_pv`
* `DNI_clr_pv`
* `DHI_clr_pv`

### 3.2 用 pvlib 的 `DNI_clr` 反推 `T_clr_dd`

`T_clr_dd = clip(DNI_clr_pv / F0, 0, 1)`

可选：保留你自己的分解式作为一致性正则。

---

## 4. 残差晴空头（只输出 FARMS 需要的内部量）

pvlib 已给出晴空辐照度，但 FARMS overcast 仍需：

* `T_clr_du`
* `T_clr_uu`
* `R_clr_uu`

### 4.1 AOD-based baseline

先构造经验 baseline：

* `tau_sca = tau_R(P) + c_sca * tau_aer`
* `tau_abs = tau_w(PWV) + c_abs * tau_aer`
* `g(mu0) = 1 / (mu0 + eps)`

baseline：

`T_du_base = sigmoid(a0 + a1 * tau_sca * g(mu0) - a2 * tau_abs * g(mu0)) * (1 - T_clr_dd)`

`T_uu_base = exp(-b0 * tau_clr * g(mu0))`

`R_uu_base = 1 - exp(-c0 * tau_sca * g(mu0))`

其中：

* `a0, a1, a2, b0, c0, c_sca, c_abs` 为少量可学标量

### 4.2 残差头输入与输出

输入特征：

`x_phys = [mu0, P, PWV, tau_aer, T_clr_dd, GHI_clr_pv, DNI_clr_pv, DHI_clr_pv, ws, wd, prcp]`

输出残差：

* `ΔT_du`
* `ΔT_uu`
* `ΔR_uu`

组合：

* `T_clr_du = clip(T_du_base + ΔT_du, 0, 1)`
* `T_clr_uu = clip(T_uu_base + ΔT_uu, 0, 1)`
* `R_clr_uu = clip(R_uu_base + ΔR_uu, 0, 1)`

并定义：

`T_clr_dt = T_clr_dd + T_clr_du`

### 必须加的硬约束

* `T_clr_dt <= 1`

---

## 5. FARMS overcast（云天）物理层 + 部分多云混合

### 5.1 云项（FARMS 原式）

* `T_cld_dd = exp(-tau_cld / mu0)`
* `T_cld_du(tau_cld, De, mu0)`：Eq(11) + 水/冰云 Eq(12)(13)，按 `p_ice` 混合
* `R_cld_uu(tau_cld)`：Eq(14a)(14b)，按 `p_ice` 混合

公式补充（按 FARMS 论文公式编号）：

直达透射：

$$
T_{dd}^{cld} = \exp(-	au_{cld} / \mu_0)
$$

散射向下透射（Eq. 11 的形式）：

$$
T_{du}^{cld}(	au_{cld}, D_e, \mu_0) = rac{(b_0 + b_1\mu_0 + b_2\mu_0^2)\,	au_{cld}}{1 + c_1	au_{cld} + c_2	au_{cld}^2}
$$

水云/冰云系数不同（Eq. 12/13），按相态混合：

$$
T_{du}^{cld} = (1 - p_{ice})\,T_{du,w}^{cld} + p_{ice}\,T_{du,i}^{cld}
$$

云反射（Eq. 14a/14b）：

$$
R_{uu}^{cld} = rac{lpha\,	au_{cld}}{1 + eta\,	au_{cld}}
$$

水云/冰云混合：

$$
R_{uu}^{cld} = (1 - p_{ice})\,R_{uu,w}^{cld} + p_{ice}\,R_{uu,i}^{cld}
$$

说明：$b_i,c_i,lpha,eta$ 为按水云/冰云分别拟合的系数。

### 5.2 overcast forward

* `F_d = mu0 * F0 * T_cld_dd * T_clr_dd`
* `DNI_cld = F0 * T_cld_dd * T_clr_dd`
* `F_1 = mu0 * F0 * T_cld_dd * T_clr_dt + mu0 * F0 * T_cld_du * T_clr_uu`
* `F_total = F_1 * [1 - R_s * (R_clr_uu + R_cld_uu * (T_clr_uu)^2)]^(-1)`
* `GHI_cld = F_total`
* `DHI_cld = F_total - F_d`

`R_s`：常数（如 0.2）或站点可学慢变参数。

### 5.3 部分多云混合（最终输出）

* `GHI_hat = f * GHI_cld + (1 - f) * GHI_clr_pv`
* `DNI_hat = f * DNI_cld + (1 - f) * DNI_clr_pv`
* `DHI_hat = GHI_hat - DNI_hat * mu0`

可选：对 `p_occ` 做 DNI gating，但建议先不写死。

---

## 6. 训练损失（PhyLoss）

设 `rho(.)` 为 Huber 或 Charbonnier。

### 6.1 主监督：辐照度重建

`L_rad = wg * rho(GHI_hat - GHI) + wn * rho(DNI_hat - DNI) + wd * rho(DHI_hat - DHI)`

建议：

* `wn = 1.0`
* `wg = 0.7`
* `wd = 0.7`

### 6.2 FARMS 内部一致性

`L_split = rho(DHI_cld - (GHI_cld - F_d))`

`L_id = rho(GHI_hat - (DNI_hat * mu0 + DHI_hat))`

### 6.3 残差晴空头物理约束

* 总透过率约束：

  `L_dt = E[ReLU(T_clr_dd + T_clr_du - 1)]`

* 残差幅度正则：

  `L_delta = E[|ΔT_du| + |ΔT_uu| + |ΔR_uu|]`

* baseline 参数正则：

  `L_base = ||theta_base||_2^2`

### 6.4 pvlib-观测一致性：晴空锚定

在晴空子集 `C` 上：

`L_clr = rho(GHI_clr_pv - GHI) + rho(DNI_clr_pv - DNI)`

并约束云量趋近 0：

`L_f0 = rho(f - 0)`

### 6.5 AOD 慢变

序列数据：

`L_aod = ||tau_aer(t) - tau_aer(t-1)||_1`

（可选）云参数弱平滑：

`L_cloud_smooth = ||tau_cld(t) - tau_cld(t-1)||_1 + ||f(t) - f(t-1)||_1`

### 6.6 总损失

`L = L_rad + lambda_split * L_split + lambda_id * L_id + lambda_dt * L_dt + lambda_delta * L_delta + lambda_base * L_base + lambda_clr * L_clr + lambda_f0 * L_f0 + lambda_aod * L_aod + lambda_smooth * L_cloud_smooth`

### 推荐起始权重

* `lambda_split = 0.2`
* `lambda_id = 0.1`
* `lambda_dt = 0.3`
* `lambda_delta = 1e-3`
* `lambda_base = 1e-4`
* `lambda_clr = 1.0`（仅晴空子集）
* `lambda_f0 = 0.2`
* `lambda_aod = 0.5`
* `lambda_smooth = 0.05`

---

## 7. 训练流程

### 阶段1：晴空底座训练（met-only）

训练：

* `AODHead`
* `delta_pwv`
* 残差晴空头（仅 `ΔT_du, ΔT_uu, ΔR_uu`）
* baseline 少量标量参数 `theta_base`

数据：晴空子集 `C`

loss：

`L_clr + L_dt + L_delta + L_base + L_aod`

### 阶段2：端到端联合训练

解冻 `CloudHead + FARMS layer`，loss 用总 `L`。

### 阶段3：精调

提高：

* `lambda_split`
* `lambda_id`

必要时提高 `lambda_delta`，让残差更小、更物理。

---

## 8. 数据预处理与晴空筛选

### 8.1 基本清洗

* 夜间剔除：`mu0 <= 0`
* 低太阳高度不进晴空训练：`mu0 < 0.25`
* 物理闭合误差过滤：

  `err = |GHI - (DNI * mu0 + DHI)| / (GHI + eps) < 0.05`

### 8.2 晴空子集 `C`

#### 方法 A：pvlib `detect_clearsky`（主筛）

* 用 pvlib 生成期望晴空 `GHI_clr`
* 用 `pvlib.clearsky.detect_clearsky(measured_ghi, expected_clearsky_ghi, ...)` 判晴空

#### 方法 B：DNI/DHI 规则（辅筛）

* `beam_frac = DNI * mu0 / GHI > 0.6`（更严格 0.7）
* `diffuse_frac = DHI / GHI < 0.25`（更严格 0.2）

时间稳定窗口：

* 1-min：`|ΔDNI| < 120`, `|ΔDHI| < 60`, 连续 `K = 10`
* 5-min：`|ΔDNI| < 200`, `|ΔDHI| < 100`, 连续 `K = 6`

最终：

* 推荐取 A 与 B 的交集
* 或 A 通过、B 作为质量分

---

## 9. 关键代码清单（按优先级）

1. pvlib：太阳位置 + `dni_extra`
2. `AODHead -> tau_aer`
3. `PWV_emp + delta_pwv` 校准
4. `pvlib.clearsky.simplified_solis -> GHI_clr_pv, DNI_clr_pv, DHI_clr_pv`
5. 由 `DNI_clr_pv` 反推 `T_clr_dd`
6. AOD-based baseline + 残差晴空头输出 `T_clr_du, T_clr_uu, R_clr_uu`
7. FARMS Eq(11)(12)(13)(14) 写成可微张量算子
8. overcast forward + partial mixing
9. loss（含 pvlib 晴空锚定与 `detect_clearsky` 筛选）

---

## 10. 方案优势

* pvlib 给出成熟稳定的晴空辐照度基线
* FARMS 负责云天与多次散射结构约束
* 残差晴空头只补 pvlib 不提供、但 FARMS 需要的内部晴空量
* AOD-based baseline 显式依赖大气参数，更物理、更可解释
* 三阶段训练提高稳定性与可辨识性
