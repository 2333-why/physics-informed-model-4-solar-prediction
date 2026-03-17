# Physics-Informed Solar Prediction (Final Version B + pvlib)

该工程已实现：

- 图像序列 + 气象序列输入
- 云参数预测 `f, tau_cld, D_e, p_ice, p_occ`
- 气象分支预测 `tau_aer`
- `pvlib.simplified_solis` 计算晴空 `GHIclr/DNIclr/DHIclr`
- 直通估计训练代理：前向保持 pvlib 晴空值，反向将梯度传回 `AODHead + delta_pwv`
- 残差晴空头预测 FARMS 内部量 `Tdu_clr/Tuu_clr/Ruu_clr`
- FARMS-like overcast + 部分多云混合
- 物理监督 loss（clear-sky 锚定、AOD 慢变、残差与 baseline 正则）

## 目录

- `folsom_pretrain/data.py`：数据读取、对齐、预处理、晴空筛选
- `folsom_pretrain/solar.py`：太阳几何、Solis 晴空
- `folsom_pretrain/models.py`：Final 模型
- `folsom_pretrain/losses.py`：物理损失
- `train_image_pretrain.py`：三阶段训练（含 test split 评估）
- `test_image_pretrain.py`：独立测试脚本
- `configs/pretrain_image.yaml`：配置

## 数据预处理与筛选

已实现：

1. 全局过滤：
   - `mu0 > min_mu0_day`
   - 闭合误差 `< closure_thresh_global`（默认 0.05）
2. 晴空筛选 A：`pvlib.clearsky.detect_clearsky`
3. 晴空筛选 B：`beam_frac / diffuse_frac / closure / mu0` 规则
4. 稳定窗口筛选：
   - 1-min: `|ΔDNI|<120, |ΔDHI|<60, K=10`
   - 5-min: `|ΔDNI|<200, |ΔDHI|<100, K=6`
   - `auto` 模式按时间步长自动选择
5. 最终晴空：`A ∩ B ∩ stable`

## 分阶段训练

- Stage 1：仅训练 `AODHead + ResidualClearSkyHead + delta_pwv`，严格使用晴空子集 C
- Stage 2：端到端联合训练
- Stage 3：提高物理一致性损失权重精调

## NPU 约束

- 训练与测试入口均为 Ascend NPU only
- `folsom_pretrain/utils.py` 只保留 NPU 设备选择与 NPU AMP 路径
- 需在当前环境中单独安装与 PyTorch 匹配的 `torch_npu`

## 运行

训练（拆成两个脚本）：

```bash
python stage1_train.py --config configs/pretrain_image.yaml
python stage2_train.py --config configs/pretrain_image.yaml --ckpt outputs/final_pvlib_farms/stage1_best.pt
```

独立测试（读取 best checkpoint）：

```bash
python test_image_pretrain.py --config configs/pretrain_image.yaml --ckpt outputs/final_pvlib_farms/best.pt
```

输出：

- 模型：`outputs/final_pvlib_farms/last.pt`, `outputs/final_pvlib_farms/best.pt`
- 测试指标：`outputs/final_pvlib_farms/test_metrics.json`

## 注意

若 `dataset/folsom/Folsom_weather.csv` 是 HTML 重定向占位文件，代码会自动回退到：

`/mnt/nvme0/chronos-forecasting-main/dataset/Folsom_weather.csv`


可以。下面我把 **pvlib 计算晴空辐照度**（clear-sky GHI/DNI/DHI）**正式融合进你之前的“FARMS + AOD预测 + 残差晴空头”版本B**，给你一个最终、完整、可落地的方案（包含：输出参数、物理层、loss、训练方式、数据筛选/预处理）。

我会默认你用 **pvlib 的 Simplified Solis** 作为晴空辐照度 baseline，因为它显式使用 **AOD + PWV + pressure**，和你现有“预测 AOD + 经验 PWV + 气压”最匹配。pvlib 的 `simplified_solis` 输入就包括 `aod700`、`precipitable_water`、`pressure`、`dni_extra`，并输出 `ghi/dni/dhi`。([pvlib-python.readthedocs.io][1])

---





## 0) 最终闭环（Final Version B + pvlib clear-sky）

**图像 + 气象 → 网络预测云参数 + 气象预测 AOD → pvlib(Solis) 计算晴空 GHI/DNI/DHI → FARMS overcast（需要的晴空内部量由“残差晴空头”提供）→ 部分多云混合 → 与观测 GHI/DNI/DHI 做物理监督 loss。**

关键取舍：

* **晴空辐照度（GHIclr/DNIclr/DHIclr）交给 pvlib**（稳、成熟、可校准）。([pvlib-python.readthedocs.io][1])
* **FARMS overcast 仍然需要晴空内部量** ((T^{clr}*{du},T^{clr}*{uu},R^{clr}_{uu}))，pvlib不会直接给，所以保留“残差晴空头”只做这些内部量的校正。
* **直达透过率 (T^{clr}_{dd})** 用 pvlib 的 DNIclr 反推（或用你分解式，两者一致性做约束）。

---

# 1) 输入与预计算

## 1.1 你有的数据

* 图像：天空图像 (I_t)（单帧/序列）
* 气象：`air_temp, relhum, press, windsp, winddir, max_windsp, precipitation`
* 辐照度观测：`GHI, DNI, DHI`
* 经纬度 + 时间戳（必须有时区）

## 1.2 预计算（pvlib）

* 太阳位置：apparent elevation / zenith（用于 Solis 输入）
* 地外辐照度：`dni_extra`
* (\mu_0 = \cos(\text{zenith}))

> pvlib 的 `simplified_solis` 需要 `apparent_elevation`、`dni_extra`，并建议 `pressure` 用 Pa。([pvlib-python.readthedocs.io][2])

---

# 2) 网络输出参数清单（最终版）

## 2.1 云参数（image-dominant）

由 `CloudHead(I, met, geo)` 输出：

1. 云量 (f\in[0,1])
2. 云光学厚度 (\tau_{cld}\ge0)
3. 云有效粒径 (D_e\in[D_{min},D_{max}])（如 5–120 μm）
4. 相态概率 (p_{ice}\in[0,1])
5. 太阳遮挡概率 (p_{occ}\in[0,1])（强烈建议）

映射：

* (f=\sigma)
* (\tau_{cld}=\text{softplus})
* (D_e = D_{min}+(D_{max}-D_{min})\sigma)
* (p_{ice},p_{occ}=\sigma)

## 2.2 大气参数（met-only）

由 `AODHead(met)` 输出：
6) **AOD proxy** (\tau_{aer}\ge0)（映射到合理范围，Solis 建议 AOD 约 0–0.45）([pvlib-python.readthedocs.io][1])

## 2.3 经验/慢变项

7. (PWV = PWV_{emp}(T,RH,P)\cdot \exp(\delta_{pwv}))（(\delta) 可学标量/站点标量）
8. (O_3) 常数/慢变（若你不进 pvlib，可留在你分解式里；但 Solis 本身不需要 O3）

---

# 3) pvlib 计算晴空辐照度（clear-sky baseline）

## 3.1 选择模型：Simplified Solis（推荐）

用 pvlib：

* 输入：

  * `apparent_elevation`（deg）
  * `aod700`（用你的 (\tau_{aer}) 映射到 700nm AOD proxy）
  * `precipitable_water`（cm！注意单位，pvlib 文档写的是 cm）
  * `pressure`（Pa）
  * `dni_extra`
* 输出：`ghi`, `dni`, `dhi`（晴空三件套）([pvlib-python.readthedocs.io][1])

得到：
[
(GHI_{clr}^{pv}, DNI_{clr}^{pv}, DHI_{clr}^{pv})
]

## 3.2 用 pvlib DNIclr 反推 (T^{clr}_{dd})

因为 FARMS overcast 里需要 (T^{clr}*{dd})，你可以定义：
[
T^{clr}*{dd} = \mathrm{clip}\left(\frac{DNI_{clr}^{pv}}{F_0}, 0, 1\right)
]
这比你自己写一堆 (\tau_R+\tau_w+\tau_{aer}) 的分解式更省事，也更“可对齐”。

> 可选：你仍保留分解式作为**一致性正则**（见 loss），以便让 AOD/PWV 更物理。

---

# 4) 残差晴空头（只输出 FARMS 需要的“内部量”）

pvlib 给了晴空辐照度，但 FARMS overcast 仍需：
[
T^{clr}*{du},;T^{clr}*{uu},;R^{clr}_{uu}
]

## 4.1 AOD-based baseline（你要求的：baseline显式用AOD等大气量）

先构造简单可解释的经验 baseline（参数少、可学标量）：

* (\tau_{sca}=\tau_R(P)+c_{sca}\tau_{aer})
* (\tau_{abs}=\tau_w(PWV)+c_{abs}\tau_{aer})
* (g(\mu_0)=\frac{1}{\mu_0+\epsilon})

baseline：
[
T^{base}*{du}= \sigma!\Big(a_0+a_1\tau*{sca}g(\mu_0)-a_2\tau_{abs}g(\mu_0)\Big)\cdot (1-T^{clr}*{dd})
]
[
T^{base}*{uu}=\exp!\big(-b_0\tau_{clr}g(\mu_0)\big)
]
[
R^{base}*{uu}=1-\exp!\big(-c_0\tau*{sca}g(\mu_0)\big)
]

其中 (a_0,a_1,a_2,b_0,c_0,c_{sca},c_{abs}) 建议设为**少量可学标量**（并加很小正则），保证稳定。

## 4.2 残差头输出并组合

输入残差头的特征（全部可从 met+pvlib+几何得到）：
[
x_{phys}=[\mu_0,;P,;PWV,;\tau_{aer},;T^{clr}*{dd},;GHI*{clr}^{pv},DNI_{clr}^{pv},DHI_{clr}^{pv},;ws,wd,prcp]
]

输出残差：
[
(\Delta T_{du},\Delta T_{uu},\Delta R_{uu})=\mathrm{MLP}(x_{phys})
]

组合：
[
T^{clr}*{du}=\mathrm{clip}(T^{base}*{du}+\Delta T_{du},0,1)
]
[
T^{clr}*{uu}=\mathrm{clip}(T^{base}*{uu}+\Delta T_{uu},0,1)
]
[
R^{clr}*{uu}=\mathrm{clip}(R^{base}*{uu}+\Delta R_{uu},0,1)
]
并定义：
[
T^{clr}*{dt}=T^{clr}*{dd}+T^{clr}_{du}
]

**硬约束必须加：**

* (T^{clr}_{dt}\le 1)

---

# 5) FARMS overcast（云天）物理层 + 部分多云混合

## 5.1 云项（FARMS 原式）

* (T^{cld}*{dd}=\exp(-\tau*{cld}/\mu_0))
* (T^{cld}*{du}(\tau*{cld},D_e,\mu_0))：Eq(11)+水/冰Eq(12)(13)，按 (p_{ice}) 混合
* (R^{cld}*{uu}(\tau*{cld}))：Eq(14a)(14b)，按 (p_{ice}) 混合

## 5.2 overcast forward（按你之前方案）

* (F_d=\mu_0F_0T^{cld}*{dd}T^{clr}*{dd})
* (DNI_{cld}=F_0T^{cld}*{dd}T^{clr}*{dd})
* (F_1=\mu_0F_0T^{cld}*{dd}T^{clr}*{dt}+\mu_0F_0T^{cld}*{du}T^{clr}*{uu})
* (F_{total}=F_1,[1-R_s(R^{clr}*{uu}+R^{cld}*{uu}(T^{clr}_{uu})^2)]^{-1})
* (GHI_{cld}=F_{total})
* (DHI_{cld}=F_{total}-F_d)

(R_s)：常数（0.2）或站点可学慢变。

## 5.3 部分多云混合（最终输出）

pvlib 给的晴空辐照度直接用作混合底座：
[
\widehat{GHI} = f,GHI_{cld}+(1-f),GHI_{clr}^{pv}
]
[
\widehat{DNI} = f,DNI_{cld}+(1-f),DNI_{clr}^{pv}
]
[
\widehat{DHI} = \widehat{GHI}-\widehat{DNI}\mu_0
]

（你也可以把 (p_{occ}) 用来对 DNI 做额外 gating，但建议先不写死。）

---

# 6) 训练损失（PhyLoss 最终版，含 pvlib 融合）

设 (\rho) 为 Huber/Charbonnier。

## 6.1 主监督：辐照度重建

[
\mathcal L_{rad}=w_g\rho(\widehat{GHI}-GHI)+w_n\rho(\widehat{DNI}-DNI)+w_d\rho(\widehat{DHI}-DHI)
]
建议：(w_n=1.0,w_g=0.7,w_d=0.7)

## 6.2 FARMS 内部一致性

[
\mathcal L_{split}=\rho(DHI_{cld}-(GHI_{cld}-F_d))
]
[
\mathcal L_{id}=\rho(\widehat{GHI}-(\widehat{DNI}\mu_0+\widehat{DHI}))
]

## 6.3 残差晴空头物理约束（必须）

* 总透过率约束：
  [
  \mathcal L_{dt}=\mathbb E[\mathrm{ReLU}(T^{clr}*{dd}+T^{clr}*{du}-1)]
  ]
* 残差幅度正则（防万能补偿）：
  [
  \mathcal L_{\Delta}=\mathbb E[|\Delta T_{du}|+|\Delta T_{uu}|+|\Delta R_{uu}|]
  ]
* baseline 参数正则（少量可学标量）：
  [
  \mathcal L_{base}=|\theta_{base}|_2^2
  ]

## 6.4 pvlib-观测一致性用于“晴空锚定”（可选但强烈建议）

虽然 pvlib 是 baseline，但它用到你预测的 AOD/PWV，仍可能有 bias。你需要晴空子集把它钉住。

晴空子集 (\mathcal C) 的构造：

* 使用 pvlib 文档提供的 `detect_clearsky()`（Ren16）对 GHI 时间序列判晴空：它通过“测量序列 vs 期望晴空序列”的滑窗统计比较来判别。([pvlib-python.readthedocs.io][3])

  * 期望晴空序列：用 pvlib Solis/Ineichen 产生的 GHIclr
* 或用你之前的 beam/diffuse 比例规则（见第8节）

在 (\mathcal C) 上：

* 锚定晴空辐照度：
  [
  \mathcal L_{clr}=\rho(GHI_{clr}^{pv}-GHI)+\rho(DNI_{clr}^{pv}-DNI)
  ]
  并约束云量趋近 0：
  [
  \mathcal L_{f0}=\rho(f-0)
  ]

## 6.5 AOD 慢变（防抢解释权）

序列数据：
[
\mathcal L_{aod}=|\tau_{aer}(t)-\tau_{aer}(t-1)|_1
]

（可选）云参数弱平滑：
[
\mathcal L_{cloud_smooth}=|\tau_{cld}(t)-\tau_{cld}(t-1)|_1+|f(t)-f(t-1)|_1
]

## 6.6 总损失

[
\mathcal L=
\mathcal L_{rad}
+\lambda_{split}\mathcal L_{split}
+\lambda_{id}\mathcal L_{id}
+\lambda_{dt}\mathcal L_{dt}
+\lambda_{\Delta}\mathcal L_{\Delta}
+\lambda_{base}\mathcal L_{base}
+\lambda_{clr}\mathcal L_{clr}
+\lambda_{f0}\mathcal L_{f0}
+\lambda_{aod}\mathcal L_{aod}
+\lambda_{smooth}\mathcal L_{cloud_smooth}
]

推荐起始权重：

* (\lambda_{split}=0.2)
* (\lambda_{id}=0.1)
* (\lambda_{dt}=0.3)
* (\lambda_{\Delta}=1e{-3})
* (\lambda_{base}=1e{-4})
* (\lambda_{clr}=1.0)（仅晴空子集）
* (\lambda_{f0}=0.2)
* (\lambda_{aod}=0.5)（分钟级）
* (\lambda_{smooth}=0.05)

---

# 7) 训练流程（最终版）

## 阶段1：晴空底座训练（met-only）

训练：

* `AODHead`
* `PWV标量校准`（(\delta_{pwv})）
* `残差晴空头`（仅 (\Delta T_{du},\Delta T_{uu},\Delta R_{uu})）
* baseline 的少量标量参数 (\theta_{base})

数据：晴空子集 (\mathcal C)

loss：
[
\mathcal L_{clr}+\mathcal L_{dt}+\mathcal L_{\Delta}+\mathcal L_{base}+\mathcal L_{aod}
]

## 阶段2：端到端联合训练

解冻 `CloudHead` + `FARMS layer`，loss 用总 (\mathcal L)。

## 阶段3：精调

提高 (\lambda_{split},\lambda_{id}) 强化物理一致性；必要时提高 (\lambda_{\Delta}) 让残差更小、更物理。

---

# 8) 数据预处理与晴空筛选（最终版）

## 8.1 基本清洗

* 夜间剔除：(\mu_0\le 0)
* 低太阳高度不进晴空训练：(\mu_0<0.25)
* 物理闭合误差（强过滤）：
  [
  err=\left|\frac{GHI-(DNI\mu_0+DHI)}{GHI+\epsilon}\right|<0.05
  ]

## 8.2 晴空子集 (\mathcal C)（两种方法，推荐组合）

### 方法A：pvlib detect_clearsky（推荐做主筛）

* 用 pvlib 产生一条期望晴空 GHI 序列（Solis/Ineichen）
* 再用 `pvlib.clearsky.detect_clearsky(measured_ghi, expected_clearsky_ghi, ...)` 判晴空，它用滑动窗口统计比较并迭代缩放晴空序列。([pvlib-python.readthedocs.io][3])

### 方法B：你已有 DNI/DHI 的物理规则（做辅筛）

* `beam_frac = DNI*mu0/GHI > 0.6`（更严格 0.7）
* `diffuse_frac = DHI/GHI < 0.25`（更严格 0.2）
* 加时间稳定窗口：

  * 1-min：(|\Delta DNI|<120), (|\Delta DHI|<60)，连续 K=10
  * 5-min：(|\Delta DNI|<200), (|\Delta DHI|<100)，连续 K=6

最终：取 A 与 B 的交集（更干净），或 A 通过、B 作为质量分。

---

# 9) 你最终要实现的关键代码清单（按优先级）

1. pvlib：太阳位置 + `dni_extra`
2. AODHead：(\tau_{aer})
3. PWV 经验 + (\delta_{pwv}) 校准
4. pvlib `simplified_solis` 计算 (GHI_{clr}^{pv},DNI_{clr}^{pv},DHI_{clr}^{pv}) ([pvlib-python.readthedocs.io][1])
5. 由 (DNI_{clr}^{pv}) 反推 (T^{clr}_{dd})
6. AOD-based baseline + 残差晴空头输出内部量 (T^{clr}*{du},T^{clr}*{uu},R^{clr}_{uu})
7. FARMS Eq(11)(12)(13)(14) 写成可微张量算子
8. overcast forward + partial mixing
9. loss（含 pvlib 晴空锚定与 detect_clearsky 筛选）

---

## 你现在这套“pvlib + FARMS + 残差晴空头”的优势

* pvlib 给你一个成熟的 **晴空辐照度底座**（尤其是 Solis 显式用 AOD+PWV）([pvlib-python.readthedocs.io][4])
* FARMS 负责云天与多次散射结构约束（物理可解释）
* 残差晴空头只补 FARMS 需要但 pvlib不输出的内部量，且 baseline 明确依赖 AOD（满足你“更物理”的要求）

---

如果你下一步要“更落地”的东西：我可以直接给你一份**PyTorch 代码骨架**（含：pvlib 预计算缓存、Dataset、Model、PhysicsLayer、Loss、训练循环），并把 **Solis 的单位换算（PWV cm、pressure Pa）和 AOD 范围 clamp**都写好。

[1]: https://pvlib-python.readthedocs.io/en/v0.13.1/reference/generated/pvlib.clearsky.simplified_solis.html?utm_source=chatgpt.com "pvlib.clearsky.simplified_solis - pvlib python - Read the Docs"
[2]: https://pvlib-python.readthedocs.io/en/v0.4.2/generated/pvlib.clearsky.simplified_solis.html?utm_source=chatgpt.com "pvlib.clearsky.simplified_solis"
[3]: https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.clearsky.detect_clearsky.html?utm_source=chatgpt.com "pvlib.clearsky.detect_clearsky"
[4]: https://pvlib-python.readthedocs.io/en/v0.8.1/clearsky.html?utm_source=chatgpt.com "Clear sky — pvlib python 0.8.1+0.gba7d753.dirty documentation"
