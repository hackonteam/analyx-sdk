# `@analyx-sdk/math` — Tài liệu kiến thức toán học

> Tham chiếu toán cho engine số học của Analyx. Mỗi mục gồm: **định nghĩa → công thức → phương pháp số (cách tính cho đúng & ổn định) → độ chính xác mục tiêu → dùng ở đâu trong PLS-SEM.**
>
> Nguyên tắc tối thượng: hàm trong `math` mà sai một chữ số thì mọi chỉ số PLS-SEM bên trên đều sai và lệch SmartPLS. Vì vậy đây không phải "toán cho vui" — đây là spec số học.

---

## 0. Quy ước & nguyên tắc số học

**Độ chính xác mục tiêu (đối chiếu R/scipy/numpy):**

| Nhóm | Sai số tối đa |
|---|---|
| Matrix mul / transpose | 1e-12 |
| Linear solve / QR / inverse | 1e-10 |
| OLS (coef, R²) | 1e-8 |
| Correlation / covariance | 1e-10 |
| Phân phối (Normal, t CDF) | 1e-6 |
| Percentile / quantile mẫu | 1e-10 |
| PRNG / bootstrap indices | **trùng khít byte** (deterministic) |

**Ba nguyên tắc không vi phạm:**

1. **`n` vs `n−1`:** dùng **sample variance (chia `n−1`, hiệu chỉnh Bessel)** làm mặc định. ⚠️ *Quyết định cần chốt với golden fixture:* PLS chuẩn hóa indicator — phải xác nhận SmartPLS dùng `n` hay `n−1` khi standardize, rồi cấu hình `math` cho khớp. Sai quy ước này → mọi loadings lệch ở chữ số thứ 3–4.
2. **Giải hệ, đừng nghịch đảo.** Cần $A^{-1}b$ thì giải $Ax=b$, không tính $A^{-1}$ rồi nhân. Nghịch đảo tường minh kém ổn định và chậm hơn.
3. **Tổng & phương sai phải ổn định số.** Dùng **Welford** cho mean/variance (tránh công thức $\sum x^2 - (\sum x)^2/n$ gây catastrophic cancellation). Tổng lớn cân nhắc Kahan/pairwise summation.

**Biểu diễn dữ liệu hot path:** `Float64Array` phẳng + dims tường minh (không `number[][]`), vì bootstrap tái ước lượng hàng nghìn lần.

---

## 1. Đại số tuyến tính

### 1.1 Nhân & chuyển vị ma trận
Với $A \in \mathbb{R}^{m\times k}, B\in\mathbb{R}^{k\times n}$:

$$(AB)_{ij} = \sum_{l=1}^{k} A_{il} B_{lj}, \qquad (A^\top)_{ij} = A_{ji}$$

*Dùng ở đâu:* nền cho mọi phép hồi quy, ma trận tương quan, ước lượng PLS.

### 1.2 Phân rã QR (Householder)
Phân rã $A = QR$ với $Q$ trực giao ($Q^\top Q = I$), $R$ tam giác trên. Dùng **phép phản xạ Householder** (ổn định hơn Gram-Schmidt):

Với vector cột $x$, dựng $v = x + \operatorname{sign}(x_1)\lVert x\rVert e_1$, ma trận phản xạ $H = I - 2\frac{vv^\top}{v^\top v}$ đưa $x$ về bội của $e_1$. Áp lần lượt qua các cột để tam giác hóa.

*Dùng ở đâu:* giải bình phương tối thiểu (OLS) ổn định — xương sống của path coefficients và Mode B.

### 1.3 Giải hệ tuyến tính & nghịch đảo
- **Đối xứng xác định dương** (vd ma trận tương quan): **Cholesky** $A = LL^\top$ rồi giải tiến/lùi.
- **Tổng quát:** LU hoặc QR.
- Giải $Rx = c$ với $R$ tam giác trên: **back-substitution**
$$x_i = \frac{1}{R_{ii}}\Big(c_i - \sum_{j>i} R_{ij}x_j\Big)$$

*Dùng ở đâu:* OLS qua QR ($R\hat\beta = Q^\top y$), tính VIF, các bước Mode B.

---

## 2. Thống kê mô tả

### 2.1 Trung bình, phương sai (Welford), độ lệch chuẩn

$$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i, \qquad s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i-\bar{x})^2, \qquad s=\sqrt{s^2}$$

**Welford (online, ổn định):**
```
M1 = 0; M2 = 0; cnt = 0
for x:
  cnt += 1
  d   = x - M1
  M1 += d / cnt
  M2 += d * (x - M1)
mean = M1
var_sample = M2 / (cnt - 1)     # var_pop = M2 / cnt
```

### 2.2 Chuẩn hóa (z-score)
$$z_i = \frac{x_i - \bar{x}}{s}$$
Kết quả: $\bar{z}=0$, $s_z=1$.

*Dùng ở đâu:* PLS-PM chạy trên indicator đã chuẩn hóa; construct score cũng được chuẩn hóa mỗi vòng lặp. Đây là hàm bị gọi liên tục — phải đúng quy ước `n`/`n−1` ở mục 0.

---

## 3. Hiệp phương sai & tương quan

$$\operatorname{cov}(x,y) = \frac{1}{n-1}\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})$$

$$r_{xy} = \frac{\operatorname{cov}(x,y)}{s_x s_y} = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2 \sum (y_i-\bar{y})^2}}$$

- **Ma trận hiệp phương sai/tương quan:** áp công thức trên theo cặp cột; ma trận đối xứng, đường chéo $=1$ (với tương quan).
- Tính ổn định: trừ trung bình trước rồi nhân tích trong, đừng dùng dạng khai triển.

*Dùng ở đâu:* **loadings phản ánh** = $\operatorname{corr}(\text{indicator}, \text{score})$; inner approximation scheme `factor`/`centroid` dùng tương quan giữa các construct score; nền cho Fornell-Larcker, HTMT, cross-loadings (tính ở `pls-sem`).

---

## 4. Hồi quy OLS

### 4.1 Mô hình & nghiệm
$$y = X\beta + \varepsilon, \qquad \hat\beta = (X^\top X)^{-1} X^\top y$$

⚠️ **Không** tính trực tiếp $(X^\top X)^{-1}$. Dùng QR: $X=QR \Rightarrow R\hat\beta = Q^\top y$, giải bằng back-substitution. Ổn định hơn nhiều khi cột gần cộng tuyến.

### 4.2 Trên dữ liệu chuẩn hóa → không cần intercept
Trong PLS, score đã chuẩn hóa (mean 0, var 1). Hồi quy $y$ chuẩn hóa lên $X$ chuẩn hóa **không có intercept** → hệ số chính là **standardized regression coefficients** = path coefficients. (Để cờ `intercept: boolean` trong API, mặc định `false` cho PLS.)

### 4.3 R², R² hiệu chỉnh
$$R^2 = 1 - \frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

$$\bar{R}^2 = 1 - (1-R^2)\frac{n-1}{n-p-1}\quad(p=\text{số predictor})$$

### 4.4 VIF (đa cộng tuyến)
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
với $R_j^2$ từ hồi quy predictor $j$ lên các predictor còn lại.

*Dùng ở đâu:* path coefficients & R² (structural model); $f^2$ tính bằng cách chạy lại OLS có/không một predictor; VIF cho cả collinearity của measurement (Mode B) lẫn structural; outer weights Mode B.

---

## 5. Số ngẫu nhiên có seed (deterministic)

### 5.1 PRNG
Cần generator **tái lập được** (cùng seed → cùng chuỗi) chất lượng đủ tốt cho bootstrap. Đề xuất **xoshiro256\*\*** hoặc tối thiểu **mulberry32**. Ví dụ mulberry32 (32-bit, đơn giản, đủ cho resample):
```
state = seed >>> 0
next():
  state = (state + 0x6D2B79F5) >>> 0
  t = state
  t = Math.imul(t ^ (t >>> 15), t | 1)
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296   # uniform [0,1)
```
*(Nếu cần chất lượng thống kê cao hơn, chuyển xoshiro256\*\* — cùng giao diện `next()`.)*

### 5.2 Số nguyên đều trên `[0, n)` — tránh modulo bias
Đừng dùng `floor(u * n)` thô khi cần đồng đều tuyệt đối; dùng **rejection** hoặc **Lemire**:
```
randInt(n):              # Lemire
  x = next_uint32()
  m = x * n              # 53-bit an toàn nếu n < 2^21; n mẫu thường nhỏ
  return floor(m / 2^32)
```

### 5.3 Bootstrap indices
Mỗi mẫu bootstrap: rút **`n` chỉ số có hoàn lại** từ `{0,…,n−1}`.
$$\text{indices}[b] = \big(\,\text{randInt}(n)\ \text{lặp}\ n\ \text{lần}\,\big),\quad b=1\dots B$$

⚠️ **Tái lập độc lập số worker:** sinh **toàn bộ `B` bộ index một lần** từ một seed trên main thread, rồi chia partition cho workers. Đừng cho mỗi worker tự seed.

*Dùng ở đâu:* toàn bộ inference (SE, t, p, CI) của PLS-SEM; sau này k-fold cho PLSpredict.

---

## 6. Phân phối xác suất

### 6.1 Hàm phụ trợ: `lnΓ` (Lanczos) và Beta
$$\ln\Gamma(z)\ \text{qua xấp xỉ Lanczos};\qquad \ln B(a,b)=\ln\Gamma(a)+\ln\Gamma(b)-\ln\Gamma(a+b)$$
*(Lanczos với bộ hệ số g=7, n=9 cho sai số ~1e-15.)*

### 6.2 Phân phối chuẩn (Normal)
$$\Phi(x) = \tfrac{1}{2}\Big[1+\operatorname{erf}\!\big(\tfrac{x}{\sqrt2}\big)\Big]$$

**`erf` (Abramowitz–Stegun 7.1.26, sai số ~1.5e-7):** với $x\ge0$, $t=\dfrac{1}{1+px}$,
$$\operatorname{erf}(x)\approx 1-(a_1t+a_2t^2+a_3t^3+a_4t^4+a_5t^5)e^{-x^2}$$
$$p=0.3275911;\ a_1=0.254829592;\ a_2=-0.284496736;\ a_3=1.421413741;\ a_4=-1.453152027;\ a_5=1.061405429$$
$\operatorname{erf}(-x)=-\operatorname{erf}(x)$. *(Cần >1e-7 thì dùng rational approximation kiểu Cody.)*

**Quantile $\Phi^{-1}(p)$ (probit) — Acklam:** xấp xỉ hữu tỉ chia 3 vùng (đuôi thấp / giữa / đuôi cao), rồi **một bước Halley** để đạt ~1e-15:
$$x_{new}=x-\frac{\Phi(x)-p}{\phi(x)}\Big[1+\frac{(\Phi(x)-p)\,x}{2\phi(x)}\Big]^{-1}$$
với $\phi$ là PDF chuẩn. *(Lấy bộ hệ số Acklam chuẩn để khởi tạo $x$.)*

### 6.3 Phân phối Student's t
Dùng **hàm beta không hoàn chỉnh chuẩn hóa** $I_x(a,b)$.

**CDF** với $\nu$ bậc tự do, đặt $x=\dfrac{\nu}{\nu+t^2}$:
$$F(t)=1-\tfrac{1}{2}I_x\!\big(\tfrac{\nu}{2},\tfrac12\big)\ (t\ge0),\qquad F(-t)=1-F(t)$$

**p-value hai phía** cho thống kê $t$ — gọn hơn, không cần qua $F$:
$$p = P(|T|>|t|) = I_x\!\big(\tfrac{\nu}{2},\tfrac12\big),\qquad x=\frac{\nu}{\nu+t^2}$$

**Quantile** $t_{p,\nu}$: tìm nghiệm $F(t)=p$ bằng Newton (đạo hàm là PDF) hoặc bisection — chỉ cần khi xuất giá trị tới hạn.

**$I_x(a,b)$ — phân số liên tục (Lentz), theo Numerical Recipes:**
$$I_x(a,b)=\frac{x^a(1-x)^b}{a\,B(a,b)}\cdot \text{CF}(x;a,b)$$
Mẹo hội tụ: nếu $x<\dfrac{a+1}{a+b+2}$ tính trực tiếp; ngược lại dùng $I_x(a,b)=1-I_{1-x}(b,a)$. Đánh giá $\ln$ ở mục 6.1 để tránh tràn số.

*Dùng ở đâu:* p-value & significance cho mọi tham số bootstrap (paths, loadings, weights, HTMT). Đây là lý do `StudentT` phải đúng — t-value khớp SmartPLS phụ thuộc trực tiếp vào nó.

---

## 7. Quantile của mẫu (percentile bootstrap CI)

Cho mẫu **đã sắp tăng** $x_{(0)}\le\dots\le x_{(N-1)}$ và $p\in[0,100]$. **Nội suy tuyến tính (numpy mặc định, "type 7"):**
$$h=\frac{p}{100}(N-1),\quad \ell=\lfloor h\rfloor,\quad Q(p)=x_{(\ell)}+(h-\ell)\big(x_{(\ell+1)}-x_{(\ell)}\big)$$

⚠️ **Quyết định khớp SmartPLS:** SmartPLS báo percentile CI (và BCa). Phải xác nhận đúng method nội suy & quy ước phần trăm SmartPLS dùng cho CI 95% (2.5 / 97.5), rồi cấu hình cho trùng. Khác method → CI lệch ở mép.

*Dùng ở đâu:* khoảng tin cậy bootstrap (`ciLower`, `ciUpper`) cho mọi tham số PLS-SEM.

---

## 8. Checklist độ chính xác & lỗi dễ mắc

| Vấn đề | Hệ quả | Phòng tránh |
|---|---|---|
| Sai quy ước `n` vs `n−1` khi standardize | loadings/CR/AVE lệch chữ số 3–4, không khớp SmartPLS | Chốt quy ước theo golden fixture (mục 0) |
| Dùng $\sum x^2-(\sum x)^2/n$ tính variance | mất chính xác với số lớn/gần nhau | Welford (mục 2.1) |
| Nghịch đảo $X^\top X$ trực tiếp | bất ổn khi gần cộng tuyến | QR + back-substitution (mục 4.1) |
| `floor(u*n)` cho random int | lệch phân phối nhẹ ở đuôi | rejection/Lemire (mục 5.2) |
| Mỗi worker tự seed | bootstrap không tái lập theo số core | sinh index 1 lần rồi chia partition (mục 5.3) |
| `erf`/`I_x` độ chính xác thấp | p-value, t-value lệch | A&S cho ≥1e-7; Lentz CF + lnΓ cho beta |
| Sai method percentile | CI lệch mép | khớp method SmartPLS (mục 7) |

**Bộ test bắt buộc (gắn vào Gate G1):** mỗi hàm ở trên có một test đối chiếu R/scipy/numpy đúng ngưỡng ở bảng mục 0; PRNG có test **determinism** (cùng seed → cùng output byte). Math chưa qua G1 thì không build `pls-sem`.
