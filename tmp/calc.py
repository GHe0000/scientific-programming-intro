import sympy as sp

# 符号
x, r = sp.symbols('x r')

# Logistic 映射
f = r*x*(1 - x)

# 迭代
f2 = sp.simplify(f.subs(x, f))
f3 = sp.simplify(f.subs(x, f2))

# 多项式与其导数条件
P = sp.expand(f3 - x)           # f^3(x) - x
Q = sp.expand(sp.diff(f3, x) - 1)  # (f^3)'(x) - 1

# 计算 resultant 消去 x
res = sp.resultant(P, Q, x)
res_factorlist = sp.factor_list(sp.simplify(res), r)  # 以 r 因式分解

# 打印因式分解结构（coeff, [(poly,exp), ...]）
print("Resultant 因式分解 (系数, [(多项式因子, 指数), ...])：")
res_factorlist

# 便于阅读，也单独挑出关键因子并打印
coeff, factors = res_factorlist
print("\n常数因子：", coeff)
print("\n主要因子及其指数：")
for fac, exp in factors:
    print("  -", sp.factor(fac), " ^", exp)

# 检查特定因子的多项式（例如 r^2 - 2 r - 7）
target = r**2 - 2*r - 7
found = False
for fac, exp in factors:
    if sp.simplify(fac - target) == 0:
        print("\n找到了目标因子 (r^2 - 2 r - 7) ，指数为：", exp)
        found = True
if not found:
    print("\n没有直接找到精确匹配的 (r^2 - 2 r - 7) 因子（但应在等价多项式形式中出现）。")

# 求出 r 的解析根（低阶因子）
r_candidates = sp.solve(target, r)
print("\nr^2 - 2 r - 7 = 0 的解析根：", r_candidates)

r_star = r_candidates[1]  # 1 + sqrt(8)
print("\n选择 r* = ", sp.N(r_star), "（即 1 + sqrt(8)）")

# 在 r = r_star 时求 f^3(x)-x 的数值根
P_rstar = sp.N(P.subs(r, r_star))
roots = sp.nroots(P_rstar)
print("\n在 r = 1+sqrt(8) 时，f^3(x)-x 的数值根（含一周期/二周期/三周期），共 %d 个（含重根）：" % len(roots))
for rt in roots:
    print("  ", rt)

# 去除一周期根（即满足 f(x)==x 的根）
f_at_r = sp.lambdify(x, sp.N(f.subs(r, r_star)), 'mpmath')
period3_roots = []
for rt in roots:
    val = complex(rt)
    fx = complex(f_at_r(val))
    if abs(fx - val) > 1e-7:
        period3_roots.append(rt)

print("\n排除一周期点后的候选（三周期或二重根等）：")
for rt in period3_roots:
    print("  ", rt)

# 去重
unique_roots = []
for rt in period3_roots:
    approx = complex(rt)
    if not any(abs(approx - complex(u)) < 1e-8 for u in unique_roots):
        unique_roots.append(rt)

print("\n去重后候选（三个真周期点应留下）：")
for rt in unique_roots:
    print("  ", rt, "，实部=", float(sp.re(rt)))

# 计算 (f^3)'(x) 在这些点的值（数值）以验证导数约为 1（切线分岔条件）
f3prime = sp.lambdify(x, sp.N(sp.diff(f3, x).subs(r, r_star)), 'mpmath')
print("\n在这些点处 (f^3)'(x) 的数值：")
for rt in unique_roots:
    val = complex(rt)
    print("  x =", val, " -> (f^3)'(x) ≈", f3prime(val.real + 0j))
