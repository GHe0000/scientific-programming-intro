import sympy as sym

sym.init_printing(use_unicode=True)

Q1, Q2, P1, P2, k, alpha, beta = sym.symbols('Q1 Q2 P1 P2 k alpha beta', real=True)

m = sym.symbols('m', real=True)

x1 = (Q1 + Q2)/2
x2 = (Q1 - Q2)/2

p1 = P1 + P2
p2 = P1 - P2

V_func = lambda x: sym.S(1)/2 * k * x**2 + sym.S(1)/3 * alpha * x**3 + sym.S(1)/4 * beta * x**4
V = V_func(x1) + V_func(-x2) + V_func(x2-x1)
T = sym.S(1)/2 * p1**2 / m + sym.S(1)/2 * p2**2 / m
H = T + V
