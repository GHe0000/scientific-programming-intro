import sympy as sym

x1, x2, p1, p2 = sym.symbols('x1 x2 p1 p2')
Q1, Q2, P1, P2 = sym.symbols('Q1 Q2 P1 P2')
m1, m2, k, alpha, beta = sym.symbols('m1 m2 k alpha beta')

def V_func(x):
    return (sym.Rational(1, 2) * k * x**2 +
            sym.Rational(1, 3) * alpha * x**3 +
            sym.Rational(1, 4) * beta * x**4)

H = (p1**2 / (2*m1) + p2**2 / (2*m2) +
     V_func(x1) + V_func(-x2) + V_func(x2 - x1))

inv_trans = {
    x1: sym.Rational(1, 2) * (Q1 + Q2),
    x2: sym.Rational(1, 2) * (Q1 - Q2),
    p1: sym.Rational(1, 2) * (P1 + P2),
    p2: sym.Rational(1, 2) * (P1 - P2)
}

H_prime = H.subs(inv_trans)

x1_dot_H = sym.diff(H, p1)
x2_dot_H = sym.diff(H, p2)
p1_dot_H = -sym.diff(H, x1)
p2_dot_H = -sym.diff(H, x2)

# Q1_dot_LHS_xp = x1_dot_H + x2_dot_H
# Q2_dot_LHS_xp = x1_dot_H - x2_dot_H
# P1_dot_LHS_xp = sp.Rational(1, 2) * (p1_dot_H + p2_dot_H)
# P2_dot_LHS_xp = sp.Rational(1, 2) * (p1_dot_H - p2_dot_H)

Q1_dot_LHS_xp = x1_dot_H + x2_dot_H
Q2_dot_LHS_xp = x1_dot_H - x2_dot_H
P1_dot_LHS_xp = p1_dot_H + p2_dot_H
P2_dot_LHS_xp = p1_dot_H - p2_dot_H

Q1_dot_LHS = Q1_dot_LHS_xp.subs(inv_trans)
Q2_dot_LHS = Q2_dot_LHS_xp.subs(inv_trans)
P1_dot_LHS = P1_dot_LHS_xp.subs(inv_trans)
P2_dot_LHS = P2_dot_LHS_xp.subs(inv_trans)

Q1_dot_RHS = sym.diff(H_prime, P1)
Q2_dot_RHS = sym.diff(H_prime, P2)
P1_dot_RHS = -sym.diff(H_prime, Q1)
P2_dot_RHS = -sym.diff(H_prime, Q2)

check_Q1 = sym.simplify(Q1_dot_LHS - Q1_dot_RHS)
check_Q2 = sym.simplify(Q2_dot_LHS - Q2_dot_RHS)
check_P1 = sym.simplify(P1_dot_LHS - P1_dot_RHS)
check_P2 = sym.simplify(P2_dot_LHS - P2_dot_RHS)

print(sym.pretty(check_Q1))
print(sym.pretty(check_Q2))
print(sym.pretty(check_P1))
print(sym.pretty(check_P2))
