import sympy as sp
 
x = sp.Symbol('x', positive=True, real=True)
midx = sp.Symbol('m', integer=True, nonnegative=True)
 
def myfunction(k, x):
    # (k-1)! * (1 - e^{-x} * sum_{m=0}^{k-1} x^m/m!)
    return sp.factorial(k-1) * (1 - sp.exp(-x) * sp.summation(x**midx/sp.factorial(midx), (midx, 0, k-1)))
 
def computeCoefficients(n, m):
    p = min(n, m)
 
    # Matrix from (8)
    M = sp.Matrix([[myfunction(n - m + i + j - 1, x) for j in range(1, p+1)]
                   for i in range(1, p+1)])
 
    # K_{m,n}
    kmn = sp.Integer(1)
    for i in range(1, p+1):
        kmn *= 1/(sp.factorial(m - i) * sp.factorial(n - i))
    kmn = sp.simplify(kmn)
 
    det1 = sp.simplify(M.det())
    f = sp.simplify(kmn * sp.diff(det1, x))  # p_{λmax}(x)
 
    # Extract alpha[k,l] in representation: f(x)=kmn*sum alpha[k,l] x^l e^{-k x}
    # Trick: rewrite exp(-2x) as (exp(-x))^2, substitute y = exp(-x), then expand in y.
    y = sp.Symbol('y')
    f1 = sp.expand(f).subs(sp.exp(-2*x), sp.exp(-x)**2).subs(sp.exp(-x), y)
    poly_y = sp.Poly(f1, y)
 
    alph = {}
    a = {}
 
    for (k_pow,), coeff_y in poly_y.terms():
        # term is coeff_y * y^k_pow  -> coeff_y * exp(-k_pow*x)
        poly_x = sp.Poly(sp.expand(coeff_y), x)
        for (l_pow,), c in poly_x.terms():
            alpha = sp.simplify(c / kmn)
            if alpha != 0:
                k = int(k_pow)
                l = int(l_pow)
                alph[(k, l)] = alpha
                a[(k, l)] = sp.simplify(kmn * alpha * sp.factorial(l) / (sp.Integer(k)**(l+1)))
 
    return kmn, sp.factor(f), alph, a
 
# Example: 2x2
kmn, f, alph, a = computeCoefficients(2, 2)
print("Kmn =", kmn)
print("f(x) =", f)
print("alph =", alph)
print("a =", a)
