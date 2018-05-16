import numpy as np
import scipy as sp
import scipy.optimize


class LagrangianConstraint:
    """Represents an optimization restriction, to be used with the Method of Lagrange Multipliers.

    Restrictions follow the form: a <= g(x) <= b

    Attributes:
        g (): restriction function
        a (int): lower bound
        b (int): upper bound
        u (int): init value for the Lagrange multiplier
        c (float): init value for the penalty constant
        c_g (float): gap threshold for increasing the penalty constant
        c_r (float): step size for penalty constant increases
        gap (int): init value for the gap (distance between g(x) and the [a, b] interval)
        name (str): restriction label

    """

    def __init__(self, g, a=0, b=0, u=0.1, c=0.1, c_g=0.9, c_r=0.1, gap=0, name=None):
        self.g = g
        self.a = a
        self.b = b
        self.u = u
        self.c = c
        self.c_g = c_g
        self.c_r = c_r
        self.gap = gap
        self.name = name

    @staticmethod
    def phi(t):
        """Quadratic penalization function.

        Args:
            t (): TODO

        Returns:
            float: penalization value

        """
        return 0.5 * t ** 2

    @staticmethod
    def dphi(t):
        """Derivative of the quadratic penalization function.

        Args:
            t ():

        Returns:

        TODO:
            * Do we need this?
        """
        return t

    @staticmethod
    def dphi_1(t):
        """Inverse derivative of the quadratic penalization function.

        Args:
            t ():

        Returns:

        TODO:
            * Do we need this?
        """
        return t

    def threshold_b(self, gx):
        """Upper bound for g(x) in the restriction of the Method of Lagrange multipliers.

        Args:
            gx (): TODO

        Returns:
            TODO

        """
        return self.u + self.dphi(self.c * (gx - self.b))

    def threshold_a(self, gx):
        """Lower bound for g(x) in the restriction of the Method of Lagrange multipliers.

        Args:
            gx (): TODO

        Returns:
            TODO

        """
        return self.u + self.dphi(self.c * (gx - self.a))

    def p(self, x):
        """Function representation of the restriction in the Augmented Lagrangian.

        Args:
            x ():

        Returns:


        """
        gx = self.g(x)
        if self.threshold_b(gx) > 0:
            return (gx - self.b) * self.u + self.phi(self.c * (gx - self.b)) / self.c
        elif self.threshold_a(gx) < 0:
            return (gx - self.a) * self.u + self.phi(self.c * (gx - self.a)) / self.c
        else:
            u_1 = self.dphi_1(-self.u)
            return (self.u * u_1 + self.phi(u_1)) / self.c

    def update_u(self, gx):
        """Updates the Lagrange multiplier (u) according to the value of g(x).

        Args:
            gx ():

        """
        new_u = self.threshold_b(gx)
        if new_u > 0:
            self.u = new_u
        else:
            new_u = self.threshold_a(gx)
            if new_u < 0:
                self.u = new_u
            else:
                self.u = 0

    def update_c(self, gx):
        """Updates the penalty constant (c) according to the value of g(x).

        Args:
            gx ():

        """
        new_gap = gx - self.b
        if new_gap < 0:
            new_gap = self.a - gx
            if new_gap < 0:
                new_gap = 0
        if new_gap > self.c_g * self.gap:
            self.c += self.c_r
        self.gap = new_gap

    def update(self, x):
        """Updates the Lagrange multiplier and penalty constant (u and c), according to the value of x.

        Args:
            x ():

        """
        gx = self.g(x)
        self.update_u(gx)
        self.update_c(gx)


class LagrangianMultiplier:
    """Optimal control solver. Uses the Method of Lagrange Multipliers, with penalization.

    Attributes:
        f (): function to be minimized
        bounds (list): list of bound intervals of the form [min, max],
            one interval for each variable to optimize
        bounds_max (): TODO
        bounds_min (): TODO
        constraints (list): list of restrictions, represented by objects of type Constraint
        max_pgap (): TODO

    """

    def __init__(self, f, bounds, constraints):
        self.f = f
        self.bounds = bounds
        self.bounds_max = np.zeros(len(self.bounds))
        self.bounds_min = np.zeros(len(self.bounds))
        for i in range(len(self.bounds)):
            self.bounds_min[i], self.bounds_max[i] = self.bounds[i]
        self.constraints = constraints
        self.max_pgap = -np.Inf

    def bound_x(self, x):
        """Applies to x the defined bounds.

        Args:
            x ():

        Returns:


        """
        return np.max([np.min([self.bounds_max, x], axis=0), self.bounds_min], axis=0)

    def augmented_lagrangian(self, x):
        """Calculates the augmented Lagrangian of x

        Args:
            x ():

        Returns:


        """
        b_x = self.bound_x(x)
        aug_lagr = self.f(b_x)
        for c_j in self.constraints:
            aug_lagr += c_j.p(b_x)
        return aug_lagr

    def approx_jacobian(self, x_k):
        """TODO

        Args:
            x_k ():

        Returns:


        """
        x_zero = np.zeros_like(x_k)
        jac = np.zeros_like(x_k)
        for i in range(len(jac)):
            x_kh = x_zero.copy()
            x_kh[i] = max(np.abs(x_k[i]) * 0.001, 0.001)
            x_u = self.bound_x(x_k + x_kh)
            x_l = self.bound_x(x_k - x_kh)
            diff = x_u[i] - x_l[i]
            if diff == 0.0:
                jac[i] = 0.0
            else:
                jac[i] = (self.augmented_lagrangian(x_u) - self.augmented_lagrangian(x_l)) / diff
        return np.nan_to_num(jac)

    def update_multipliers(self, x, debug=False):
        """Updates the Lagrange multiplier and penalty constant (u and c) for every restriction,
         according to the value of x.

        Args:
            x ():
            debug ():

        """
        self.max_pgap = -np.Inf
        for c_j in self.constraints:
            c_j.update(x)
            if c_j.g(x) != 0.0:
                self.max_pgap = np.max((self.max_pgap, np.nan_to_num(c_j.gap / c_j.g(x))))
            if debug:
                print("{0}: u={1:.4e}, c={2:.4e}, gx={3:.4e}, px={4:.4e}, gap={5:.4e}".
                      format(c_j.name, c_j.u, c_j.c, c_j.g(x), c_j.p(x), c_j.gap))
        if debug:
            print("max-pgap: {}\n".format(self.max_pgap))

    def minimize(self, x0, args=(), method='BFGS', tol=None, maxgap=1e-4, maxiter=1000, miniter=10, debug=False):
        """Execute the optimization process.

        Args:
            x0 (): init point
            args (): additional arguments for the augmented Lagrangian (not used currently)
            method (): method used for calculating the optimization direction, same options as in SciPy
            tol (): tolerance for the stop criteria to use with SciPy's minimize method
            maxgap (): tolerance for restrictions in the method of Lagrange multipliers
            maxiter (): maximum number of iterations for the method of Lagrange multipliers
            miniter (): minimum number of iterations for the method of Lagrange multipliers
            debug (): debug mode flag

        Returns:


        """
        xk = x0
        opt = None
        for i in range(maxiter):
            opt = sp.optimize.minimize(self.augmented_lagrangian, xk, args=args, method=method,
                                       jac=self.approx_jacobian,
                                       hess=None, hessp=None, bounds=None, constraints=(), tol=tol, callback=None,
                                       options={'maxiter': 1})
            xk = self.bound_x(opt.x)
            self.update_multipliers(xk, debug=debug)
            if i in range(miniter):
                continue
            elif self.max_pgap < maxgap:
                break
        if debug:
            if self.max_pgap < maxgap:
                print("Stop-Criteria: maxgap")
            else:
                print("Stop-Criteria: maxiter")
            print("Iterations: {}/{}".format(i + 1, maxiter))
        return opt
