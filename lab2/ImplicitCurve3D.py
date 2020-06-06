#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# author: Valentyn Kofanov

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import *


class ImplicitCurve3D:
    """
    Implementation of curve in specified by the intersection of two planes  3-dimensional space
    """

    def __init__(self, f: str, g: str, p: tuple, params="x y z", log=False) -> None:
        """
        constructor
        curve defined as the intersection of two surfaces
        :param f: F(x, y, z) = 0
        :param g: G(x, y, z) = 0
        :param p: point (x0, y0, z0)
        :param params: variables symbols
        :param log: logging if True
        """
        self.diffs = None
        self._p = p
        self._f = parse_expr(f)
        self._g = parse_expr(g)
        self._x, self._y, self._z = symbols(params)
        self._check = True
        self._log = log
        self._pre_count()

        if not self._check and self._log:
            print(f"can't apply theorem\n")

    def _pre_count(self) -> None:
        """
        auxiliary function
        calculation of all relevant components and saving class fields and logging
        :return: None
        """
        test = self._check_theorem_conditions()
        if not test:
            self._check = False
            return

        if self._log:
            print("3) -- checking jacobians --\n")

        cur_params = [self._x, self._y, self._z]
        jac = False
        for _ in range(3):
            self._x, self._y, self._z = cur_params
            jz = self._check_jacobian_z()
            if jz:
                jac = True
                break
            cur_params = cur_params[-1:] + cur_params[:2]
            self._p = self._p[-1:] + self._p[:2]

        if not jac:
            self._check = False
            return

        if self._log:
            print(f"taking {self._z} as a variable\n")
            print(f"Then there are:\n"
                  f" -- neighborhood ({self._z}0 - a, {self._z}0 + a)\n"
                  f" -- neighborhood V({self._x}0, {self._y}0)\n"
                  f" -- functions f, g:\n"
                  f"    f({self._z}), g({self._z}) є V, {self._z} є ({self._z}0 - a, {self._z}0 + a)\n"
                  f"    (a) f, g є C^oo\n"
                  f"    (b) f({self._z}0) = {self._x}0, g({self._z}0) = {self._y}0\n\n")

        r = self._get_r_diffs_z()

        if self._log:
            print(f"r({self._z}0) = {r[0]}\n"
                  f"-- finding derivatives --\n"
                  f"r'({self._z}0) = {r[1]}\n"
                  f"r''({self._z}0) = {r[2]}\n")
        self.diffs = r

    def __repr__(self) -> str:
        """
        string representation
        :return:
        """
        return f"F({self._x}, {self._y}, {self._z}) = {self._f}\n" \
               f"G({self._x}, {self._y}, {self._z}) = {self._g}\n"

    def __str__(self) -> str:
        """
        string representation
        :return:
        """
        return f"F({self._x}, {self._y}, {self._z}) = {self._f}\n" \
               f"G({self._x}, {self._y}, {self._z}) = {self._g}\n"

    def _check_theorem_conditions(self) -> bool:
        """
        checking conditions of the implicit function theorem
        :return: True if conditions are True / False otherwise
        """
        error = ""
        res_log = f"--Implicit function theorem--\n" \
                  f"F: {self._f.__str__()}\nG: {self._g.__str__()}\nP: {self._p}\n--checking conditions--\n" \
                  f"1) Suppose that F, G є C^oo\n2) "
        success = True
        success_f_p = self._f.subs({self._x: self._p[0], self._y: self._p[1], self._z: self._p[2]}) == 0
        if not success_f_p:
            success = False
            error += f"F({self._p}) != 0, "
        else:
            res_log += f"F({self._p}) = 0, "
        success_g_p = self._g.subs({self._x: self._p[0], self._y: self._p[1], self._z: self._p[2]}) == 0
        if not success_g_p:
            success = False
            error += f"G({self._p}) != 0\n"
        else:
            res_log += f"G({self._p}) = 0\n"
        if not success:
            print(error)
        if self._log and success:
            print(res_log)
        return success

    def _check_jacobian_z(self) -> bool:
        """
        finding jacobian
        | Fx(p), Fy(p) |
        | Gx(p), Gy(p) |
        :return: True if jacobian != 0 / False otherwise
        """
        p0 = {self._x: self._p[0], self._y: self._p[1], self._z: self._p[2]}
        m = Matrix([
            [self._f.diff(self._x).subs(p0), self._f.diff(self._y).subs(p0)],
            [self._g.diff(self._x).subs(p0), self._g.diff(self._y).subs(p0)]
        ])
        d = m.det()
        if self._log:
            print(f"checking variable: {self._z}\n")
            print(f"jacobian by {self._z}:\nmatrix: {m}\ndet = {m.det()}\n")
        return d != 0

    def _get_first_diff_z(self) -> tuple:
        """
        finding r' = (f'(z), g'(z), 1)
        :return: tuple (f'(z), g'(z))
        """
        assert self._check, "all jacobians are zero"
        df1, dg1 = symbols("df1 dg1")
        f = Function("f")(self._z)
        g = Function("g")(self._z)
        s = {self._x: f, self._y: g}
        df = self._f.subs(s).diff(self._z).subs({self._z: self._p[2]})
        dg = self._g.subs(s).diff(self._z).subs({self._z: self._p[2]})

        df_str = df.__str__()
        df_str = df_str.replace(" ", "").replace(f"f({self._p[2]})", f"({self._p[0]})").replace(f"g({self._p[2]})",
                                                                                                f"({self._p[1]})")
        df_str = df_str.replace(f"Subs(Derivative(f({self._z}),{self._z}),{self._z},{self._p[2]})", "df1")
        df_str = df_str.replace(f"Subs(Derivative(g({self._z}),{self._z}),{self._z},{self._p[2]})", "dg1")

        dg_str = dg.__str__()
        dg_str = dg_str.replace(" ", "").replace(f"f({self._p[2]})", f"({self._p[0]})").replace(f"g({self._p[2]})",
                                                                                                f"({self._p[1]})")
        dg_str = dg_str.replace(f"Subs(Derivative(f({self._z}),{self._z}),{self._z},{self._p[2]})", "df1")
        dg_str = dg_str.replace(f"Subs(Derivative(g({self._z}),{self._z}),{self._z},{self._p[2]})", "dg1")

        e1 = Eq(parse_expr(df_str), 0)
        e2 = Eq(parse_expr(dg_str), 0)
        res = solve([e1, e2])
        return float(res[df1]), float(res[dg1])

    def _get_second_diff_z(self) -> tuple:
        """
        finding r'' = (f''(z), g''(z), 0)
        :param p: point (x0, y0, z0)
        :param log: logging if True
        :return: tuple (f''(z), g''(z))
        """
        assert self._check, "all jacobians are zero"
        df2, dg2 = symbols("df2 dg2")
        f = Function("f")(self._z)
        g = Function("g")(self._z)
        s = {self._x: f, self._y: g}
        df = self._f.subs(s).diff(self._z, 2).subs({self._z: self._p[2]})
        dg = self._g.subs(s).diff(self._z, 2).subs({self._z: self._p[2]})


        df_str = df.__str__()
        df_str = df_str.replace(" ", "").replace(f"f({self._p[2]})", f"({self._p[0]})").replace(f"g({self._p[2]})",
                                                                                                f"({self._p[1]})")
        df_str = df_str.replace(f"Subs(Derivative(f({self._z}),{self._z}),{self._z},{self._p[2]})", "df1")
        df_str = df_str.replace(f"Subs(Derivative(g({self._z}),{self._z}),{self._z},{self._p[2]})", "dg1")
        df_str = df_str.replace(f"Subs(Derivative(f({self._z}),({self._z},2)),{self._z},{self._p[2]})", "df2")
        df_str = df_str.replace(f"Subs(Derivative(g({self._z}),({self._z},2)),{self._z},{self._p[2]})", "dg2")

        dg_str = dg.__str__()
        dg_str = dg_str.replace(" ", "").replace(f"f({self._p[2]})", f"({self._p[0]})").replace(f"g({self._p[2]})",
                                                                                                f"({self._p[1]})")
        dg_str = dg_str.replace(f"Subs(Derivative(f({self._z}),{self._z}),{self._z},{self._p[2]})", "df1")
        dg_str = dg_str.replace(f"Subs(Derivative(g({self._z}),{self._z}),{self._z},{self._p[2]})", "dg1")
        dg_str = dg_str.replace(f"Subs(Derivative(f({self._z}),({self._z},2)),{self._z},{self._p[2]})", "df2")
        dg_str = dg_str.replace(f"Subs(Derivative(g({self._z}),({self._z},2)),{self._z},{self._p[2]})", "dg2")

        df1, dg1 = self._get_first_diff_z()

        e1 = Eq(parse_expr(df_str).subs({"df1": df1, "dg1": dg1}), 0)
        e2 = Eq(parse_expr(dg_str).subs({"df1": df1, "dg1": dg1}), 0)

        res = solve([e1, e2])
        return float(res[df2]), float(res[dg2])

    def _get_r_diffs_z(self) -> np.ndarray:
        """
        finding matrix
        [f(z0), g(z0), z0]
        [f'(z0), g'(z0), 1]
        [f''(z0), g''(z0), 0]
        :return: needed matrix
        """
        assert self._check, "all jacobians are zero"
        df1, dg1 = self._get_first_diff_z()
        df2, dg2 = self._get_second_diff_z()
        r = np.array([self._p[0], self._p[1], self._p[2]])
        r1 = np.array([df1, dg1, 1])
        r2 = np.array([df2, dg2, 0])
        return np.array([r, r1, r2])

    def get_tangent_unit(self) -> np.ndarray:
        """
        finding tangent unit vector
        τ(t) = r'(t) / |r'(t)|
        :return: tangent unit vector
        """
        assert self._check, "all jacobians are zero"
        r1 = self.diffs[1]
        return r1 / np.linalg.norm(r1)

    def get_normal_unit(self) -> np.ndarray:
        """
        finding normal unit vector
        ν(t) = [[r'(t), r''(t)], r'(t)] / |[[r'(t), r''(t)], r'(t)]|
        :return: normal unit vector
        """
        assert self._check, "all jacobians are zero"
        r = self.diffs
        r1 = r[1]
        r2 = r[2]
        c = np.cross(r1, r2)
        m = np.cross(c, r1)
        return m / (np.linalg.norm(c) * np.linalg.norm(r1))

    def get_binormal_unit(self) -> np.ndarray:
        """
        finding binormal unit vector
        β(t) = [r'(t), r''(t)] / |[r'(t), r''(t)]|
        :return: binormal unit vector
        """
        assert self._check, "all jacobians are zero"
        r = self.diffs
        r1 = r[1]
        r2 = r[2]
        c = np.cross(r1, r2)
        return c / np.linalg.norm(c)

    def get_osculating_plane(self) -> Expr:
        """
        finding osculating plane at point
        (R - r(t0), r'(t0), r''(t0)) = 0
        R = (X, Y, Z)
        :return: plane equation
        """
        assert self._check, "all jacobians are zero"
        x, y, z = symbols("X Y Z")
        r = self.diffs
        m = r[0]
        r1 = r[1]
        r2 = r[2]
        mt = Matrix([
            [x - m[0], y - m[1], z - m[2]],
            [r1[0], r1[1], r1[2]],
            [r2[0], r2[1], r2[2]]
        ])
        return mt.det()

    def get_normal_plane(self) -> Expr:
        """
        finding normal plane at point
        x'(t0)*(X - x(t0)) + y'(t0) * (Y - y(t0)) + z'(t0)*(Z - z(t0)) = 0
        :return: plane equation
        """
        assert self._check, "all jacobians are zero"
        x, y, z = symbols("X Y Z")
        r = self.diffs
        m = r[0]
        r1 = r[1]
        return r1[0] * (x - m[0]) + r1[1] * (y - m[1]) + r1[2] * (z - m[2])

    def get_reference_plane(self) -> Expr:
        """
        finding reference plane at point
        ν_x(t0) * (X - x(t0)) + ν_y(t0) * (Y - y(t0)) + ν_z(t0) * (Z - z(t0)) = 0
        :return: plane equation
        """
        assert self._check, "all jacobians are zero"
        x, y, z = symbols("X Y Z")
        r = self.diffs
        m = r[0]
        n = self.get_normal_unit()
        return n[0] * (x - m[0]) + n[1] * (y - m[1]) + n[2] * (z - m[2])

    def get_curvature(self) -> float:
        """
        finding curvature at point
        k(t) = |[r'(t), r''(t)]| / |r'(t)|^3
        :return: curvature
        """
        assert self._check, "all jacobians are zero"
        r = self.diffs
        r1 = r[1]
        r2 = r[2]
        c = np.cross(r1, r2)
        return np.linalg.norm(c) / (np.linalg.norm(r1)) ** 3

    def get_torsion(self) -> float:
        """
        finding torsion at point
        "kappa" K(t) = (r'(t), r''(t), r'''(t)) / |[r'(t), r''(t)]|^2
        assuming torsion = 0 for our examples
        :return: torsion
        """
        assert self._check, "all jacobians are zero"
        return 0

    def get_osculating_circle(self) -> tuple:
        """
        finding osculating circle as the intersection of the osculating sphere and plane
        :return: tuple of Expr: (sphere, plane)
        """
        assert self._check, "all jacobians are zero"
        r = self.diffs
        n = self.get_normal_unit()
        m = r[0]
        r_s = 1 / self.get_curvature()  # sphere radius
        r_c = m + r_s * n
        x, y, z = symbols("X Y Z")
        sphere = (x - r_c[0]) ** 2 + (y - r_c[1]) ** 2 + (z - r_c[2]) ** 2 - r_s ** 2
        osculating_plane = self.get_osculating_plane()
        return sphere, osculating_plane


def test():
    # for tests
    print("====== 1 test ======")
    p1 = (5, -2, 17)
    F1 = "x**2 + y**2 - 2*z + 5"
    G1 = "x - 2*y - z + 8"
    r1 = ImplicitCurve3D(F1, G1, p1, log=True)
    print(f"tangent unit: {r1.get_tangent_unit()}\n")
    print(f"normal unit: {r1.get_normal_unit()}\n")
    print(f"binormal unit: {r1.get_binormal_unit()}\n")
    print(f"osculating plane: {r1.get_osculating_plane()}\n")
    print(f"normal plane: {r1.get_normal_plane()}\n")
    print(f"reference plane: {r1.get_reference_plane()}\n")
    print(f"curvature: {r1.get_curvature()}\n")
    print(f"torsion: {r1.get_torsion()}\n")
    circle = r1.get_osculating_circle()
    print(f"osculating circle:\nsphere: {circle[0]}\nosculating plane: {circle[1]}")
    print("\n\n\n")
    print("====== 2 test ======")
    p2 = (1, 1, 1)
    F2 = "x**2+y**2+z**2-3"
    G2 = "x**2+y**2-2"
    r2 = ImplicitCurve3D(F2, G2, p2, log=True)
    print(f"tangent unit: {r2.get_tangent_unit()}\n")
    print(f"normal unit: {r2.get_normal_unit()}\n")
    print(f"binormal unit: {r2.get_binormal_unit()}\n")
    print(f"osculating plane: {r2.get_osculating_plane()}\n")
    print(f"normal plane: {r2.get_normal_plane()}\n")
    print(f"reference plane: {r2.get_reference_plane()}\n")
    print(f"curvature: {r2.get_curvature()}\n")
    print(f"torsion: {r2.get_torsion()}\n")
    circle = r2.get_osculating_circle()
    print(f"osculating circle:\nsphere: {circle[0]}\nosculating plane: {circle[1]}")


if __name__ == '__main__':
    test()
