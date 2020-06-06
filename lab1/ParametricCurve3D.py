#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# author: Valentyn Kofanov

import numpy as np
from matplotlib import pyplot as plt
from sympy.parsing.sympy_parser import parse_expr
from mpl_toolkits.mplot3d import Axes3D
from sympy import *


class ParametricCurve3D:
    """
    implementation of predetermined parametric curve equation in 3-dimensional space
    """

    def __init__(self, x: str, y: str, z: str, param: str, interval: tuple) -> None:
        """
        constructor vector-valued function
        r(t) = (x(t), y(t), z(t))
        :param x: str - function x(t)
        :param y: str - function y(t)
        :param z: str - function z(t)
        :param param: str - variable symbol - r(t) - "t"
        :param interval: interval (a, b), a, b: float
        """
        assert interval[0] < interval[1], "invalid interval"
        self._x = x
        self._y = y
        self._z = z
        self._param = symbols(param)
        self._interval = interval
        self._fun = parse_expr(x), parse_expr(y), parse_expr(z)

    def __repr__(self) -> str:
        """
        string representation
        :return: str vector-valued function
        """
        return f"r({self._param}) = ({self._x}, {self._y}, {self._z})"

    def __str__(self) -> str:
        """
        string representation
        :return: str vector-valued function
        """
        return f"r({self._param}) = ({self._x}, {self._y}, {self._z})"

    def __call__(self, t0: float) -> np.ndarray:
        """
        calculating the value of a function at a point
        :param t0: parameter value
        :return: needed value
        """
        assert self._interval[0] <= t0 <= self._interval[1], f"t0 must be in {self._interval}"
        x = float(self._fun[0].subs(self._param, t0))
        y = float(self._fun[1].subs(self._param, t0))
        z = float(self._fun[2].subs(self._param, t0))
        return np.array([x, y, z])

    def modulo(self, t0: float) -> float:
        """
        calculating the value of a function modulo at a point
        :param t0: parameter value
        :return: needed value
        """
        x = float(self._fun[0].subs(self._param, t0))
        y = float(self._fun[1].subs(self._param, t0))
        z = float(self._fun[2].subs(self._param, t0))
        return np.linalg.norm(np.array([x, y, z]))

    def derivative(self, order: int):
        """
        finding a derivative of any order
        :param order: order of derivative ( >= 1 )
        :return: new vector-valued function
        """
        assert order > 0, "invalid derivative order"
        dx = diff(self._fun[0], self._param, order)
        dy = diff(self._fun[1], self._param, order)
        dz = diff(self._fun[2], self._param, order)
        return ParametricCurve3D(dx.__str__(), dy.__str__(), dz.__str__(), self._param.__str__(), self._interval)

    def plot(self, num=1000, show=True, save=False, filename="", frenet=False, t0=0) -> None:
        """
        plotting vector-valued function and support vectors
        :param num: dot density
        :param show: True/False - shows if True
        :param save: True/False - saves to filename if True
        :param filename: path to file
        :param frenet: True/False shows Frenet trihedron if True at t0 point
        :param t0: parameter value for Frenet trihedron
        :return: None
        """
        if filename == "":
            filename = f"{self.__str__()}.svg"

        fx = lambdify(self._param, self._fun[0], 'numpy')
        fy = lambdify(self._param, self._fun[1], 'numpy')
        fz = lambdify(self._param, self._fun[2], 'numpy')

        dots = np.linspace(self._interval[0], self._interval[1], num=num)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.plot3D(fx(dots), fy(dots), fz(dots),
                  label=f"r({self._param}) = ({self._x}, {self._y}, {self._z})")

        if frenet:
            assert self._interval[0] <= t0 <= self._interval[1], f"t0 must be in {self._interval}"
            m = self(t0)
            ax.quiver(*m, *self.get_tangent_unit(t0), label="tangent unit vector", color="red")
            ax.quiver(*m, *self.get_normal_unit(t0), label="normal unit vector", color="green")
            ax.quiver(*m, *self.get_binormal_unit(t0), label="binormal unit vector", color="black")

        plt.legend()

        if show:
            plt.show()
        if save:
            fig.savefig(filename)

    def get_tangent_unit(self, t0: float) -> np.ndarray:
        """
        finding tangent unit vector
        τ(t) = r'(t) / |r'(t)|
        :param t0: parameter value
        :return: tangent unit vector
        """
        assert self._interval[0] <= t0 <= self._interval[1], f"t0 must be in {self._interval}"
        m = self.modulo(t0)
        dr = self.derivative(order=1)
        return dr(t0) / m

    def get_normal_unit(self, t0: float) -> np.ndarray:
        """
        finding normal unit vector
        ν(t) = [[r'(t), r''(t)], r'(t)] / |[[r'(t), r''(t)], r'(t)]|
        :param t0: parameter value
        :return: normal unit vector
        """
        assert self._interval[0] <= t0 <= self._interval[1], f"t0 must be in {self._interval}"
        dr1 = self.derivative(order=1)(t0)
        dr2 = self.derivative(order=2)(t0)
        c = np.cross(dr1, dr2)
        m = np.cross(c, dr1)
        return m / (np.linalg.norm(c) * np.linalg.norm(dr1))

    def get_binormal_unit(self, t0: float) -> np.ndarray:
        """
        finding binormal unit vector
        β(t) = [r'(t), r''(t)] / |[r'(t), r''(t)]|
        :param t0: parameter value
        :return: binormal unit vector
        """
        assert self._interval[0] <= t0 <= self._interval[1], f"t0 must be in {self._interval}"
        dr1 = self.derivative(order=1)(t0)
        dr2 = self.derivative(order=2)(t0)
        c = np.cross(dr1, dr2)
        return c / np.linalg.norm(c)

    def get_osculating_plane(self, t0: float) -> Expr:
        """
        finding osculating plane at point
        (R - r(t0), r'(t0), r''(t0)) = 0
        R = (X, Y, Z)
        :param t0: parameter value
        :return: plane equation
        """
        assert self._interval[0] <= t0 <= self._interval[1], f"t0 must be in {self._interval}"
        x, y, z = symbols("X Y Z")
        m = self(t0)
        dr1 = self.derivative(order=1)(t0)
        dr2 = self.derivative(order=2)(t0)
        mt = Matrix([
            [x - m[0], y - m[1], z - m[2]],
            [dr1[0], dr1[1], dr1[2]],
            [dr2[0], dr2[1], dr2[2]]
        ])
        return mt.det()

    def get_normal_plane(self, t0: float) -> Expr:
        """
        finding normal plane at point
        x'(t0)*(X - x(t0)) + y'(t0) * (Y - y(t0)) + z'(t0)*(Z - z(t0)) = 0
        :param t0: parameter value
        :return: plane equation
        """
        assert self._interval[0] <= t0 <= self._interval[1], f"t0 must be in {self._interval}"
        x, y, z = symbols("X Y Z")
        m = self(t0)
        dr1 = self.derivative(order=1)(t0)
        return dr1[0] * (x - m[0]) + dr1[1] * (y - m[1]) + dr1[2] * (z - m[2])

    def get_reference_plane(self, t0: float) -> Expr:
        """
        finding reference plane at point
        ν_x(t0) * (X - x(t0)) + ν_y(t0) * (Y - y(t0)) + ν_z(t0) * (Z - z(t0)) = 0
        :param t0: parameter value
        :return: plane equation
        """
        assert self._interval[0] <= t0 <= self._interval[1], f"t0 must be in {self._interval}"
        x, y, z = symbols("X Y Z")
        m = self(t0)
        n = self.get_normal_unit(t0)
        return n[0] * (x - m[0]) + n[1] * (y - m[1]) + n[2] * (z - m[2])

    def get_curvature(self, t0: float) -> float:
        """
        finding curvature at point
        k(t) = |[r'(t), r''(t)]| / |r'(t)|^3
        :param t0: parameter value
        :return: curvature
        """
        assert self._interval[0] <= t0 <= self._interval[1], f"t0 must be in {self._interval}"
        dr1 = self.derivative(order=1)(t0)
        dr2 = self.derivative(order=2)(t0)
        c = np.cross(dr1, dr2)
        return np.linalg.norm(c) / (np.linalg.norm(dr1)) ** 3

    def get_torsion(self, t0: float) -> float:
        """
        finding torsion at point
        "kappa" K(t) = (r'(t), r''(t), r'''(t)) / |[r'(t), r''(t)]|^2
        :param t0: parameter value
        :return: torsion
        """
        assert self._interval[0] <= t0 <= self._interval[1], f"t0 must be in {self._interval}"
        dr1 = self.derivative(order=1)(t0)
        dr2 = self.derivative(order=2)(t0)
        dr3 = self.derivative(order=3)(t0)
        c = np.cross(dr1, dr2)
        m = np.dot(c, dr3)
        return m / np.linalg.norm(c) ** 2

    def get_osculating_circle(self, t0: float) -> tuple:
        """
        finding osculating circle as the intersection of the osculating sphere and plane
        :param t0: parameter value
        :return: tuple of Expr: (sphere, plane)
        """
        assert self._interval[0] <= t0 <= self._interval[1], f"t0 must be in {self._interval}"
        n = self.get_normal_unit(t0)
        m = self(t0)
        r_s = 1 / self.get_curvature(t0)  # sphere radius
        r_c = m + r_s * n
        x, y, z = symbols("X Y Z")
        sphere = (x - r_c[0]) ** 2 + (y - r_c[1]) ** 2 + (z - r_c[2]) ** 2 - r_s ** 2
        osculating_plane = self.get_osculating_plane(t0)
        return sphere, osculating_plane


def test() -> None:
    """
    simple test for curve r(t) = (t, t**3, t**2 + 4) in the interval (0, 2), t0 = 1
    :return: None
    """
    r = ParametricCurve3D("t", "t**3", "t**2 + 4", "t", (0, 2))
    t = 1
    print(f"print test: {r}")
    print(f"value at point {t}: {r(t)}")
    print(f"tangent unit vector at point {t}: {r.get_tangent_unit(t)}")
    print(f"normal unit vector at point {t}: {r.get_normal_unit(t)}")
    print(f"binormal unit vector at point {t}: {r.get_binormal_unit(t)}")
    print(f"osculating plane at point {t}: {r.get_osculating_plane(t)} = 0")
    print(f"normal plane at point {t}: {r.get_normal_plane(t)} = 0")
    print(f"reference plane at point {t}: {r.get_reference_plane(t)} = 0")
    print(f"curvature value ar point {t}: {r.get_curvature(t)}")
    print(f"torsion value ar point {t}: {r.get_torsion(t)}")
    circle = r.get_osculating_circle(t)
    print(f"osculating circle ar point {t}:\nsphere: {circle[0]} = 0\nplane: {circle[1]} = 0")
    r.plot(show=True, save=True, frenet=True, t0=t, filename="test.png")


if __name__ == '__main__':
    test()
