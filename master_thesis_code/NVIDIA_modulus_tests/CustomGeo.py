from sympy import Symbol, pi, sin, cos, sqrt

from modulus.sym.geometry.geometry import Geometry, csg_curve_naming
from modulus.sym.geometry.helper import _sympy_sdf_to_sdf
from modulus.sym.geometry.curve import SympyCurve
from modulus.sym.geometry.parameterization import Parameterization, Parameter, Bounds

class InfiniteCylinder(Geometry):
    """
3D Infinite Cylinder
Axis parallel to z-axis, no caps on ends

Parameters
----------
center : tuple with 3 ints or floats
center of cylinder
radius : int or float
radius of cylinder
height : int or float
height of cylinder
parameterization : Parameterization
Parameterization of geometry.
"""

    def __init__(self, center, radius, height, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        h, r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        theta = Symbol(csg_curve_naming(2))

        # surface of the cylinder
        curve_parameterization = Parameterization(
            {h: (-1, 1), r: (0, 1), theta: (0, 2 * pi)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + radius * cos(theta),
                "y": center[1] + radius * sin(theta),
                "z": center[2] + 0.5 * h * height,
                "normal_x": 1 * cos(theta),
                "normal_y": 1 * sin(theta),
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=height * 2 * pi * radius,
        )
        curves = [curve_1]

        # calculate SDF
        r_dist = sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        sdf = radius - r_dist

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - radius, center[0] + radius),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - height / 2, center[2] + height / 2),
            },
            parameterization=parameterization,
        )

        # initialize Infinite Cylinder
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )