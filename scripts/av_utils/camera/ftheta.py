# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
File originally from https://github.com/nv-tlabs/Cosmos-Drive-Dreams/tree/main/cosmos-drive-dreams-toolkits
"""

import json
from typing import Any, TypeVar

import numpy as np
import torch
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit

from av_utils.camera.base import CameraBase

CropParams = TypeVar("CropParams")
ScaleParams = TypeVar("ScaleParams")


def compute_max_distance_to_border(image_size_component: float, principal_point_component: float) -> float:
    """Given an image size component (x or y) and corresponding principal point component (x or y),
    returns the maximum distance (in image domain units) from the principal point to either image boundary."""

    center = 0.5 * image_size_component
    if principal_point_component > center:
        return principal_point_component
    else:
        return image_size_component - principal_point_component


def compute_max_radius(
    image_size: np.ndarray | tuple[int, int], principal_point: np.ndarray | tuple[float, float]
) -> float:
    """Compute the maximum radius from the principal point to the image boundaries."""

    max_diag = np.array(
        [
            compute_max_distance_to_border(image_size[0], principal_point[0]),
            compute_max_distance_to_border(image_size[1], principal_point[1]),
        ]
    )
    return np.linalg.norm(max_diag).item()


def compute_ftheta_fw_mapping_max_angle(fwpoly: np.ndarray, max_radius_pixels: float) -> float:
    """
    Best guess of the valid domain of a forward mapping (ray to pixel).

    Args:
        fw_mapping (callable): Forward mapping function that maps angle to pixel radius.
        max_radius_pixels (float): Maximum radius in pixels.

    Returns:
        float: Maximum angle in radians.
    """

    # Constants
    MAX_ANGLE = np.pi  # Try up to 180 degrees (covers 360 degrees FOV if principal point is in the middle)
    DOMAIN_SAMPLE_COUNT = 2000  # Enough steps to get a fine enough resolution
    STEP_SIZE = MAX_ANGLE / DOMAIN_SAMPLE_COUNT

    angle = 0.0
    for i in range(1, DOMAIN_SAMPLE_COUNT + 1):
        next_angle = i * STEP_SIZE
        next_r = eval_polynomial(
            np.asarray(next_angle).reshape(1), fwpoly
        ).item()  # Forward polynomial maps angle to pixel radius

        # Compute the derivative at the next angle
        d_next_r = eval_polynomial_derivative(np.asarray(next_angle).reshape(1), fwpoly).item()

        if d_next_r <= 0.0:
            # Polynomial is monotonically increasing in valid domain, so derivative must be positive.
            # At this point, this is not the case -> stop
            break

        angle = next_angle
        if next_r > max_radius_pixels:
            # Now we're outside of the image pixels -> stop
            break

    return angle


def eval_polynomial(xs: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluates polynomial coeffs [D,] at given points [N,] using Horner scheme"""
    ret = np.zeros(len(xs), dtype=xs.dtype)

    for coeff in reversed(coeffs):
        ret = ret * xs + coeff

    return ret


def eval_polynomial_derivative(xs: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluates the derivative of polynomial coeffs [D,] at given points [N,] using Horner scheme"""
    ret = np.zeros(len(xs), dtype=xs.dtype)

    for i, coeff in enumerate(reversed(coeffs[:-1])):
        ret = ret * xs + coeff * (len(coeffs) - 1 - i)

    return ret


def approximate_polynomial_inverse(coeffs: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
    """
    Computes an approximate inverse polynomial in the provided range, fixing the first coefficient to zero.

    The fitting is performed via least squares on inverted sampled points (y, x) of the provided polynomial.

    Parameters:
    -----------
    coeffs : np.ndarray
        Coefficients of the polynomial to be inverted.
    range_min : float
        Minimum value of the range for sampling.
    range_max : float
        Maximum value of the range for sampling.

    Returns:
    --------
    np.ndarray
        Coefficients of the approximate inverse polynomial.

    Raises:
    -------
    ValueError
        If the polynomial degree is not supported (i.e., not 4 or 5).
    """

    def f4(x: float, b: float, x1: float, x2: float, x3: float, x4: float) -> float:
        """4th degree polynomial."""
        return b + x * (x1 + x * (x2 + x * (x3 + x * x4)))

    def f5(x: float, b: float, x1: float, x2: float, x3: float, x4: float, x5: float) -> float:
        """5th degree polynomial."""
        return b + x * (x1 + x * (x2 + x * (x3 + x * (x4 + x * x5))))

    SAMPLE_COUNT = 500
    samples_y = np.linspace(range_min, range_max, SAMPLE_COUNT)
    samples_x = eval_polynomial(samples_y, coeffs.astype(np.float64))  # use high-precision estimation

    # The constant term in the polynomial should be zero, so add the `bounds` condition
    if (poly_degree := len(coeffs) - 1) == 4:
        # Fit a 4th degree polynomial
        bounds = (
            [0, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf],
        )
        inv_coeffs, _ = curve_fit(f4, samples_x, samples_y, bounds=bounds)
    elif poly_degree == 5:
        # Fit a 5th degree polynomial
        bounds = (
            [0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf, np.inf],
        )
        inv_coeffs, _ = curve_fit(f5, samples_x, samples_y, bounds=bounds)
    else:
        raise ValueError(f"Unsupported polynomial degree {poly_degree}")

    return np.array([np.float32(val) if i > 0 else 0 for i, val in enumerate(inv_coeffs)], dtype=np.float32)


class FThetaCamera(CameraBase):
    """Defines an FTheta camera model."""

    @classmethod
    def from_rig(cls, rig_file: str, sensor_name: str) -> "FThetaCamera":
        """Helper method to initialize a new object using a rig file and the sensor's name.

        Args:
            rig_file (str): the rig file path.
            sensor_name (str): the name of the sensor. usually available sensors are:
                - camera:front:wide:120fov
                - camera:cross:left:120fov
                - camera:cross:right:120fov
                - camera:rear:left:70fov
                - camera:rear:right:70fov
                - camera:rear:tele:30fov
        Returns:
            FThetaCamera: the newly created object.
        """
        with open(rig_file) as fp:
            rig = json.load(fp)

        # Parse the properties from the rig file
        sensors = rig["rig"]["sensors"]
        sensor = None
        sensor_found = False

        for sensor in sensors:
            if sensor["name"] == sensor_name:
                sensor_found = True
                break

        if not sensor_found:
            raise ValueError(f"The camera '{sensor_name}' was not found in the rig!")

        assert sensor is not None
        return cls.from_dict(sensor)

    @classmethod
    def from_dict(cls, rig_dict: dict[str, Any]) -> "FThetaCamera":
        """Helper method to initialize a new object using a dictionary of the rig.

        Args:
            rig_dict (dict): the sensor dictionary to initialize with.

        Returns:
            FThetaCamera: the newly created object.
        """
        cx, cy, width, height, bw_poly, is_bw_poly, linear_cde = FThetaCamera.get_ftheta_parameters_from_json(rig_dict)
        return cls(cx, cy, width, height, bw_poly, is_bw_poly, linear_cde)

    @classmethod
    def from_numpy(cls, intrinsics: np.ndarray, device: str | None = None) -> "FThetaCamera":
        """Helper method to initialize a new object using an array of intrinsics.

        Args:
            intrinsics (np.ndarray): the intrinsics array. The ordering is expected to be
                "cx, cy, width, height, *poly (6 coefficients), is_bw_poly, (linear_c, linear_d, linear_e)".
                This is the same ordering as the `intrinsics` property of this class.

        Returns:
            FThetaCaamera: the newly created object.
        """
        return cls(
            cx=intrinsics[0],
            cy=intrinsics[1],
            width=intrinsics[2],
            height=intrinsics[3],
            poly=intrinsics[4:10],
            is_bw_poly=intrinsics[10] > 0,
            linear_cde=np.array([1, 0, 0], dtype=np.float32) if len(intrinsics) < 14 else intrinsics[11:14],
            device=device,
        )

    def __init__(
        self,
        cx: float,
        cy: float,
        width: int,
        height: int,
        poly: np.ndarray,
        is_bw_poly: bool = True,
        linear_cde: np.ndarray | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | None = None,
    ):
        """The __init__ method.

        Args:
            cx (float): optical center x.
            cy (float): optical center y.
            width (int): the width of the image.
            height (int): the height of the image.
            is_bw_poly (bool): whether the poly is bw poly
            poly (np.ndarray): the polynomial of the FTheta model. Usually 5 coefficients.
            linear_cde (np.ndarray): the linear constrain matrix of the FTheta model. Usually 3 coefficients.
                A = [[c, d], [e, 1]]. A is used in forward (ray2pixel) mapping
            device (str): the device to use. if None, use cuda if available, otherwise cpu.
        """
        if linear_cde is None:
            linear_cde = np.array([1, 0, 0], dtype=np.float32)
        assert linear_cde is not None
        assert len(poly) == 6, f"poly must have 6 coefficients, got {len(poly)}"
        assert len(linear_cde) == 3, f"linear_cde must have 3 coefficients, got {len(linear_cde)}"

        self._center = np.asarray([cx, cy], dtype=np.float32)
        self._width = int(width)
        self._height = int(height)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.reference_poly = "bw" if is_bw_poly else "fw"

        # new: sometimes we have non-identity linear constrain matrix
        # todo: this should be applied to forward / backward projection
        self.linear_cde = linear_cde
        c, d, e = linear_cde.tolist()

        self.A = np.array([[c, d], [e, 1]], dtype=np.float32)
        self.inv_A = np.array([[1, -d], [-e, c]], dtype=np.float32) / (c - e * d)

        if is_bw_poly:
            self._bw_poly = Polynomial(poly)
            self._fw_poly = self._compute_fw_poly()
        else:
            self._fw_poly = Polynomial(poly)
            self._bw_poly = self._compute_bw_poly()

        # Other properties that need to be computed
        self._horizontal_fov = None
        self._vertical_fov = None
        self._max_angle = None
        self._max_ray_angle = None

        # Populate the array of intrinsics. We need to store the reference poly instead of approximated poly
        self._intrinsics = np.array(
            [cx, cy, width, height, *poly, 1 if is_bw_poly else 0, *linear_cde], dtype=np.float32
        )

        self._update_calibrated_camera()
        self._cache_torch_tensors()

    def _cache_torch_tensors(self) -> None:
        # caches in torch
        self._max_ray_angle_torch = torch.tensor(self._max_ray_angle, dtype=self.dtype, device=self.device)
        self._fw_poly_torch = torch.tensor(self._fw_poly.coef, dtype=self.dtype, device=self.device)
        self._fw_poly_powers_torch = torch.arange(len(self._fw_poly_torch), dtype=self.dtype, device=self.device)
        self._max_ray_distortion_torch = torch.tensor(self._max_ray_distortion, dtype=self.dtype, device=self.device)
        self._center_torch = torch.tensor(self._center, dtype=self.dtype, device=self.device)
        self._A_torch = torch.tensor(self.A, dtype=self.dtype, device=self.device)
        self._inv_A_torch = torch.tensor(self.inv_A, dtype=self.dtype, device=self.device)

    def rescale(self, ratio: float) -> None:
        """
        if rescale to a smaller image, ratio < 1
        """
        self._width = int(self._width * ratio)
        self._height = int(self._height * ratio)
        self._center = self._center * ratio

        # update poly. if ratio = 0.5, bw_poly_coef[i] = bw_poly_coef[i] * (2 ** i)
        if self.reference_poly == "bw":
            bw_poly_coef = self._bw_poly.coef
            for i in range(len(bw_poly_coef)):
                bw_poly_coef[i] = bw_poly_coef[i] * (1 / ratio) ** i

            self._intrinsics = np.array(
                [self._center[0], self._center[1], self._width, self._height, *bw_poly_coef, 1, *self.linear_cde],
                dtype=np.float32,
            )
            self._bw_poly = Polynomial(bw_poly_coef)
            self._fw_poly = self._compute_fw_poly()

        elif self.reference_poly == "fw":
            fw_poly_coef = self._fw_poly.coef
            fw_poly_coef /= 1 / ratio

            self._intrinsics = np.array(
                [self._center[0], self._center[1], self._width, self._height, *fw_poly_coef, 0, *self.linear_cde],
                dtype=np.float32,
            )
            self._fw_poly = Polynomial(fw_poly_coef)
            self._bw_poly = self._compute_bw_poly()

        else:
            raise ValueError(f"Invalid reference poly: {self.reference_poly}")

        # update cached torch tensors
        self._update_calibrated_camera()
        self._cache_torch_tensors()

    @staticmethod
    def get_ftheta_parameters_from_json(rig_dict: dict[str, Any]) -> tuple[Any, ...]:
        """Helper method for obtaining FTheta camera model parameters from a rig dict.

        Args:
            rig_dict (Dict[str, Any]): the rig dictionary to parse.

        Raises:
            ValueError: if the provided rig is not supported.
            AssertionError: if the provided model is supported, but cannot be parsed properly.

        Returns:
            tuple[Any, ...]: the values `cx`, `cy`, `width`, `height` and `bw_poly` that were parsed.
        """
        props = rig_dict["properties"]

        if props["Model"] != "ftheta":
            raise ValueError("The given camera is not an FTheta camera")

        cx = float(props["cx"])
        cy = float(props["cy"])
        width = int(props["width"])
        height = int(props["height"])

        if "bw-poly" in props:  # Is this a regular rig?
            poly = props["bw-poly"]
        elif "polynomial" in props:  # Is this a VT rig?
            # VT rigs have a slightly different format, so need to handle these
            # specifically. Refer to the following thread for more details:
            # https://nvidia.slack.com/archives/C017LLEG763/p1633304770105300
            poly_type = props["polynomial-type"]
            if poly_type == "pixeldistance-to-angle":
                is_bw_poly = True
            elif poly_type == "angle-to-pixeldistance":
                is_bw_poly = False
            else:
                raise ValueError(f"Unsupported polynomial type: {poly_type} from rig dict {rig_dict}")

            linear_c = float(props["linear-c"]) if "linear-c" in props else 1
            linear_d = float(props["linear-d"]) if "linear-d" in props else 0
            linear_e = float(props["linear-e"]) if "linear-e" in props else 0

            poly = props["polynomial"]
            linear_cde = np.array([linear_c, linear_d, linear_e], dtype=np.float32)
        else:
            raise ValueError(f"Unable to parse the rig. Only FTheta rigs are supported! Rig:\n{rig_dict}")

        bw_poly = [np.float32(val) for val in poly.split()]
        return cx, cy, width, height, bw_poly, is_bw_poly, linear_cde

    @property
    def fov(self) -> tuple:
        """Returns a tuple of horizontal and vertical fov of the sensor."""
        if self._vertical_fov is None or self._horizontal_fov is None:
            self._compute_fov()
        return self._horizontal_fov, self._vertical_fov

    @property
    def width(self) -> int:
        """Returns the width of the sensor."""
        return self._width

    @property
    def height(self) -> int:
        """Returns the height of the sensor."""
        return self._height

    @property
    def center(self) -> np.ndarray:
        """Returns the center of the sensor."""
        return self._center

    @property
    def intrinsics(self) -> np.ndarray:
        """Obtain an array of the intrinsics of this camera model.

        Returns:
            np.ndarray: an array of intrinsics.
                The ordering is "cx, cy, width, height, poly, is_bw_poly, linear_c, linear_d, linear_e".
                dtype is np.float32.
        """
        return self._intrinsics

    def __str__(self) -> str:
        """Returns a string representation of this object."""
        assert self._horizontal_fov is not None, "horizontal_fov is not computed"
        assert self._vertical_fov is not None, "vertical_fov is not computed"
        horizontal_deg = np.degrees(self._horizontal_fov)
        vertical_deg = np.degrees(self._vertical_fov)
        return (
            f"FTheta camera model:\n\t{self._bw_poly}\n\t"
            f"center={self._center}\n\twidth={self._width}\n\theight={self._height}\n\t"
            f"h_fov={horizontal_deg}\n\tv_fov={vertical_deg}"
        )

    def _update_calibrated_camera(self) -> None:
        """Updates the internals of this object after calulating various properties."""
        self._compute_fov()
        self._max_ray_angle = self._max_angle.copy()
        is_fw_poly_slope_negative_in_domain = False
        ray_angle = self._max_ray_angle.copy()
        assert ray_angle is not None
        deg2rad = np.pi / 180.0

        while ray_angle >= np.float32(0.0):
            temp_dval = self._fw_poly.deriv()(self._max_ray_angle).item()
            if temp_dval < 0:
                is_fw_poly_slope_negative_in_domain = True
            ray_angle -= deg2rad * np.float32(1.0)

        if is_fw_poly_slope_negative_in_domain:
            ray_angle = self._max_ray_angle.copy()
            assert ray_angle is not None
            while ray_angle >= np.float32(0.0):
                ray_angle -= deg2rad * np.float32(1.0)
            raise ArithmeticError("FThetaCamera: derivative of distortion within image interior is negative")

        # Evaluate the forward polynomial at point (self._max_ray_angle, 0)
        # Also evaluate its derivative at the same point
        val = self._fw_poly(self._max_ray_angle).item()
        dval = self._fw_poly.deriv()(self._max_ray_angle).item()

        if dval < 0:
            raise ArithmeticError("FThetaCamera: derivative of distortion at edge of image is negative")

        self._max_ray_distortion = np.asarray([val, dval], dtype=np.float32)

    def _compute_fw_poly(self) -> Polynomial:
        """Computes the forward polynomial for this camera.

        This function is a replication of the logic in the following file from the DW repo:
        src/dw/calibration/cameramodel/CameraModels.cpp
        """
        max_value = compute_max_radius((self._width, self._height), self._center)
        coeffs = approximate_polynomial_inverse(self._bw_poly.coef, 0, max_value)

        # Return the polynomial and hardcode the bias value to 0
        return Polynomial([np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)])

    def _compute_bw_poly(self) -> Polynomial:
        """Computes the backward polynomial for this camera."""
        max_pixel_radius = compute_max_radius((self._width, self._height), self._center)
        max_value = compute_ftheta_fw_mapping_max_angle(self._fw_poly.coef, max_pixel_radius)
        coeffs = approximate_polynomial_inverse(self._fw_poly.coef, 0, max_value)
        return Polynomial([np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)])

    def _get_rays_impl(self) -> torch.Tensor:
        """
        Returns:
            rays: (H, W, 3), normalized camera rays in opencv convention

          z (front)
         /
        o ------> x (right)
        |
        v y (down)
        """
        u = torch.arange(self.width, dtype=torch.int32, device=self.device)
        v = torch.arange(self.height, dtype=torch.int32, device=self.device)
        u, v = torch.meshgrid(u, v, indexing="xy")  # must pass indexing='xy'
        uv_coords = torch.stack([u, v], dim=-1).reshape(-1, 2)  # shape (H, W, 2)

        # call pixel2ray to get the rays
        rays = self.pixel2ray(uv_coords.cpu().numpy())[0]
        rays = torch.tensor(rays, device=self.device, dtype=self.dtype).reshape(self.height, self.width, 3)
        return rays

    def pixel2ray_np(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Backproject 2D pixels into 3D rays.

        Args:
            x (np.ndarray): the pixels to backproject. Size of (n_points, 2), where the first
                column contains the `x` values, and the second column contains the `y` values.

        Returns:
            rays (np.ndarray): the backprojected 3D rays. Size of (n_points, 3).
            valid (np.ndarray): bool flag indicating the validity of each backprojected pixel.
        """
        # Make sure x is n x 2
        if np.ndim(x) == 1:
            x = x[np.newaxis, :]

        # Fix the type
        x = x.astype(np.float32)
        xd = np.einsum("ij,nj -> ni", self.inv_A, x - self._center)
        xd_norm = np.linalg.norm(xd, axis=1, keepdims=True)
        alpha = self._bw_poly(xd_norm)
        sin_alpha = np.sin(alpha)

        rx = sin_alpha * xd[:, 0:1] / xd_norm
        ry = sin_alpha * xd[:, 1:] / xd_norm
        rz = np.cos(alpha)

        rays = np.hstack((rx, ry, rz))
        # special case: ray is perpendicular to image plane normal
        valid = (xd_norm > np.finfo(np.float32).eps).squeeze()
        rays[~valid, :] = (0, 0, 1)  # This is what DW sets these rays to

        # note:
        # if constant coefficient of bwPoly is non-zero,
        # the resulting ray might not be normalized.
        return rays, valid

    def pixel2ray_torch(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixels (torch.Tensor): pixel coordinates. shape: (M, 2)

        Returns:
            rays (torch.Tensor): ray direction vector. shape: (M, 3)
            valid (torch.Tensor): bool flag indicating the validity of each backprojected pixel. shape: (M,)
        """
        # 确保输入形状为 (n_points, 2)
        if pixels.dim() == 1:
            pixels = pixels.unsqueeze(0)

        # Calculate the offset relative to the optical center
        xd = torch.einsum("ij,nj -> ni", self._inv_A_torch, pixels - self._center_torch)
        xd_norm = torch.norm(xd, dim=1, keepdim=False)

        # Calculate angle
        alpha = torch.zeros_like(xd_norm)
        for i, coef in enumerate(self._bw_poly.coef):
            alpha += coef * torch.pow(xd_norm, i)

        sin_alpha = torch.sin(alpha)

        # Calculate ray direction
        valid = (xd_norm > torch.finfo(self.dtype).eps).squeeze()
        rays = torch.empty(pixels.shape[0], 3, dtype=self.dtype, device=pixels.device)

        # For effective point calculation rays
        rays[valid, 0] = sin_alpha[valid] * xd[valid, 0] / xd_norm[valid]
        rays[valid, 1] = sin_alpha[valid] * xd[valid, 1] / xd_norm[valid]
        rays[valid, 2] = torch.cos(alpha[valid])

        # For the invalid point set to (0,0,1)
        rays[~valid, 0] = 0
        rays[~valid, 1] = 0
        rays[~valid, 2] = 1

        return rays, valid

    def ray2pixel_np(self, rays: np.ndarray) -> np.ndarray:
        """Project 3D rays to 2D pixel coordinates.

        Args:
            rays (np.ndarray): the rays. shape: (M, 3)

        Returns:
            result (np.ndarray): the projected pixel coordinates. shape: (M, 2)
        """
        # Make sure the input shape is (n_points, 3)
        if np.ndim(rays) == 1:
            rays = rays[np.newaxis, :]

        # Fix the type
        rays = rays.astype(np.float32)

        xy_norm = np.linalg.norm(rays[:, :2], axis=1, keepdims=True)
        cos_alpha = rays[:, 2:] / np.linalg.norm(rays, axis=1, keepdims=True)

        alpha = np.empty_like(cos_alpha)
        cos_alpha_condition = np.logical_and(cos_alpha > -1, cos_alpha < 1).squeeze()
        alpha[cos_alpha_condition] = np.arccos(cos_alpha[cos_alpha_condition])
        alpha[~cos_alpha_condition] = xy_norm[~cos_alpha_condition]

        delta = np.empty_like(cos_alpha)
        alpha_cond = alpha <= self._max_ray_angle
        delta[alpha_cond] = self._fw_poly(alpha[alpha_cond])

        # For outside the model (which need to do linear extrapolation)
        delta[~alpha_cond] = (
            self._max_ray_distortion[0] + (alpha[~alpha_cond] - self._max_ray_angle) * self._max_ray_distortion[1]
        )

        # Determine the bad points with a norm of zero, and avoid division by zero
        bad_norm = xy_norm <= 0
        xy_norm[bad_norm] = 1
        delta[bad_norm] = 0

        # compute pixel relative to center
        scale = delta / xy_norm
        pixel = scale * rays

        # Handle the edge cases (ray along image plane normal)
        edge_case_cond = (xy_norm <= 0).squeeze()
        pixel[edge_case_cond, :] = rays[edge_case_cond, :]
        result = np.einsum("ij,nj -> ni", self.A, pixel[:, :2]) + self._center

        return result

    def ray2pixel_torch(self, rays: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rays (torch.Tensor): ray direction vector. shape: (M, 3)

        Returns:
            result (torch.Tensor): projected pixel coordinates. shape: (M, 2)
        """
        # ensure input shape is (n_points, 3)
        if rays.dim() == 1:
            rays = rays.unsqueeze(0)

        # dtype and device
        rays = rays.to(dtype=self.dtype, device=self.device)

        xy_norm = torch.norm(rays[:, :2], dim=1, keepdim=True)
        cos_alpha = rays[:, 2:] / torch.norm(rays, dim=1, keepdim=True)

        alpha = torch.empty_like(cos_alpha)
        cos_alpha_condition = torch.logical_and(cos_alpha > -1, cos_alpha < 1).squeeze()
        alpha[cos_alpha_condition] = torch.acos(cos_alpha[cos_alpha_condition])
        alpha[~cos_alpha_condition] = xy_norm[~cos_alpha_condition]

        delta = torch.empty_like(cos_alpha)
        alpha_cond = alpha <= self._max_ray_angle_torch

        # Polynomial computation
        alpha_powers = alpha[alpha_cond].unsqueeze(-1) ** self._fw_poly_powers_torch
        delta[alpha_cond] = torch.sum(self._fw_poly_torch * alpha_powers, dim=-1, keepdim=False)

        # For outside the model (which need to do linear extrapolation)
        delta[~alpha_cond] = (
            self._max_ray_distortion_torch[0]
            + (alpha[~alpha_cond] - self._max_ray_angle_torch) * self._max_ray_distortion_torch[1]
        )

        # Determine the bad points with a norm of zero, and avoid division by zero
        bad_norm = xy_norm <= 0
        xy_norm[bad_norm] = 1
        delta[bad_norm] = 0

        # compute pixel relative to center
        scale = delta / xy_norm
        pixel = scale * rays

        # handle edge cases (rays along image plane normal)
        edge_case_cond = (xy_norm <= 0).squeeze()
        pixel[edge_case_cond, :] = rays[edge_case_cond, :]

        result = torch.einsum("ij,nj -> ni", self._A_torch, pixel[:, :2]) + self._center_torch

        return result

    def _get_pixel_fov(self, pt: np.ndarray) -> np.ndarray:
        """Gets the FOV for a given point. Used internally for FOV computation.

        Args:
            pt (np.ndarray): 2D pixel.

        Returns:
            fov (np.ndarray): the FOV of the pixel.
        """
        ray, _ = self.pixel2ray(pt)
        fov = np.arctan2(np.linalg.norm(ray[:, :2], axis=1), ray[:, 2])
        return fov

    def _compute_fov(self) -> None:
        """Computes the FOV of this camera model."""
        max_x = self._width - 1
        max_y = self._height - 1

        point_left = np.asarray([0, self._center[1]], dtype=np.float32)
        point_right = np.asarray([max_x, self._center[1]], dtype=np.float32)
        point_top = np.asarray([self._center[0], 0], dtype=np.float32)
        point_bottom = np.asarray([self._center[0], max_y], dtype=np.float32)

        fov_left = self._get_pixel_fov(point_left)
        fov_right = self._get_pixel_fov(point_right)
        fov_top = self._get_pixel_fov(point_top)
        fov_bottom = self._get_pixel_fov(point_bottom)

        self._vertical_fov = fov_top + fov_bottom
        self._horizontal_fov = fov_left + fov_right
        self._compute_max_angle()

    def _compute_max_angle(self) -> None:
        """Computes the maximum ray angle for this camera."""
        max_x = self._width - 1
        max_y = self._height - 1

        p = np.asarray([[0, 0], [max_x, 0], [0, max_y], [max_x, max_y]], dtype=np.float32)

        self._max_angle = np.maximum(
            np.maximum(self._get_pixel_fov(p[0, ...]), self._get_pixel_fov(p[1, ...])),
            np.maximum(self._get_pixel_fov(p[2, ...]), self._get_pixel_fov(p[3, ...])),
        )

    def is_ray_inside_fov(self, ray: np.ndarray) -> np.ndarray:
        """Determines whether a given ray is inside the FOV of this camera.

        Args:
            ray (np.ndarray): the 3D ray.

        Returns:
            np.ndarray: whether the ray is inside the FOV.
        """
        if np.ndim(ray) == 1:
            ray = ray[np.newaxis, :]

        ray_angle = np.arctan2(np.linalg.norm(ray[:, :2], axis=1), ray[:, 2])
        return ray_angle <= self._max_angle
