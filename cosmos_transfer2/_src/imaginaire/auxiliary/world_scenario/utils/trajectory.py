# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TypedDict

import numpy as np
import numpy.typing as npt
import torch
from scipy.interpolate import interp1d


class EgoMotionData(TypedDict):
    """Encompasses all required information to interpolate ego motion data."""

    tmin: int
    tmax: int
    tparam: npt.NDArray[np.float32]
    xyzs: npt.NDArray[np.float32]
    quats: npt.NDArray[np.float32] | torch.Tensor


class EgoPoseInterp:
    """Interpolates egopose data."""

    def __init__(
        self,
        tmin: int,
        tmax: int,
        tparam: npt.NDArray[np.float32],
        xyzs: npt.NDArray[np.float32],
        quats: npt.NDArray[np.float32] | torch.Tensor,
    ):
        """Initialize the interpolator.

        Args:
            tmin: int, the start time of the egopose data in microseconds
            tmax: int, the end time of the egopose data in microseconds
            tparam: list of floats, the relative (starting from 0)
                timestamps of the egopose data in seconds
            xyzs: list of lists of floats, the x,y,z position of the egopose
            quats: list of lists of floats, the quaternion orientation of
                the egopose
        """
        self.tmin = tmin
        self.tmax = tmax

        self.interp = interp1d(
            tparam,
            np.concatenate((xyzs, quats), 1),
            kind="linear",
            axis=0,
            copy=False,
            bounds_error=True,
            assume_sorted=True,
        )

    def convert_tstamp(self, tstamp: int | npt.NDArray[np.int64] | npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Converts the absolute timestamp (microsecond) to relative (s)."""
        result = 1e-6 * (tstamp - self.tmin)
        return np.asarray(result, dtype=np.float32)

    def __call__(
        self, t: npt.NDArray[np.float32] | npt.NDArray[np.int64], is_microsecond: bool = False
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Interpolate pose for t in seconds or microsecond."""
        EPS = 1e-5
        if is_microsecond:
            t = self.convert_tstamp(t)

        out = self.interp(t)
        xyzs = out[..., :3]
        quats = out[..., 3:]

        # normalize quats
        norm = np.linalg.norm(quats, axis=-1, keepdims=True)
        assert np.all(EPS < norm), norm
        quats = quats / norm

        return xyzs, quats


def adjust_orientation(
    vals: npt.NDArray[np.float32] | torch.Tensor,
) -> npt.NDArray[np.float32] | torch.Tensor:
    """Adjusts the orientation of the quaternions.

    Adjusts the orientation of the quaternions so that the dot product
    between vals[i] and vals[i+1] is non-negative.

    Args:
        vals (np.array or torch.tensor): (N, C)

    Returns:
        vals (np.array or torch.tensor): (N, C) adjusted quaternions
    """
    N, C = vals.shape
    if isinstance(vals, torch.Tensor):
        signs = torch.ones(N, dtype=vals.dtype, device=vals.device)
        signs[1:] = torch.where(0 <= (vals[:-1] * vals[1:]).sum(dim=1), 1.0, -1.0)
        signs = torch.cumprod(signs, dim=0)

        return vals * signs.reshape((N, 1))

    else:
        signs = np.ones(N, dtype=vals.dtype)
        signs[1:] = np.where(0 <= (vals[:-1] * vals[1:]).sum(axis=1), 1.0, -1.0)
        signs = np.cumprod(signs)

        return vals * signs.reshape((N, 1))


def preprocess_egopose(poses: dict) -> EgoMotionData:
    """Converts the poses to for interpolation.

    The dtype of all the inputs to the interpolator is float32.
    TODO: instead of a linear interpolator for quaternions,
    it'd be better to do slerp.

    Args:
        poses (dict): a dict containing the raw egopose data.

    Returns:
        A dictionary containing the following
            tmin: int, the start time of the egopose data in microseconds
            tmax: int, the end time of the egopose data in microseconds
            tparam: list of floats, the relative (starting from 0)
                timestamps of the egopose data in seconds
            xyzs: list of lists of floats, the x,y,z position of the egopose
            quats: list of lists of floats, the quaternion orientation of
                the egopose
    """
    # bounds of the interpolator as timestamps (ints)
    tmin = poses["timestamp"][0]
    tmax = poses["timestamp"][-1]

    # convert timestamps to float32 only after subtracting off tmin and
    # converting from microseconds to seconds
    tparam = (1e-6 * (poses["timestamp"] - tmin)).astype(np.float32)

    # prep x,y,z
    # convert to float64, subtract off mean, convert to float32
    xyzs = np.stack(
        (
            poses["x"].astype(np.float64),
            poses["y"].astype(np.float64),
            poses["z"].astype(np.float64),
        ),
        1,
    )
    xyzs = xyzs - xyzs.mean(axis=0, keepdims=True)
    xyzs = xyzs.astype(np.float32)

    # prep quaternions
    # parse directly as float32
    quats = np.stack(
        (
            poses["qw"].astype(np.float32),
            poses["qx"].astype(np.float32),
            poses["qy"].astype(np.float32),
            poses["qz"].astype(np.float32),
        ),
        1,
    )

    # prep quaternions for interpolation https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L847
    # make sure normalized
    quat_norm = np.linalg.norm(quats, axis=1)
    EPS = 1e-3
    if not np.all(np.abs(quat_norm - 1.0) < EPS):
        raise ValueError(f"Raw pose quaternions are too far from normalized; {quat_norm=}")
    # adjust signs so that sequential dot product is always positive
    quats = adjust_orientation(quats / quat_norm[:, None])

    return EgoMotionData(
        tmin=tmin,
        tmax=tmax,
        tparam=tparam,
        xyzs=xyzs,
        quats=quats,
    )


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros((*point.shape[:-1], 1))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternions: Quaternions as tensor of shape (..., 4), with real part first.
                    Must be unit quaternions.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # Extract quaternion components (w, x, y, z)
    w, x, y, z = torch.unbind(quaternions, dim=-1)

    # Compute rotation matrix elements
    # First row
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)

    # Second row
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)

    # Third row
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    # Stack into rotation matrix
    rotation_matrix = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )

    return rotation_matrix


def rotation_matrix_to_quaternion(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions using Shepperd's method for numerical stability.

    Args:
        rotation_matrices: Rotation matrices as tensor of shape (..., 3, 3).
                           Must be valid rotation matrices (orthogonal with determinant 1).

    Returns:
        Quaternions as tensor of shape (..., 4), with real part first.
        The quaternions are normalized and have non-negative real parts.
    """
    # Extract rotation matrix elements
    r00 = rotation_matrices[..., 0, 0]
    r01 = rotation_matrices[..., 0, 1]
    r02 = rotation_matrices[..., 0, 2]
    r10 = rotation_matrices[..., 1, 0]
    r11 = rotation_matrices[..., 1, 1]
    r12 = rotation_matrices[..., 1, 2]
    r20 = rotation_matrices[..., 2, 0]
    r21 = rotation_matrices[..., 2, 1]
    r22 = rotation_matrices[..., 2, 2]

    # Compute the trace
    trace = r00 + r11 + r22

    # Initialize quaternion components
    w = torch.zeros_like(trace)
    x = torch.zeros_like(trace)
    y = torch.zeros_like(trace)
    z = torch.zeros_like(trace)

    # Case 1: trace > 0 (most common case)
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
    w[mask1] = 0.25 * s1
    x[mask1] = (r21[mask1] - r12[mask1]) / s1
    y[mask1] = (r02[mask1] - r20[mask1]) / s1
    z[mask1] = (r10[mask1] - r01[mask1]) / s1

    # Case 2: r00 > r11 and r00 > r22
    mask2 = (~mask1) & (r00 > r11) & (r00 > r22)
    s2 = torch.sqrt(1.0 + r00[mask2] - r11[mask2] - r22[mask2]) * 2  # s = 4 * x
    w[mask2] = (r21[mask2] - r12[mask2]) / s2
    x[mask2] = 0.25 * s2
    y[mask2] = (r01[mask2] + r10[mask2]) / s2
    z[mask2] = (r02[mask2] + r20[mask2]) / s2

    # Case 3: r11 > r22
    mask3 = (~mask1) & (~mask2) & (r11 > r22)
    s3 = torch.sqrt(1.0 + r11[mask3] - r00[mask3] - r22[mask3]) * 2  # s = 4 * y
    w[mask3] = (r02[mask3] - r20[mask3]) / s3
    x[mask3] = (r01[mask3] + r10[mask3]) / s3
    y[mask3] = 0.25 * s3
    z[mask3] = (r12[mask3] + r21[mask3]) / s3

    # Case 4: r22 is largest
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + r22[mask4] - r00[mask4] - r11[mask4]) * 2  # s = 4 * z
    w[mask4] = (r10[mask4] - r01[mask4]) / s4
    x[mask4] = (r02[mask4] + r20[mask4]) / s4
    y[mask4] = (r12[mask4] + r21[mask4]) / s4
    z[mask4] = 0.25 * s4

    # Stack quaternion components (w, x, y, z)
    quaternions = torch.stack([w, x, y, z], dim=-1)

    # Normalize quaternions
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)

    # Ensure positive real part (standardize)
    quaternions = standardize_quaternion(quaternions)

    return quaternions
