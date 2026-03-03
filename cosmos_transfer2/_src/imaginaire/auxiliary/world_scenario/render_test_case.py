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


import numpy as np
from scipy.spatial.transform import Rotation

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import (
    Crosswalk,
    DynamicObject,
    EgoPose,
    LaneBoundary,
    LaneLine,
    LaneLineColor,
    LaneLineStyle,
    ObjectType,
    Pole,
    RoadBoundary,
    RoadMarking,
    SceneData,
    TrafficSign,
    WaitLine,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.dataloaders.data_utils import (
    fix_static_objects,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.rendering.config import SETTINGS
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.scripts.local import (
    convert_scene_data_for_rendering,
    render_multi_camera_tiled,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.ftheta import FThetaCamera
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.laneline_utils import build_lane_line_type


def map_object_type(type_str: str) -> ObjectType:
    """Map object type to our ObjectType enum."""
    type_mapping = {
        "Automobile": ObjectType.CAR,
        "Other_vehicle": ObjectType.CAR,
        "Vehicle": ObjectType.CAR,
        "Car": ObjectType.CAR,
        "Pedestrian": ObjectType.PEDESTRIAN,
        "Person": ObjectType.PEDESTRIAN,
        "Bicycle": ObjectType.CYCLIST,
        "Cyclist": ObjectType.CYCLIST,
        "Motorcycle": ObjectType.CYCLIST,
        "Rider": ObjectType.CYCLIST,
        "Bus": ObjectType.TRUCK,
        "Truck": ObjectType.TRUCK,
        "Heavy_truck": ObjectType.TRUCK,
        "Train_or_tram_car": ObjectType.TRUCK,
        "Trolley_bus": ObjectType.TRUCK,
        "Trailer": ObjectType.TRUCK,
    }
    return type_mapping.get(type_str, ObjectType.OTHER)


def render_test_case():
    """Demonstrates how to render HD map control videos from scratch.

    Use this as a template for integrating the HD map rendering pipeline into your
    own codebase when aligning with parquet or RDS-HQ formats is difficult.

    For detailed data schemas and format specifications, see:
    `packages/cosmos-transfer2/docs/world_scenario_parquet.md`

    **Pipeline Steps:**

    REQUIRED (rendering fails without these):
      1. Scene Initialization - Create SceneData container
      2. Ego Poses (egomotion_estimate)
      3. Camera Calibration (calibration_estimate)
      4. Dynamic Objects (obstacle)

    OPTIONAL (enhance visualization):
      5. Lane Lines (lane_line)
      6. Lane Boundaries (lane)
      7. Road Boundaries (road_boundary)
      8. Crosswalks (crosswalk)
      9. Traffic Signs (traffic_sign)
      10. Poles (pole)
      11. Wait Lines (wait_line)
      12. Road Markings (road_marking)

    FINALIZE & RENDER:
      13. Finalize Scene Data
      14. Render to Video

    **Example Data:**
    Dataset: nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams
    Link: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams
    Data ID: 009799c8-eb9e-40e9-8269-2dd4cd8ae602_2948576399000_2948596399000
    This example uses the initial 2 frames from the data clip.
    """

    # ==========================================================================
    # STEP 1: Scene Initialization [REQUIRED]
    # ==========================================================================
    # The unique id for the data clip
    clip_id = "009799c8-eb9e-40e9-8269-2dd4cd8ae602_2948576399000_2948596399000"
    # The target fps for the data clip
    target_fps = 30
    # Initialize SceneData object
    scene_data = SceneData(
        scene_id=clip_id,
        frame_rate=target_fps,
        duration_seconds=0.0,
    )

    # ==========================================================================
    # STEP 2: Ego Poses [REQUIRED]
    # ==========================================================================
    # The time stamps are in microseconds. For this example we made up the time stamps
    # for the first 2 frames. In real case, you should load the time stamps from the data.
    time_stamps = [0, 33333]
    # Positions for ego vehicle in world coordinates: (x, y, z)
    ego_position = [np.array([0, 0, 0]), np.array([-3.6753314e-05, 3.3141394e-05, -4.7709480e-05])]
    # Rotation quaternion for ego vehicle in world coordinates: (x, y, z, w)
    ego_orientation = [
        np.array([8.712313e-19, 0.000000e00, -8.135135e-19, 1.000000e00]),
        np.array([-3.3501308e-06, 4.6475353e-07, 9.8401870e-08, 1.0000000e00]),
    ]
    # here we add Ego Pose for each time stamps into SceneData object
    for frame_idx in range(len(time_stamps)):
        ego_pose = EgoPose(
            timestamp=time_stamps[frame_idx],
            position=ego_position[frame_idx],
            orientation=ego_orientation[frame_idx],
        )
        scene_data.ego_poses.append(ego_pose)

    # ==========================================================================
    # STEP 3: Camera Calibration [REQUIRED]
    # ==========================================================================
    # Set camera type as Front-Left-Up (FLU)
    scene_data.metadata["coordinate_frame"] = "flu"
    # here we add camera calibrations for each camera, in real case you should list all your camera in camera_names list
    camera_names = ["camera_front_wide_120fov"]
    # intrinsic parameters for camera
    # format: # Format: [cx, cy, w, h, *poly (6 params), is_bw_poly, *linear_cde (3 params, optional)]
    intrinsic_params = [
        np.array(
            [
                9.64379400e02,
                7.57468700e02,
                1.92000000e03,
                1.08000000e03,
                0.00000000e00,
                1.05576540e-03,
                1.45430680e-08,
                -5.17833864e-11,
                1.00661632e-13,
                -3.69273984e-17,
                1.00000000e00,
            ]
        )
    ]
    # extrinsics matrics for camera to vehicle
    extrinsic_params = [
        np.array(
            [
                [-1.1054944e-02, -1.8103380e-02, 9.9977499e-01, 1.8492830e00],
                [-9.9993873e-01, 7.9657644e-04, -1.1042330e-02, -5.4168846e-02],
                [-5.9649372e-04, -9.9983579e-01, -1.8111076e-02, 1.3234164e00],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
            ]
        )
    ]
    # optional resize resolution for the camera, here we use 720p as the resize resolution
    resize_resolutions = [[720, 1280]]
    # process each camera
    for camera_idx in range(len(camera_names)):
        camera_name = camera_names[camera_idx]
        intrinsic_param = intrinsic_params[camera_idx]
        extrinsic_param = extrinsic_params[camera_idx]
        cx, cy, width, height = intrinsic_param[0], intrinsic_param[1], int(intrinsic_param[2]), int(intrinsic_param[3])
        poly_coeffs = intrinsic_param[4:10]
        # is_bw_poly = True if poly coefficients are for backward polynomial (pixel-distance-to-angle)
        # is_bw_poly = False if poly coefficients are for forward polynomial (angle-to-pixel-distance)
        is_bw_poly = bool(intrinsic_param[10])
        # Linear CDE parameters (default to [1, 0, 0] if not present)
        if len(intrinsic_param) > 11:
            linear_cde = intrinsic_param[11:14]
        else:
            linear_cde = np.array([1.0, 0.0, 0.0])
        resize_hw = resize_resolutions[camera_idx]
        # create camera model
        camera_model = FThetaCamera(
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            poly=poly_coeffs,
            is_bw_poly=is_bw_poly,
            linear_cde=linear_cde,
        )
        # resize the camera model if resize_hw is provided
        if resize_hw:
            resize_h, resize_w = resize_hw
            scale_h = resize_h / camera_model.height
            scale_w = resize_w / camera_model.width
            camera_model.rescale(ratio_h=scale_h, ratio_w=scale_w)
        # add camera model and extrinsics to scene data
        scene_data.camera_models[camera_name] = camera_model
        scene_data.camera_extrinsics[camera_name] = extrinsic_param.astype(np.float32)

    # ==========================================================================
    # STEP 4: Dynamic Objects [REQUIRED]
    # ==========================================================================
    # Load dynamic objects (e.g., car, pedestrian, cyclist, etc.)
    # all_object_info: A nested dictionary containing object annotations.
    # - Outer key: object_id (unique identifier for each object in the scene)
    # - Inner key: object properties
    # Each object dictionary contains the following necessary fields:
    # - poses: list of 4x4 transformation matrices (numpy array) representing object pose in world coordinates
    #   If world coordinates are not available, use object-to-ego transformation instead (keep the key name unchanged)
    # - lwh_values: list of Object dimensions [length, width, height] as a 3-element array
    #   Example: [[5.242823123931885, 2.438650608062744, 1.9042154550552368], [5.242823123931885, 2.438650608062744, 1.9042154550552368]]
    # - is_moving: Boolean flag indicating whether the object is moving
    #   Example: False
    # - type: String specifying the object category, you can map your naming style in to follow main types
    #   Scenedata main types:  'Car', 'Truck', 'Pedestrian', 'Cyclist', 'Others'
    # - time_stamps: list of time stamps in microseconds
    #   Example: [0, 33333]
    all_object_info = {
        "1045": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [4.316809177398682, 1.9877864122390747, 1.6775076389312744],
                [4.316809177398682, 1.9877864122390747, 1.6775076389312744],
            ],
            "poses": [
                np.array(
                    [
                        [9.99963463e-01, 8.54810281e-03, 5.76048497e-05, 1.35943878e01],
                        [-8.54810326e-03, 9.99963464e-01, 7.61687528e-06, 2.97443706e00],
                        [-5.75376352e-05, -8.10900918e-06, 9.99999998e-01, 8.96019188e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [9.99962847e-01, 8.61962627e-03, 8.05585076e-05, 1.35946863e01],
                        [-8.61962410e-03, 9.99962850e-01, -2.73011191e-05, 2.97164258e00],
                        [-8.07908404e-05, 2.66057210e-05, 9.99999996e-01, 8.96299289e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1407": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [3.7682204246520996, 1.891742467880249, 1.5337352752685547],
                [3.7682204246520996, 1.891742467880249, 1.5337352752685547],
            ],
            "poses": [
                np.array(
                    [
                        [9.98839187e-01, 4.81692095e-02, 7.85788606e-05, 1.49059921e01],
                        [-4.81692393e-02, 9.98839093e-01, 4.37331676e-04, 3.76871338e-01],
                        [-5.74217167e-05, -4.40609099e-04, 9.99999901e-01, 8.47473216e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [9.98824105e-01, 4.84808989e-02, 9.62645669e-05, 1.49058164e01],
                        [-4.84809343e-02, 9.98824025e-01, 4.07192518e-04, 3.73561485e-01],
                        [-7.64103030e-05, -4.11380698e-04, 9.99999912e-01, 8.47713665e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1564": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [5.060017108917236, 2.032839298248291, 1.7018378973007202],
                [5.060017108917236, 2.032839298248291, 1.7018378973007202],
            ],
            "poses": [
                np.array(
                    [
                        [-5.95154539e-02, -9.98227284e-01, 4.47241492e-04, 1.00466048e01],
                        [9.98227384e-01, -5.95154495e-02, 2.32231267e-05, -9.75287649e00],
                        [3.43581973e-06, 4.47830840e-04, 9.99999900e-01, 1.25459937e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-5.93940689e-02, -9.98234506e-01, 4.64964685e-04, 1.00451359e01],
                        [9.98234613e-01, -5.93940831e-02, -1.67980450e-05, -9.75444002e00],
                        [4.43845396e-05, 4.63146138e-04, 9.99999892e-01, 1.25426662e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1814": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [5.1844587326049805, 2.05204701423645, 1.8140590190887451],
                [5.1844587326049805, 2.05204701423645, 1.8140590190887451],
            ],
            "poses": [
                np.array(
                    [
                        [9.99999278e-01, -1.17953198e-03, -2.29937192e-04, 5.04795119e00],
                        [1.17946939e-03, 9.99999267e-01, -2.72116329e-04, 3.84658966e00],
                        [2.30257993e-04, 2.71844929e-04, 9.99999937e-01, 9.21155819e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [9.99999188e-01, -1.25666530e-03, -2.10224881e-04, 5.04727134e00],
                        [1.25660220e-03, 9.99999165e-01, -3.00001806e-04, 3.84552076e00],
                        [2.10601708e-04, 2.99737393e-04, 9.99999933e-01, 9.21567368e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1820": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [3.6017138957977295, 1.8042832612991333, 1.4598264694213867],
                [3.6017138957977295, 1.8042832612991333, 1.4598264694213867],
            ],
            "poses": [
                np.array(
                    [
                        [9.99068229e-01, 1.85043497e-03, -4.31190229e-02, 2.25458975e01],
                        [-4.25551546e-04, 9.99454233e-01, 3.30311295e-02, 2.67753829e-02],
                        [4.31566120e-02, -3.29820027e-02, 9.98523758e-01, 8.78843509e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [9.99068242e-01, 3.72306718e-03, -4.29975237e-02, 2.25442298e01],
                        [-2.31068439e-03, 9.99457586e-01, 3.28511090e-02, 2.28636344e-02],
                        [4.30965082e-02, -3.27211460e-02, 9.98534936e-01, 8.78895003e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1935": {
            "type": "Automobile",
            "is_moving": True,
            "lwh_values": [
                [4.760626316070557, 2.094951868057251, 1.5067020654678345],
                [4.760626316070557, 2.094951868057251, 1.5067020654678345],
            ],
            "poses": [
                np.array(
                    [
                        [-9.99875716e-01, 1.57592713e-02, -4.44278841e-04, 9.01954331e01],
                        [-1.57593917e-02, -9.99875777e-01, 2.68753181e-04, 9.84533840e00],
                        [-4.39988297e-04, 2.75721344e-04, 9.99999865e-01, 2.63134312e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-9.99873910e-01, 1.58736287e-02, -4.38734729e-04, 8.98614400e01],
                        [-1.58737349e-02, -9.99873976e-01, 2.39594815e-04, 9.83213222e00],
                        [-4.34876198e-04, 2.46528963e-04, 9.99999875e-01, 2.61928866e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1941": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [3.8677594661712646, 1.7036863565444946, 1.5169320106506348],
                [3.8677594661712646, 1.7036863565444946, 1.5169320106506348],
            ],
            "poses": [
                np.array(
                    [
                        [-9.99999141e-01, -1.30124533e-03, 1.54915074e-04, 9.63364240e01],
                        [1.30123816e-03, -9.99999152e-01, -4.63573098e-05, 1.34748330e01],
                        [1.54975265e-04, -4.61556886e-05, 9.99999987e-01, 2.73372150e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-9.99999347e-01, -1.12856747e-03, 1.80489953e-04, 9.63370109e01],
                        [1.12855673e-03, -9.99999361e-01, -5.95752814e-05, 1.34612010e01],
                        [1.80557072e-04, -5.93715497e-05, 9.99999982e-01, 2.73245317e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1943": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [4.191658973693848, 1.8524914979934692, 1.5427976846694946],
                [4.191658973693848, 1.8524914979934692, 1.5427976846694946],
            ],
            "poses": [
                np.array(
                    [
                        [-9.99961820e-01, -8.73711162e-03, -1.47432528e-04, 7.88728093e01],
                        [8.73709854e-03, -9.99961827e-01, 8.91246720e-05, 1.32648568e01],
                        [-1.48205592e-04, 8.78331367e-05, 9.99999985e-01, 2.14381362e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-9.99957739e-01, -9.19265014e-03, -1.22159555e-04, 7.88733654e01],
                        [9.19264308e-03, -9.99957745e-01, 5.82809922e-05, 1.32520727e01],
                        [-1.22690149e-04, 5.71555598e-05, 9.99999991e-01, 2.14309879e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1946": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [4.031059741973877, 1.7789041996002197, 1.5013806819915771],
                [4.031059741973877, 1.7789041996002197, 1.5013806819915771],
            ],
            "poses": [
                np.array(
                    [
                        [-9.98818086e-01, -4.86045233e-02, 1.79606340e-04, 7.10137822e01],
                        [4.86045272e-02, -9.98818101e-01, 1.71566117e-05, 1.33704210e01],
                        [1.78560175e-04, 2.58660152e-05, 9.99999984e-01, 1.69545309e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-9.98796334e-01, -4.90494532e-02, 1.86280874e-04, 7.10149187e01],
                        [4.90494506e-02, -9.98796351e-01, -1.86165138e-05, 1.33600265e01],
                        [1.86969787e-04, -9.45713147e-06, 9.99999982e-01, 1.69447806e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1953": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [4.642054080963135, 2.0360729694366455, 1.453634262084961],
                [4.642054080963135, 2.0360729694366455, 1.453634262084961],
            ],
            "poses": [
                np.array(
                    [
                        [-9.39336004e-01, 3.42993115e-01, -1.89613863e-03, 4.79386880e01],
                        [-3.42993017e-01, -9.39337874e-01, -3.86998444e-04, 1.32256774e01],
                        [-1.91385263e-03, 2.86840738e-04, 9.99998127e-01, 9.70664919e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-9.39115783e-01, 3.43595400e-01, -1.93588753e-03, 4.79408121e01],
                        [-3.43595217e-01, -9.39117743e-01, -4.37129367e-04, 1.32170463e01],
                        [-1.96822196e-03, 2.54646606e-04, 9.99998031e-01, 9.70736371e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1955": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [3.617568016052246, 1.686632752418518, 1.4736372232437134],
                [3.617568016052246, 1.686632752418518, 1.4736372232437134],
            ],
            "poses": [
                np.array(
                    [
                        [-5.96722154e-01, 8.02447826e-01, -3.96739031e-04, 5.47939484e01],
                        [-8.02447797e-01, -5.96722265e-01, -2.68801288e-04, 1.62487007e01],
                        [-4.52442022e-04, 1.57962677e-04, 9.99999885e-01, 1.16734957e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-5.97095068e-01, 8.02170390e-01, -3.80469118e-04, 5.47960646e01],
                        [-8.02170340e-01, -5.97095177e-01, -3.08394798e-04, 1.62406821e01],
                        [-4.74561450e-04, 1.21060029e-04, 9.99999880e-01, 1.16724044e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1959": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [3.59891414642334, 1.6402720212936401, 1.453256607055664],
                [3.59891414642334, 1.6402720212936401, 1.453256607055664],
            ],
            "poses": [
                np.array(
                    [
                        [-9.99998902e-01, -1.48179864e-03, 2.37230245e-05, 8.69532309e01],
                        [1.48180121e-03, -9.99998896e-01, 1.09014477e-04, 1.32729017e01],
                        [2.35614608e-05, 1.09049510e-04, 9.99999994e-01, 2.32956711e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-9.99998265e-01, -1.86215635e-03, 4.79804003e-05, 8.69540343e01],
                        [1.86216032e-03, -9.99998263e-01, 8.28885284e-05, 1.32519132e01],
                        [4.78259656e-05, 8.29777315e-05, 9.99999995e-01, 2.32775081e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1960": {
            "type": "Person",
            "is_moving": True,
            "lwh_values": [
                [0.7625482678413391, 0.788737952709198, 1.6646729707717896],
                [0.7625482678413391, 0.788737952709198, 1.6646729707717896],
            ],
            "poses": [
                np.array(
                    [
                        [-9.89017556e-01, -1.47796331e-01, -7.19399868e-04, 3.94631379e01],
                        [1.47796925e-01, -9.89017358e-01, -8.57352323e-04, -4.30441686e00],
                        [-5.84785428e-04, -9.54261588e-04, 9.99999374e-01, 1.51413119e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-9.89428879e-01, -1.45017318e-01, -6.86225047e-04, 3.94156311e01],
                        [1.45017885e-01, -9.89428657e-01, -8.63708243e-04, -4.30131360e00],
                        [-5.53718073e-04, -9.54092784e-04, 9.99999392e-01, 1.51230212e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1962": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [4.757304668426514, 2.025083303451538, 1.8426896333694458],
                [4.757304668426514, 2.025083303451538, 1.8426896333694458],
            ],
            "poses": [
                np.array(
                    [
                        [-8.64968898e-01, 5.01825316e-01, -3.97085879e-04, 4.71014631e01],
                        [-5.01825396e-01, -8.64968933e-01, 1.28947328e-04, 1.77021328e01],
                        [-2.78757916e-04, 3.10803207e-04, 9.99999913e-01, 1.14218683e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-8.65133818e-01, 5.01540958e-01, -3.80352177e-04, 4.71026448e01],
                        [-5.01541022e-01, -8.65133859e-01, 9.18043398e-05, 1.76947276e01],
                        [-2.83011911e-04, 2.70185259e-04, 9.99999923e-01, 1.14228033e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1965": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [4.357414722442627, 1.8420692682266235, 1.4439302682876587],
                [4.357414722442627, 1.8420692682266235, 1.4439302682876587],
            ],
            "poses": [
                np.array(
                    [
                        [-3.43164748e-01, 9.39275210e-01, -1.87109297e-04, 5.18471434e01],
                        [-9.39275185e-01, -3.43164664e-01, 3.74108015e-04, 2.36028219e01],
                        [2.87181085e-04, 3.04127803e-04, 9.99999913e-01, 7.05827237e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-3.43347202e-01, 9.39208534e-01, -1.67948093e-04, 5.18496513e01],
                        [-9.39208510e-01, -3.43347131e-01, 3.51219727e-04, 2.35945586e01],
                        [2.72204069e-04, 2.78328589e-04, 9.99999924e-01, 7.06187494e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "1969": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [4.382998466491699, 2.0050790309906006, 1.774078369140625],
                [4.382998466491699, 2.0050790309906006, 1.774078369140625],
            ],
            "poses": [
                np.array(
                    [
                        [-1.75043603e-01, 9.84560668e-01, 1.68164850e-04, 5.81620198e01],
                        [-9.84560675e-01, -1.75043578e-01, -1.52352576e-04, 2.87282342e01],
                        [-1.20564177e-04, -1.92236842e-04, 9.99999974e-01, 7.06135013e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-1.75417278e-01, 9.84494156e-01, 1.86332969e-04, 5.81646348e01],
                        [-9.84494164e-01, -1.75417246e-01, -1.77967830e-04, 2.87188746e01],
                        [-1.42522272e-04, -2.14662353e-04, 9.99999967e-01, 7.06503184e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "2034": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [3.825087547302246, 1.6715636253356934, 1.5301766395568848],
                [3.825087547302246, 1.6715636253356934, 1.5301766395568848],
            ],
            "poses": [
                np.array(
                    [
                        [-9.98058912e-01, 6.22753910e-02, -4.29420238e-04, 8.59100556e01],
                        [-6.22751222e-02, -9.98058832e-01, -6.13249095e-04, 1.72350553e01],
                        [-4.66776989e-04, -5.85316526e-04, 9.99999720e-01, 2.65368443e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-9.98067130e-01, 6.21436622e-02, -4.12056751e-04, 8.59119354e01],
                        [-6.21433899e-02, -9.98067024e-01, -6.43586414e-04, 1.72120616e01],
                        [-4.51255072e-04, -6.16735842e-04, 9.99999708e-01, 2.64920535e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "2069": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [4.104939937591553, 1.7771795988082886, 1.6461807489395142],
                [4.104939937591553, 1.7771795988082886, 1.6461807489395142],
            ],
            "poses": [
                np.array(
                    [
                        [-9.99979059e-01, -6.47139988e-03, -4.41408735e-05, 9.39226303e01],
                        [6.47140073e-03, -9.99979060e-01, -1.91220061e-05, 1.71133857e01],
                        [-4.40162031e-05, -1.94072590e-05, 9.99999999e-01, 3.00208833e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-9.99978197e-01, -6.60337167e-03, -2.67949225e-05, 9.39238230e01],
                        [6.60337300e-03, -9.99978196e-01, -4.95599754e-05, 1.71001074e01],
                        [-2.64670753e-05, -4.97358320e-05, 9.99999998e-01, 3.00154005e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "2112": {
            "type": "Automobile",
            "is_moving": True,
            "lwh_values": [
                [3.609227418899536, 1.6320360898971558, 1.4500794410705566],
                [3.609227418899536, 1.6320360898971558, 1.4500794410705566],
            ],
            "poses": [
                np.array(
                    [
                        [-9.97229237e-01, 7.42697761e-02, 4.22490903e-03, 7.17746079e01],
                        [-7.42589209e-02, -9.97235418e-01, 2.67087269e-03, 1.71756043e01],
                        [4.41159404e-03, 2.34973515e-03, 9.99987508e-01, 1.59662340e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [-9.97238952e-01, 7.41382001e-02, 4.24271318e-03, 7.17539251e01],
                        [-7.41274392e-02, -9.97245283e-01, 2.63994312e-03, 1.71636631e01],
                        [4.42674634e-03, 2.31815264e-03, 9.99987515e-01, 1.59896559e00],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "631": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [3.644240617752075, 2.024747133255005, 1.7225943803787231],
                [3.644240617752075, 2.024747133255005, 1.7225943803787231],
            ],
            "poses": [
                np.array(
                    [
                        [9.99920719e-01, 1.25917521e-02, -5.47825602e-05, 7.98705107e00],
                        [-1.25917357e-02, 9.99920679e-01, 2.90577113e-04, 9.35997819e-02],
                        [5.84370897e-05, -2.89864268e-04, 9.99999956e-01, 8.60751587e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [9.99919759e-01, 1.26678509e-02, -3.69723616e-05, 7.98635764e00],
                        [-1.26678406e-02, 9.99919724e-01, 2.66913554e-04, 9.15705854e-02],
                        [4.03506146e-05, -2.66423776e-04, 9.99999964e-01, 8.61115239e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
        "837": {
            "type": "Automobile",
            "is_moving": False,
            "lwh_values": [
                [4.567673683166504, 1.9848620891571045, 1.4297089576721191],
                [4.567673683166504, 1.9848620891571045, 1.4297089576721191],
            ],
            "poses": [
                np.array(
                    [
                        [9.98923743e-01, 4.63826058e-02, -9.73493076e-05, 2.06638683e01],
                        [-4.63825635e-02, 9.98923670e-01, 3.99256763e-04, 3.06518698e00],
                        [1.15763097e-04, -3.94311750e-04, 9.99999916e-01, 8.43326792e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                np.array(
                    [
                        [9.98922388e-01, 4.64118432e-02, -6.75077986e-05, 2.06638109e01],
                        [-4.64118158e-02, 9.98922325e-01, 3.62847696e-04, 3.06197109e00],
                        [8.42754775e-05, -3.59323527e-04, 9.99999932e-01, 8.40889679e-01],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "timestamps": [0, 33333],
        },
    }

    for track_id, obj_data in all_object_info.items():
        poses = np.array(obj_data["poses"])  # Shape: (N, 4, 4)
        timestamps = np.array(obj_data["timestamps"], dtype=np.int64)

        # Extract centers and orientations from pose matrices
        centers = poses[:, :3, 3]  # Translation part
        orientations = []
        for pose in poses:
            rotation_matrix = pose[:3, :3]
            quat = Rotation.from_matrix(rotation_matrix).as_quat()
            orientations.append(quat)
        orientations = np.array(orientations, dtype=np.float32)

        # Dimensions (per observation)
        dimensions = np.array(obj_data["lwh_values"], dtype=np.float32)
        if dimensions.ndim == 1:
            dimensions = np.repeat(dimensions.reshape(1, -1), poses.shape[0], axis=0)
        elif dimensions.shape[0] != poses.shape[0]:
            last = dimensions[-1]
            pad_count = poses.shape[0] - dimensions.shape[0]
            if pad_count > 0:
                pad = np.repeat(last.reshape(1, -1), pad_count, axis=0)
                dimensions = np.concatenate([dimensions, pad], axis=0)

        obj_type = map_object_type(str(obj_data["type"]))

        dynamic_obj = DynamicObject(
            track_id=track_id,
            object_type=obj_type,
            timestamps=timestamps,
            centers=centers.astype(np.float32),
            dimensions=dimensions,
            orientations=orientations,
            is_moving=bool(obj_data["is_moving"]),
            metadata={"original_type": obj_data["type"]},
            max_extrapolation_us=0.0,
        )

        scene_data.dynamic_objects[track_id] = dynamic_obj
    fix_static_objects(scene_data.dynamic_objects)

    # ==========================================================================
    # STEP 5: Lane Lines [OPTIONAL]
    # ==========================================================================
    # Load lane lines (lane markings painted on the road surface)
    # laneline_data: A list of dictionaries, each representing a lane line segment.
    # Each dictionary contains the following fields:
    # - element_id: Unique identifier for the lane line (e.g., 'laneline_0')
    # - points: numpy array of shape (N, 3) containing 3D coordinates [x, y, z] in world frame
    #   Each row represents a point along the lane line polyline
    # - color: String specifying the lane line color
    #   Supported values: 'YELLOW', 'WHITE', 'BLUE', 'GREEN', 'RED', 'UNKNOWN'
    # - style: String specifying the lane line style/pattern
    #   Supported values: 'SOLID_SINGLE', 'DASHED_SINGLE', 'SOLID_DOUBLE', 'DASHED_DOUBLE',
    #                     'SOLID_DASHED', 'DASHED_SOLID', 'SHORT_DASHED_SINGLE', 'LONG_DASHED_SINGLE',
    #                     'DOT_DASHED_SINGLE', 'DOT_SOLID_GROUP', 'SOLID_GROUP', 'OTHER', 'UNKNOWN'
    laneline_data = [
        {
            "element_id": "laneline_0",
            "points": np.array(
                [
                    [96.40687715, -88.53900518, -1.01595366],
                    [96.31889754, -102.57942297, -0.53467304],
                    [96.36777807, -122.14292952, 0.15236486],
                    [96.2829684, -134.13526702, 0.55741805],
                    [96.11321885, -149.93347894, 1.07017354],
                    [96.10399887, -150.63262903, 1.09437614],
                ]
            ),
            "color": "YELLOW",
            "style": "SOLID_SINGLE",
        },
        {
            "element_id": "laneline_1",
            "points": np.array(
                [
                    [-200.00589654, 1.59097945, -2.37797548],
                    [-198.66673657, 1.59993309, -2.34319519],
                    [-180.47463675, 1.73790701, -2.09719699],
                    [-171.77847685, 1.69881703, -1.95217507],
                    [-159.04968698, 1.73415973, -1.77710922],
                    [-152.02707705, 1.68661008, -1.66505142],
                    [-150.20641705, 1.68378967, -1.66428754],
                ]
            ),
            "color": "WHITE",
            "style": "DOT_DASHED_SINGLE",
        },
        {
            "element_id": "laneline_2",
            "points": np.array(
                [
                    [56.98224079, -80.43901341, 1.73134347],
                    [57.00140568, -106.33453578, 2.36900656],
                    [57.02601935, -126.71322779, 2.91091263],
                ]
            ),
            "color": "WHITE",
            "style": "LONG_DASHED_SINGLE",
        },
        {
            "element_id": "laneline_3",
            "points": np.array(
                [
                    [92.84263737, -92.32015127, -1.04076612],
                    [92.77563768, -104.30735286, -0.61249883],
                    [92.70099408, -117.79631445, -0.18477664],
                    [92.53575871, -140.85321739, 0.60297448],
                    [92.43706893, -148.39323833, 0.85452631],
                ]
            ),
            "color": "WHITE",
            "style": "LONG_DASHED_SINGLE",
        },
        {
            "element_id": "laneline_4",
            "points": np.array(
                [
                    [85.44810745, -87.00215379, -1.54121382],
                    [85.36981826, -117.22025373, -0.48162739],
                    [85.16627398, -143.16566699, 0.39415187],
                    [85.156469, -143.94644709, 0.41932143],
                ]
            ),
            "color": "WHITE",
            "style": "LONG_DASHED_SINGLE",
        },
        {
            "element_id": "laneline_5",
            "points": np.array(
                [
                    [81.5336676, -87.6458395, -1.67211769],
                    [81.5182779, -98.6381908, -1.3136554],
                    [81.496854, -103.630147, -1.11392194],
                    [81.4223783, -112.548063, -0.820723497],
                    [81.3268948, -132.612115, -0.14142249],
                    [81.2311991, -141.549086, 0.17803574],
                ]
            ),
            "color": "WHITE",
            "style": "SHORT_DASHED_SINGLE",
        },
    ]

    for lane in laneline_data:
        try:
            color = LaneLineColor[lane["color"].upper()]
        except KeyError:
            color = LaneLineColor.UNKNOWN
        try:
            style = LaneLineStyle[lane["style"].upper()]
        except KeyError:
            style = LaneLineStyle.UNKNOWN
        lane_line = LaneLine(
            element_id=lane["element_id"],
            points=lane["points"],
            lane_type=build_lane_line_type(
                color=color,
                style=style,
            ),
        )
        scene_data.lane_lines.append(lane_line)

    # ==========================================================================
    # STEP 6: Lane Boundaries [OPTIONAL]
    # ==========================================================================
    # Load lane boundaries (boundaries that separate lanes, e.g., curbs, barriers)
    # lanes: A list of dictionaries, each representing a lane boundary segment.
    # Each dictionary contains the following fields:
    # - element_id: Unique identifier for the lane boundary (e.g., 'lane_boundary_0')
    # - points: numpy array of shape (N, 3) containing 3D coordinates [x, y, z] in world frame
    #   Each row represents a point along the lane boundary polyline
    lanes = [
        {
            "element_id": "lane_boundary_0",
            "points": np.array([[-141.16596493, -57.486701, -0.33460385], [-141.22391481, -62.32049168, -0.1519915]]),
        },
        {
            "element_id": "lane_boundary_1",
            "points": np.array([[-145.227605, -57.4140886, -0.239536744], [-145.253085, -62.5524101, -0.102173575]]),
        },
        {
            "element_id": "lane_boundary_2",
            "points": np.array([[-141.16596493, -57.486701, -0.33460385], [-140.89751577, -33.45383384, -0.91196842]]),
        },
        {
            "element_id": "lane_boundary_3",
            "points": np.array([[-136.631735, -57.5571708, -0.0615086012], [-136.291346, -33.2776004, -0.69092895]]),
        },
        {
            "element_id": "lane_boundary_4",
            "points": np.array(
                [
                    [-150.18059616, -21.10970102, -1.26278773],
                    [-156.24741614, -21.24175825, -1.28461592],
                    [-161.71838612, -21.29524846, -1.30520269],
                ]
            ),
        },
    ]
    for lane in lanes:
        lane_boundary = LaneBoundary(
            element_id=lane["element_id"],
            points=lane["points"],
        )
        scene_data.lane_boundaries.append(lane_boundary)

    # ==========================================================================
    # STEP 7: Road Boundaries [OPTIONAL]
    # ==========================================================================
    # Load road boundaries (physical edges of the road, e.g., curbs, guardrails, walls)
    # road_boundaries: A list of dictionaries, each representing a road boundary segment.
    # Each dictionary contains the following fields:
    # - element_id: Unique identifier for the road boundary (e.g., 'road_boundary_0')
    # - points: numpy array of shape (N, 3) containing 3D coordinates [x, y, z] in world frame
    #   Each row represents a point along the road boundary polyline
    road_boundaries = [
        {
            "element_id": "road_boundary_0",
            "points": np.array([[-200.00679632, -2.99880323, -2.35654961], [-188.9631264, -2.89996687, -2.24689122]]),
        },
        {
            "element_id": "road_boundary_1",
            "points": np.array(
                [
                    [-176.66989579, -23.84653519, -1.58473766],
                    [-161.69324604, -23.78910372, -1.23374953],
                    [-150.18053611, -23.81158349, -1.13524706],
                    [-146.8044161, -23.82613228, -1.14430832],
                ]
            ),
        },
        {
            "element_id": "road_boundary_2",
            "points": np.array([[-178.22403623, -12.16171544, -1.80859669], [-178.23976651, -5.5124931, -1.90010892]]),
        },
        {
            "element_id": "road_boundary_3",
            "points": np.array(
                [
                    [-65.68234556, -76.35789048, 1.92445108],
                    [-60.83985375, -76.42612527, 2.20548756],
                    [-57.41135275, -76.54898109, 2.20472534],
                    [-56.30037177, -76.55441113, 2.2313162],
                    [-55.19465078, -76.48047713, 2.24608426],
                    [-54.59371579, -76.3812871, 2.24824435],
                    [-54.0022188, -76.23971108, 2.2489546],
                    [-53.4632478, -76.05816704, 2.24716165],
                    [-52.57858882, -75.60691097, 2.23922546],
                    [-51.70315784, -75.01639588, 2.22710377],
                    [-50.54488587, -74.05375069, 2.19226016],
                    [-49.76266388, -73.2838745, 2.15174858],
                    [-49.08280589, -72.4839003, 2.10770883],
                    [-48.67113392, -71.87047525, 2.1010602],
                    [-47.82574396, -70.38350005, 2.05753941],
                ]
            ),
        },
        {
            "element_id": "road_boundary_4",
            "points": np.array([[-57.33598982, -73.30683058, 2.06784852], [-65.61303566, -73.38441418, 1.84577997]]),
        },
    ]
    for road_boundary_item in road_boundaries:
        road_boundary = RoadBoundary(
            element_id=road_boundary_item["element_id"],
            points=road_boundary_item["points"],
        )
        scene_data.road_boundaries.append(road_boundary)

    # ==========================================================================
    # STEP 8: Crosswalks [OPTIONAL]
    # ==========================================================================
    # Load crosswalks (pedestrian crossing areas)
    # crosswalks: A list of dictionaries, each representing a crosswalk polygon.
    # Each dictionary contains the following fields:
    # - element_id: Unique identifier for the crosswalk (e.g., 'crosswalk_0')
    # - vertices: numpy array of shape (N, 3) containing 3D coordinates [x, y, z] in world frame
    #   Each row represents a vertex of the crosswalk polygon (dtype=np.float32)
    crosswalks = [
        {
            "element_id": "crosswalk_0",
            "vertices": np.array(
                [
                    [-147.28804, 20.04666, -2.1015491],
                    [-146.80101, 7.9370174, -1.7449733],
                    [-146.24974, -2.2952774, -1.5924973],
                    [-149.91774, -2.3174927, -1.633071],
                    [-151.20915, 16.582893, -2.1135685],
                    [-149.23979, 17.550648, -2.0817218],
                    [-147.89851, 18.94918, -2.0863526],
                ],
                dtype=np.float32,
            ),
        },
        {
            "element_id": "crosswalk_1",
            "vertices": np.array(
                [
                    [-132.70198, 17.011494, -1.8371806],
                    [-147.60928, 16.529783, -2.0100906],
                    [-147.16847, 20.336943, -2.1066644],
                    [-135.15324, 20.576433, -1.9555995],
                    [-134.10806, 18.556128, -1.9099921],
                ],
                dtype=np.float32,
            ),
        },
    ]
    for crosswalk_item in crosswalks:
        crosswalk = Crosswalk(
            element_id=crosswalk_item["element_id"],
            vertices=crosswalk_item["vertices"],
        )
        scene_data.crosswalks.append(crosswalk)

    # ==========================================================================
    # STEP 9: Traffic Signs [OPTIONAL]
    # ==========================================================================
    # Load traffic signs (road signs like stop signs, speed limits, etc.)
    # traffic_signs: A list of dictionaries, each representing a traffic sign.
    # Each dictionary contains the following fields:
    # - element_id: Unique identifier for the traffic sign (e.g., 'traffic_sign_0')
    # - center: numpy array of shape (3,) containing 3D position [x, y, z] in world frame (dtype=np.float32)
    # - dimensions: numpy array of shape (3,) containing sign dimensions [depth, width, height] (dtype=np.float32)
    # - orientation: numpy array of shape (4,) containing quaternion [w, x, y, z] for sign orientation (dtype=np.float32)
    traffic_signs = [
        {
            "element_id": "traffic_sign_0",
            "center": np.array([-177.57785, -7.294852, -0.5110163], dtype=np.float32),
            "dimensions": np.array([0.01, 0.3114329, 0.46135485], dtype=np.float32),
            "orientation": np.array([0.95607156, -0.01418184, 0.00490689, 0.29274896], dtype=np.float32),
        },
        {
            "element_id": "traffic_sign_1",
            "center": np.array([-185.94432, -12.632289, -0.83062035], dtype=np.float32),
            "dimensions": np.array([0.01, 0.31310344, 0.44303071], dtype=np.float32),
            "orientation": np.array([-0.62973928, 0.00049557892, -0.019341411, 0.77656561], dtype=np.float32),
        },
    ]
    for traffic_sign_item in traffic_signs:
        traffic_sign = TrafficSign(
            element_id=traffic_sign_item["element_id"],
            center=traffic_sign_item["center"],
            dimensions=traffic_sign_item["dimensions"],
            orientation=traffic_sign_item["orientation"],
        )
        scene_data.traffic_signs.append(traffic_sign)

    # ==========================================================================
    # STEP 10: Poles [OPTIONAL]
    # ==========================================================================
    # Load poles (vertical structures like lamp posts, traffic light poles, utility poles)
    # poles: A list of dictionaries, each representing a pole.
    # Each dictionary contains the following fields:
    # - element_id: Unique identifier for the pole (e.g., 'pole_0')
    # - points: numpy array of shape (2, 3) containing 3D coordinates [x, y, z] in world frame
    #   First row is the top point, second row is the bottom point of the pole
    poles = [
        {
            "element_id": "pole_0",
            "points": np.array([[-177.4852185, -7.22948493, 1.02735274], [-177.46898668, -7.38523394, -1.52917452]]),
        },
        {
            "element_id": "pole_1",
            "points": np.array([[-87.64449698, -47.71341127, 1.83355044], [-87.71322621, -48.07716037, 0.78075561]]),
        },
    ]
    for pole_item in poles:
        pole = Pole(
            element_id=pole_item["element_id"],
            points=pole_item["points"],
        )
        scene_data.poles.append(pole)

    # ==========================================================================
    # STEP 11: Wait Lines [OPTIONAL]
    # ==========================================================================
    # Load wait lines (stop lines at intersections where vehicles should wait)
    # wait_lines: A list of dictionaries, each representing a wait/stop line.
    # Each dictionary contains the following fields:
    # - element_id: Unique identifier for the wait line (e.g., 'wait_line_0')
    # - points: numpy array of shape (2, 3) containing 3D coordinates [x, y, z] in world frame
    #   Two points define the start and end of the wait line
    wait_lines = [
        {
            "element_id": "wait_line_0",
            "points": np.array([[-141.14844683, -7.43866935, -1.31700508], [-137.77913692, -7.5475187, -1.1749609]]),
        },
        {
            "element_id": "wait_line_1",
            "points": np.array([[-196.36947784, 32.67426037, -2.81746752], [-200.2944778, 32.66141438, -2.87685535]]),
        },
    ]
    for wait_line_item in wait_lines:
        wait_line = WaitLine(
            element_id=wait_line_item["element_id"],
            points=wait_line_item["points"],
        )
        scene_data.wait_lines.append(wait_line)

    # ==========================================================================
    # STEP 12: Road Markings [OPTIONAL]
    # ==========================================================================
    # Load road markings (painted symbols/text on road surface, e.g., arrows, text, symbols)
    # road_markings: A list of dictionaries, each representing a road marking polygon.
    # Each dictionary contains the following fields:
    # - element_id: Unique identifier for the road marking (e.g., 'road_marking_0')
    # - vertices: numpy array of shape (N, 3) containing 3D coordinates [x, y, z] in world frame
    #   Each row represents a vertex of the road marking polygon (dtype=np.float32)
    road_markings = [
        {
            "element_id": "road_marking_0",
            "vertices": np.array(
                [
                    [-161.96597, -18.275469, -1.3583272],
                    [-161.96617, -16.648682, -1.3458775],
                    [-159.36215, -16.597614, -1.3769454],
                    [-159.42549, -18.235456, -1.3321064],
                ],
                dtype=np.float32,
            ),
        },
        {
            "element_id": "road_marking_1",
            "vertices": np.array(
                [
                    [-157.89343, -23.77597, -1.1425674],
                    [-159.63556, -23.76312, -1.1415883],
                    [-159.36215, -16.597614, -1.3769454],
                    [-157.58902, -16.599054, -1.3305275],
                ],
                dtype=np.float32,
            ),
        },
    ]
    for road_marking_item in road_markings:
        road_marking = RoadMarking(
            element_id=road_marking_item["element_id"],
            vertices=road_marking_item["vertices"],
        )
        scene_data.road_markings.append(road_marking)

    # ==========================================================================
    # STEP 13: Finalize Scene Data
    # ==========================================================================
    # Update duration
    if scene_data.ego_poses:
        scene_data.duration_seconds = len(scene_data.ego_poses) / scene_data.frame_rate

    # Parse camera names after loading to see what's available
    if "all" in camera_names:
        camera_names = list(scene_data.camera_models.keys())

    # Convert to rendering format
    all_camera_models, all_camera_poses = convert_scene_data_for_rendering(
        scene_data,
        camera_names,
        SETTINGS["RESIZE_RESOLUTION"],
    )

    # ==========================================================================
    # STEP 14: Render to Video
    # ==========================================================================
    output_dir = "output_render_test_case"
    max_frames = 2
    chunk_output = False
    overlay_camera = False
    data_path = "None"  # This setting is only for overlay
    alpha = 0.5
    use_persistent_vbos = True
    multi_sample = 4

    render_multi_camera_tiled(
        all_camera_models,
        all_camera_poses,
        scene_data,
        camera_names,
        output_dir,
        scene_data.scene_id,
        max_frames,
        chunk_output,
        overlay_camera,
        alpha,
        data_path,
        use_persistent_vbos,
        multi_sample,
    )


if __name__ == "__main__":
    render_test_case()
