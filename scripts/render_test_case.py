import numpy as np
import cv2
from pathlib import Path
from av_utils.bbox_utils import create_bbox_geometry_objects_for_frame
from av_utils.laneline_utils import  create_laneline_geometry_objects_from_data
from av_utils.minimap_utils import create_minimap_geometry_objects_from_data
from av_utils.graphics_utils import render_geometries
from av_utils.camera.pinhole import PinholeCamera

def render_sample_hdmap_v3():
    """
    Main rendering function for cosmos-transfer v2.5.
    
    This function demonstrates how to render HD map elements, lane lines, and 3D bounding boxes
    into camera view. If you feel it's difficult to align the data format, 
    you can try to understand this code To integrate this rendering pipeline into your own codebase, align your
    own data format with the input format of the following utility functions:
    - create_minimap_geometry_objects_from_data
    - create_laneline_geometry_objects_from_data
    - create_bbox_geometry_objects_for_frame
    
    Then copy these functions and their corresponding utils code into your project.
    Detailed format specifications are provided in the comments below.
    The example data is open-sourced data from WAYMO dataset.
    """

    # ========== Object Information ==========
    # all_object_info: A nested dictionary containing object annotations.
    # - Outer key: frame_id (e.g., '000000.all_object_info.json'), representing the frame number
    # - Inner key: object_id (unique identifier for each object in the frame)
    # 
    # Note: Both frame_id and object_id can be customized based on your actual data format.
    # 
    # Each object dictionary contains the following fields:
    # - object_to_world: 4x4 transformation matrix (numpy array) representing object pose in world coordinates
    #   If world coordinates are not available, use object-to-ego transformation instead (keep the key name unchanged)
    # - object_lwh: Object dimensions [length, width, height] as a 3-element array
    #   Example: [4.698626484833993, 2.1684678367586816, 1.6800000000000064]
    # - object_is_moving: Boolean flag indicating whether the object is moving
    #   Example: False
    # - object_type: String specifying the object category
    #   Supported types: 'Car', 'Truck', 'Pedestrian', 'Cyclist', 'Others'
    #   Example: 'Car'
    all_object_info = {
        '000000.all_object_info.json': {
            '0jXFwa0ipnBsrKs7iweXeg': {'object_to_world': [[0.9810033468056915, -0.19303916669260013, 0.01919206683395235, -372.9239874079048], [0.19283847588427486, 0.9811592388540997, 0.011826785897378969, 1460.0897147886062], [-0.02111292908769759, -0.00790114762995593, 0.9997458635603244, 6.130314074095118], [0.0, 0.0, 0.0, 1.0]], 
                                        'object_lwh': [4.698626484833993, 2.1684678367586816, 1.6800000000000064], 
                                        'object_is_moving': False, 
                                        'object_type': 'Car'},
            '7YXnueTWgDkYCBcmd76UpQ': {'object_to_world': [[-0.9990954845300554, 0.0335510082340885, 0.026125516992592385, -382.7044489177638], [-0.03325310339051761, -0.9993778306793842, 0.011755112141871816, 1386.553191662891], [0.026503658361698938, 0.010875724943403157, 0.9995895531168792, 12.15409549675546], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.605977307800487, 2.0849057025635354, 2.1799999999999997], 'object_is_moving': True, 'object_type': 'Car'}
        } ,
        '000001.all_object_info.json': {
                '0jXFwa0ipnBsrKs7iweXeg': {'object_to_world': [[0.9810033468056915, -0.19303916669260013, 0.01919206683395235, -372.9239874079048], [0.19283847588427486, 0.9811592388540997, 0.011826785897378969, 1460.0897147886062], [-0.02111292908769759, -0.00790114762995593, 0.9997458635603244, 6.130314074095118], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.698626484833993, 2.1684678367586816, 1.6800000000000064], 'object_is_moving': False, 'object_type': 'Car'}, '7YXnueTWgDkYCBcmd76UpQ': {'object_to_world': [[-0.9990954845300554, 0.0335510082340885, 0.026125516992592385, -382.7044489177638], [-0.03325310339051761, -0.9993778306793842, 0.011755112141871816, 1386.553191662891], [0.026503658361698938, 0.010875724943403157, 0.9995895531168792, 12.15409549675546], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.605977307800487, 2.0849057025635354, 2.1799999999999997], 'object_is_moving': True, 'object_type': 'Car'}, '8MNPn67dbl63RA7ebKEs_A': {'object_to_world': [[0.9995882240600623, 0.011867589574995452, 0.026125516992592385, -434.6325198314958], [-0.012178012494789246, 0.9998567464143097, 0.011755112141871816, 1371.4712164090845], [-0.025982269572297396, -0.01206842854188914, 0.9995895531168792, 12.333113979604189], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [16.617961942131657, 2.988683196198371, 1.9900000000000038], 'object_is_moving': True, 'object_type': 'Car'}, '9XDhAMiadbJOGRDv8B8_cQ': {'object_to_world': [[-0.9995752787923903, 0.012911986252517833, 0.026125516992592378, -390.6644033299939], [-0.01260831433356377, -0.9998514128350262, 0.011755112141871813, 1393.4691744975055], [0.02627341692246163, 0.011420720766077863, 0.999589553116879, 12.050696121741298], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.6387289814468895, 2.0508126279540413, 1.8700000000000045], 'object_is_moving': True, 'object_type': 'Car'}, 'ApuXUfEFETPMrCt5lgpjqw': {'object_to_world': [[-0.9993910545567518, 0.01971995544100503, 0.02878619476080342, -373.80493862841445], [-0.019381373100315537, -0.9997402192673052, 0.0119940138764966, 1390.0951742874784], [0.029015210643555858, 0.01142879419560278, 0.9995136310337761, 11.518974592673962], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.475768555554, 2.040813772447736, 1.47], 'object_is_moving': False, 'object_type': 'Car'}, 'CDpk2HxG9zo8L3wb6eiJYQ': {'object_to_world': [[-0.9996560573088467, 0.0022857050555927074, 0.026125516992592374, -392.68403570991336], [-0.0019791149711015288, -0.9999289476970166, 0.011755112141871811, 1383.2202649408582], [0.02615052943369516, 0.011699363655159122, 0.9995895531168789, 13.134241933696295], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [7.019925159854434, 3.0715977302150907, 3.5599999999999703], 'object_is_moving': True, 'object_type': 'Car'}, 'FF66eikA7UNidfhOo6K8Mw': {'object_to_world': [[-0.9995370422370967, 0.011882011919858356, 0.02800929947356617, -364.88671125540276], [-0.011549186683083763, -0.9998611224432266, 0.012014662824671353, 1390.069828843425], [0.028148134625153382, 0.011685615914765397, 0.9995354556093593, 11.355209748079817], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.5617116251391865, 2.0414557473728907, 1.5600000000000032], 'object_is_moving': False, 'object_type': 'Car'}, 'Fdyi0MfUNJ9IPgCVC5SzhQ': {'object_to_world': [[-0.9991679483084143, -0.031318819183103565, 0.026125516992592378, -385.4036618279566], [0.031634409156431696, -0.999430378513609, 0.011755112141871813, 1382.4726758645215], [0.025742479105121938, 0.012571796574896366, 0.999589553116879, 12.71434029717883], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.232063560466546, 2.203372213225901, 3.149999999999979], 'object_is_moving': True, 'object_type': 'Car'}, 'IFbgB6nXjtWoQbG0V4qvJg': {'object_to_world': [[-0.8781846478942389, -0.47760776958163204, 0.026125516992592374, -396.2939854165563], [0.4780076954610823, -0.8782769838829417, 0.011755112141871811, 1402.129805450924], [0.01733110737537522, 0.02281135718762542, 0.9995895531168789, 11.80818871830239], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.644034531435028, 2.0090277379839, 1.6300000000000043], 'object_is_moving': True, 'object_type': 'Car'}, 'KdhPci5k83yIqH6LHLt4hg': {'object_to_world': [[0.999636429027319, -0.006668367369861577, 0.026125516992592378, -424.56297935991165], [0.006362976635799337, 0.9999106609426985, 0.011755112141871813, 1371.3341872911215], [-0.026201570369768667, -0.011584602270094375, 0.999589553116879, 12.891994310973482], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [6.083109031283052, 3.25073295180958, 2.5099999999999927], 'object_is_moving': True, 'object_type': 'Car'}, 'OjS8-f5p3CE4XqevYlK-gw': {'object_to_world': [[0.9818703486663479, -0.18860958468209355, 0.01889631747960446, -362.4603878464028], [0.18839423026685095, 0.9820125306155176, 0.012609808909769591, 1471.6448980000935], [-0.020934001337182432, -0.008821240284405849, 0.9997419276518161, 6.060964972315766], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [3.8520842838897047, 1.7774080982367995, 1.8800000000000001], 'object_is_moving': False, 'object_type': 'Car'}, 'P1pHMj0J3nm2t6lhmuwUeQ': {'object_to_world': [[-0.9996495902523259, 0.004260759348609708, 0.026125516992592378, -382.3852482808229], [-0.003954709022502949, -0.9999230858496465, 0.011755112141871813, 1393.6538997990654], [0.026173593274602787, 0.01164767421822414, 0.999589553116879, 11.964421560816916], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [5.2191688176686215, 2.1464582495413076, 2.110000000000003], 'object_is_moving': True, 'object_type': 'Car'}, 'SdoipodyP32lZsbRMYrC3A': {'object_to_world': [[0.9725890278659111, -0.23168436600060702, 0.01982346709902052, -362.902873110388], [0.2314356458126721, 0.9727485047017576, 0.014067477484250272, 1474.2062316368178], [-0.02254162199384753, -0.00909401734062771, 0.9997045244618052, 6.140229318706957], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.038944698156169, 1.9928756091425954, 1.8900000000000081], 'object_is_moving': False, 'object_type': 'Car'}, 'Uyes785HqqO9671zhYLHUA': {'object_to_world': [[-0.19345994647019302, -0.9807602696243459, 0.026125516992592378, -410.67272079312835], [0.9810867663494732, -0.19321121663212573, 0.011755112141871813, 1428.972455710827], [-0.006481204030444575, 0.027905542351188306, 0.999589553116879, 12.957453977582198], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [7.082310676574707, 2.825537681579643, 3.2299999999999915], 'object_is_moving': True, 'object_type': 'Car'}, 'Va4VoAu8hxGuWW0lxOnJPQ': {'object_to_world': [[-0.9994961540893295, -0.014460640436934647, 0.028254706616330116, -375.9489093377799], [0.014810243851877323, -0.9998158495028527, 0.012203432142273303, 1383.0731542190529], [0.028073023308638326, 0.012615742587843376, 0.9995262616849915, 11.733666012511733], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.736656188964844, 2.144813060760334, 1.6700000000000035], 'object_is_moving': False, 'object_type': 'Car'}, 'YHB0qWkx7vqSVkN9AcpOjw': {'object_to_world': [[-0.9995011087319953, -0.017747985952832564, 0.026125516992592378, -373.3510191285997], [0.018059983224233186, -0.9997678002138659, 0.011755112141871813, 1398.5284551451286], [0.025910821087966145, 0.012221074017680454, 0.999589553116879, 11.442409475870038], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.243753465165197, 2.011677737389861, 1.5800000000000036], 'object_is_moving': True, 'object_type': 'Car'}, '_Em9baV0QLUKwKd_l9j33w': {'object_to_world': [[-0.9995969163810072, 0.0012520575540606866, 0.028362625087110623, -374.0690424310517], [-0.0009123079683346384, -0.999927718414578, 0.011988562921516687, 1386.600740749205], [0.028375560888773483, 0.011957855079317905, 0.9995258054984097, 11.63767927015681], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.831748034199749, 1.9670940367520573, 1.5800000000000045], 'object_is_moving': False, 'object_type': 'Car'}, '_ObcLKstHDXCT26inxqDVg': {'object_to_world': [[-0.9840071858270257, -0.17620248466649693, 0.026125516992592378, -387.59933954304387], [0.17655286461858144, -0.9842209626570169, 0.011755112141871813, 1398.9392102002682], [0.023642001517430403, 0.016179649682487994, 0.999589553116879, 11.80145533459305], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.341783046722412, 1.8740310668946354, 1.6100000000000048], 'object_is_moving': True, 'object_type': 'Car'}, 'buozXKmmAWM8bsfeS2E9ww': {'object_to_world': [[-0.9996558113720337, -0.0023908475465314587, 0.026125516992592378, -380.9068541596106], [0.0026987108946612175, -0.9999272645037934, 0.011755112141871813, 1389.7099672881404], [0.026095512059126674, 0.01182157138328868, 0.999589553116879, 11.926594750112004], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [5.287640571594238, 2.158054351806721, 1.9700000000000042], 'object_is_moving': True, 'object_type': 'Car'}, 'i5wuxzS4_p7gNr6CSlxZtA': {'object_to_world': [[-0.9994237721032396, -0.016685063107834984, 0.029558555034393107, -367.088266622073], [0.017047077541246958, -0.9997822204696718, 0.012037813276272357, 1386.871479137574], [0.02935167631046453, 0.01253476373212615, 0.9994905616741983, 11.70860179982626], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.960479781255451, 2.235798777223253, 1.8600000000000054], 'object_is_moving': False, 'object_type': 'Car'}, 'n2NZ40wt1uUPtbrF5mrDCA': {'object_to_world': [[-0.9799031774364311, 0.19856601767935095, 0.019008555635020626, -374.5536521750163], [-0.1983571646740909, -0.9800523118473677, 0.012324960046713294, 1468.4572581504337], [0.02107600703268819, 0.008306784311240992, 0.9997433521521998, 5.9932331305266615], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.861531205620283, 2.2017198489388865, 1.8000000000000096], 'object_is_moving': False, 'object_type': 'Car'}, 'nRfo8gNaT86l9p5TPJ88kg': {'object_to_world': [[0.9818667596982716, -0.18860649698498277, 0.019112268728099915, -358.9068674692844], [0.18840158618186778, 0.9820184598488282, 0.012024543888457171, 1460.1775959278357], [-0.02103588656968948, -0.008205718200702549, 0.9997450332601505, 5.958138963311444], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [5.052132436417317, 1.8516006430176561, 1.6899999999999995], 'object_is_moving': False, 'object_type': 'Car'}, 't3NeQE0g1LIKRRCHSVhRCQ': {'object_to_world': [[-0.9996456073348677, -0.0051104890150975935, 0.026125516992592378, -390.3805792394594], [0.0054190897356688845, -0.9999162218930989, 0.011755112141871813, 1386.4197232442373], [0.02606325387476465, 0.011892522737324527, 0.999589553116879, 11.926163170283433], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.420174598693848, 1.9115607738496085, 1.810000000000004], 'object_is_moving': True, 'object_type': 'Car'}, 'w_uCN0b1-WWgab9UuPPOiw': {'object_to_world': [[-0.9995361746987873, -0.015649052060285377, 0.026125516992592378, -391.6708726390177], [0.015960488607017626, -0.9998035207689346, 0.011755112141871813, 1389.6284634237115], [0.025936427509219845, 0.012166635839754535, 0.999589553116879, 11.942898495638628], [0.0, 0.0, 0.0, 1.0]], 'object_lwh': [4.570503206327252, 1.7427639763029286, 1.600000000000005], 'object_is_moving': True, 'object_type': 'Car'}
        }
         ,
    }

    # ========== Camera Intrinsics ==========
    # intrinsic_this_cam: A 6-element numpy array containing camera intrinsic parameters
    # Format: [fx, fy, cx, cy, w, h]
    # - fx, fy: Focal lengths in pixels
    # - cx, cy: Principal point coordinates
    # - w, h: Original image width and height
    # Example: array([2076.0849352, 2076.0849352, 1013.33840467, 248.59750025, 1920., 886.])
    
    intrinsic_this_cam = np.array([2076.0849352 , 2076.0849352 , 1013.33840467,  248.59750025,1920.,  886. ])
    
    # Target resolution for rendering
    camera_model_h = 720   # Target height
    camera_model_w = 1280  # Target width
    
    # Initialize camera model
    camera_model = PinholeCamera.from_numpy(intrinsic_this_cam, device='cpu')
    rescale_h = camera_model_h / camera_model.height
    rescale_w = camera_model_w / camera_model.width
    
    # Rescale camera model to target resolution
    camera_model.rescale(rescale_h, rescale_w)

    # ========== Camera Poses ==========
    # pose_all_frames: Camera-to-world transformation matrices for all frames
    # - Convention: OpenCV convention
    # - Type: numpy.ndarray
    # - Shape: [T, 4, 4] where T is the number of frames
    pose_all_frames = np.array([[[ 8.18104170e-03, -2.42539431e-02, -9.99672355e-01,
         -3.60896706e+02],
        [ 9.99913935e-01, -1.00552650e-02,  8.42697849e-03,
          1.39848054e+03],
        [-1.02563579e-02, -9.99655260e-01,  2.41695931e-02,
          1.24938211e+01],
        [-0.00000000e+00, -0.00000000e+00,  0.00000000e+00,
          1.00000000e+00]],

       [[ 7.98361092e-03, -2.48905240e-02, -9.99658304e-01,
         -3.61016955e+02],
        [ 9.99915885e-01, -1.00202438e-02,  8.23516243e-03,
          1.39848171e+03],
        [-1.02217974e-02, -9.99639963e-01,  2.48084326e-02,
          1.24971061e+01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          1.00000000e+00]]])
        

    # ========== Rendering Colors for Map Elements ==========
    # minimap_to_rgb: Rendering colors for each HD map element type
    # - Keys: element type names (string)
    # - Values: RGB color values [R, G, B] in range [0, 255]
    # These colors are typically extracted from color scheme configuration files
    minimap_to_rgb = {'lanelines': [98, 183, 249], 'poles': [183, 69, 177], 'road_boundaries': [253, 1, 232], 'wait_lines': [108, 179, 59], 
        'crosswalks': [139, 93, 255], 'road_markings': [20, 254, 185], 'traffic_signs': [8, 2, 255], 'traffic_lights': [100, 100, 100]}

    # ========== Minimap Data ==========
    # minimap_name_to_minimap_data: Dictionary mapping element type to geometric data
    # - Key: Element type name (e.g., 'road_boundaries', 'crosswalks')
    # - Value: List of numpy arrays, each with shape [N, 3] representing 3D coordinates (x, y, z)
    minimap_name_to_minimap_data = {'road_boundaries': [np.array([[-405.41573266, 1301.03694931,   10.03837072],
       [-405.37320402, 1301.53193546,   10.05246908],
       [-405.33060533, 1302.02691559,   10.06656744],
       [-405.2879366 , 1302.52188969,   10.0806658 ],
       [-405.24519782, 1303.01685774,   10.09476416],
       [-405.202389  , 1303.51181973,   10.10886252],
       [-405.15951014, 1304.00677567,   10.12296089],
       [-405.11656123, 1304.50172553,   10.13705925],
       [-405.07354229, 1304.9966693 ,   10.15115761],
       [-295.99671571, 1360.95686724,    8.45087345]]), 
       np.array([[-463.89746938, 1326.3300332 ,    4.76136468],
       [-463.89968598, 1326.82943728,    4.76136468],
       [-463.90185642, 1327.32884157,    4.76136468],
       [-463.90398156, 1327.82824605,    4.76136468],
       [-463.90606221, 1328.32765071,    4.76136468],
       [-463.90809923, 1328.82705556,    4.76136468],
       [-463.91009344, 1329.32646058,    4.76136468],
       [-463.91204568, 1329.82586576,    4.76136468],
       [-456.34754207, 1348.67801305,    4.48471842],
       [-456.3103539 , 1348.18004609,    4.48123004]])],
    'crosswalks': [np.array([[-392.51644071, 1352.09516576,   11.27420679],
       [-393.88544878, 1369.19747324,   11.27420679],
       [-394.78880784, 1381.92993043,   11.27420679],
       [-395.51522028, 1396.07094343,   11.27420679],
       [-396.17644186, 1408.98085647,   11.27420679],
       [-399.63155745, 1408.80340062,   11.27420679],
       [-398.96102289, 1395.89348758,   11.27420679],
       [-398.23461045, 1381.69701963,   11.27420679],
       [-397.33125139, 1368.93128947,   11.27420679],
       [-395.96224332, 1351.817891  ,   11.27420679],
       [-392.51644071, 1352.09516576,   11.27420679]]), np.array([[-390.64453173, 1356.39847011,   11.28320679],
       [-414.51369954, 1357.66284304,   11.28320679],
       [-414.7185851 , 1353.84754228,   11.28320679],
       [-390.84941729, 1352.58316935,   11.28320679],
       [-390.64453173, 1356.39847011,   11.28320679]]), np.array([[-394.21140308, 1404.71082509,   11.32820679],
       [-395.05888426, 1408.01594029,   11.32820679],
       [-414.06667649, 1403.13590443,   11.32820679],
       [-413.21919531, 1399.83078924,   11.32820679],
       [-394.21140308, 1404.71082509,   11.32820679]])]
    }

    # ========== Lane Line Data ==========
    # processed_lanelines: List of dictionaries, each representing a lane line
    # Each dictionary contains the following keys:
    # - 'pattern_segments_list': List containing one numpy array of shape [N, 2, 3]
    #   Represents N line segments, each defined by 2 endpoints with 3D coordinates
    #   Notice that the format is [N,2,3] instead of [N,3] 
    #   Given a subdivided polyline with N points, the line segments are constructed by stacking the first N-1 points and the last N-1 points.
    #   eg: pattern_segments_list = np.stack([polyline_subdivided[0:-1], polyline_subdivided[1:]], axis=1)
    # - 'rgb_float': Lane line rendering color as a 3-element array in range [0, 1]
    #   Example: [0.70980392, 0.64313725, 0.27843137]
    # - 'line_width': Width of the lane line in pixels (scalar value)
    #   Example: 12.0
    # Note: This example contains only one lane line; multiple lane lines would have multiple dictionaries
    processed_lanelines = [
            {
            "pattern_segments_list": [
            np.array([
            [[-470.61360269, 1332.26447099,    4.83082974],
        [-470.61383741, 1332.32696876,    4.83111662]],

       [[-470.61383741, 1332.32696876,    4.83111662],
        [-470.61407212, 1332.38946653,    4.83140351]],

       [[-470.61407212, 1332.38946653,    4.83140351],
        [-470.61430684, 1332.4519643 ,    4.83169039]],

       [[-470.61430684, 1332.4519643 ,    4.83169039],
        [-470.61454155, 1332.51446207,    4.83197728]],

       [[-470.61454155, 1332.51446207,    4.83197728],
        [-470.61477626, 1332.57695984,    4.83226416]]
        ])  
        ]  ,
        "rgb_float": np.array([0.70980392, 0.64313725, 0.27843137]),
        "line_width": 12.0
            },
        ]

    # ========== Bounding Box Color Configuration ==========
    # bbox_per_vertex_color_map: Per-vertex color mapping for each object type
    # The front and back vertices of each bounding box have different colors to create a gradient effect
    # during rendering, which helps visualize the object's orientation
    bbox_per_vertex_color_map = {'Car': np.array([[0.        , 0.18039216, 0.53333333],
       [0.        , 0.18039216, 0.53333333],
       [0.49411765, 0.80784314, 1.        ],
       [0.49411765, 0.80784314, 1.        ],
       [0.        , 0.18039216, 0.53333333],
       [0.        , 0.18039216, 0.53333333],
       [0.49411765, 0.80784314, 1.        ],
       [0.49411765, 0.80784314, 1.        ]]), 'Truck': np.array([[0.8       , 0.21568627, 0.        ],
       [0.8       , 0.21568627, 0.        ],
       [1.        , 0.75294118, 0.25098039],
       [1.        , 0.75294118, 0.25098039],
       [0.8       , 0.21568627, 0.        ],
       [0.8       , 0.21568627, 0.        ],
       [1.        , 0.75294118, 0.25098039],
       [1.        , 0.75294118, 0.25098039]]), 'Pedestrian': np.array([[0.58039216, 0.        , 0.24313725],
       [0.58039216, 0.        , 0.24313725],
       [1.        , 0.48627451, 0.67058824],
       [1.        , 0.48627451, 0.67058824],
       [0.58039216, 0.        , 0.24313725],
       [0.58039216, 0.        , 0.24313725],
       [1.        , 0.48627451, 0.67058824],
       [1.        , 0.48627451, 0.67058824]]), 'Cyclist': np.array([[0.        , 0.31372549, 0.25882353],
       [0.        , 0.31372549, 0.25882353],
       [0.4       , 0.81568627, 0.77647059],
       [0.4       , 0.81568627, 0.77647059],
       [0.        , 0.31372549, 0.25882353],
       [0.        , 0.31372549, 0.25882353],
       [0.4       , 0.81568627, 0.77647059],
       [0.4       , 0.81568627, 0.77647059]]), 'Others': np.array([[0.20784314, 0.10196078, 0.07843137],
       [0.20784314, 0.10196078, 0.07843137],
       [0.65098039, 0.53333333, 0.49019608],
       [0.65098039, 0.53333333, 0.49019608],
       [0.20784314, 0.10196078, 0.07843137],
       [0.20784314, 0.10196078, 0.07843137],
       [0.65098039, 0.53333333, 0.49019608],
       [0.65098039, 0.53333333, 0.49019608]])}
        
    bbox_color_version = "v3"

    # ========== Reference Coordinate Frames ==========
    # camera_pose_init: Initial camera pose at the start of the clip (shape: [4, 4])
    # Essentially represents the camera-to-ego transformation
    # If world coordinates are not available, set this to camera_to_ego directly
    camera_pose_init = pose_all_frames[0]

    # camera_pose: Current frame's camera pose (shape: [4, 4])
    # If world coordinates are not available and ego frame is used as reference,
    # this should be the camera-to-ego transformation
    camera_pose = pose_all_frames[1]

    # ========== Rendering Pipeline ==========
    # Collect all geometry objects to be rendered together
    geometry_objects = []

    # Add HD map elements (road boundaries and crosswalks)
    # Input format for create_minimap_geometry_objects_from_data is described above
    # camera_pose_init represents the camera-to-ego transformation, assumed to be from the first frame
    geometry_objects.extend(
            create_minimap_geometry_objects_from_data(
                    minimap_name_to_minimap_data,
                    camera_pose,
                    camera_model,
                    minimap_to_rgb,
                    camera_pose_init=camera_pose_init,
                )
            )

    # Add lane lines
    # Input format for create_laneline_geometry_objects_from_data is described above
    geometry_objects.extend(
                create_laneline_geometry_objects_from_data(
                    processed_lanelines,
                    camera_pose,
                    camera_model,
                    camera_pose_init=camera_pose_init,
                )
            )

    # Add traffic lights (optional)
    # Note: Waymo dataset does not include traffic light data, so this is commented out
    # If you need to render traffic lights, provide your own traffic light data
    #geometry_objects.extend(
    #    create_traffic_light_status_geometry_objects_from_data(
    #        tl_position_list,
    #        tl_status_dict,
    #        frame_id,
    #        camera_pose,
    #        camera_model,
    #        tl_status_to_rgb,
    #    )
    #)

    # Add 3D bounding boxes
    geometry_objects.extend(
                create_bbox_geometry_objects_for_frame(
                    all_object_info[f"000001.all_object_info.json"],
                    camera_pose,
                    camera_model,
                    fill_face='all',
                    fill_face_style='solid',
                    object_type_to_per_vertex_color=bbox_per_vertex_color_map,
                    color_version=bbox_color_version,
                    line_width=4,
                    edge_color=[200, 200, 200],
                )
            )

    # ========== Final Rendering ==========
    # Render all collected geometry objects for the current frame
    combined_frame = render_geometries(
                geometry_objects,
                camera_model.height,
                camera_model.width,
                depth_max=200,
                depth_gradient=True,
            )
    # Output: combined_frame is a numpy array with shape [H, W, 3] representing the rendered frame (RGB format)
            
    # ========== Save Output ==========
    # Save the rendered frame to disk
    output_dir = Path('./render_test_cases_frames')
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_filename = output_dir / f"test_frame.png"
    cv2.imwrite(str(frame_filename), cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
    print(f"Rendered frame saved to: {frame_filename}")
            
if __name__ == "__main__":
    render_sample_hdmap_v3()

