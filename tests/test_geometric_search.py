# The MIT License (MIT)
#
# Copyright (c) 2018-2025 BeamMe Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""This script is used to test the functionality of the geometric_search
module."""

import random

import numpy as np
import pytest

from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.four_c.element_beam import Beam3rHerm2Line3
from beamme.geometric_search.find_close_points import (
    FindClosePointAlgorithm,
    find_close_points,
    partner_indices_to_point_partners,
    point_partners_to_partner_indices,
    point_partners_to_unique_indices,
)
from beamme.mesh_creation_functions.applications.beam_honeycomb import (
    create_beam_mesh_honeycomb_flat,
)
from beamme.utils.nodes import filter_nodes, get_nodal_coordinates

PYTEST_GEOMETRIC_SEARCH_PARAMETRIZE = [
    "algorithm",
    [
        FindClosePointAlgorithm.kd_tree_scipy,
        FindClosePointAlgorithm.brute_force_cython,
        pytest.param(
            FindClosePointAlgorithm.boundary_volume_hierarchy_arborx,
            marks=pytest.mark.arborx,
        ),
    ],
]


def assert_unique_id_coordinates(
    coords,
    point_partners,
    algorithm,
    tol,
    unique_indices_ref,
    inverse_indices_ref,
    assert_results_close,
):
    """Test if the unique coordinates are really unique and that the inverse
    indices result in the original array."""

    # Get the array with the unique indices.
    unique_indices, inverse_indices = point_partners_to_unique_indices(point_partners)
    unique_points = coords[unique_indices]

    # In the unique indices there should be no partners.
    _, unique_n_partners = find_close_points(
        unique_points, algorithm=algorithm, tol=tol
    )
    assert unique_n_partners == 0

    # Check that the inverse IDs result in the original array.
    reconstructed_coords = unique_points[inverse_indices]
    assert_results_close(coords, reconstructed_coords, rtol=0.0, atol=tol)

    # Check the IDs
    assert unique_indices == unique_indices_ref
    assert inverse_indices == inverse_indices_ref


@pytest.mark.parametrize(*PYTEST_GEOMETRIC_SEARCH_PARAMETRIZE)
def test_find_close_points_between_bins(algorithm, assert_results_close):
    """Test if the find_close_points function returns the expected results.

    The points are chosen such that for n_bins = [4, 4, 4], some points
    are exactly at the boundary between bins. This test case can be used
    for all algorithms, not just binning.

    This test was used to test the binning implementation, we still keep
    it for the other algorithms.
    """

    # Set the seed for the pseudo random numbers
    random.seed(0)

    # Add random nodes to a cube with width 2. Randomly add nodes close to
    # existing nodes. The distance has to be shorter than 0.5 * eps_medium
    # since the algorithm needs all close nodes to be within a sphere with
    # radius eps_medium.
    eps_medium = 1e-5
    eps_medium_factor = 0.49 * eps_medium
    n_nodes_random = 999
    n_nodes_total = n_nodes_random + 2
    coords = np.zeros([n_nodes_total, 3])
    for i in range(n_nodes_random):
        # Check if this one should be close to another one.
        if random.randint(0, 4) == 0 and i > 0:
            close_node = random.randint(0, i - 1)
            for j in range(3):
                coords[i, j] = coords[close_node, j]
        else:
            for j in range(3):
                coords[i, j] = random.uniform(-1, 1)

    # Create a random vector for each node. The length of the random
    # vectors is scaled, so that is is a maximum of 1.
    diff = np.random.rand(n_nodes_random, 3)
    diff /= np.linalg.norm(diff, axis=1)[:, None]
    coords[:n_nodes_random] += eps_medium_factor * diff

    # Add nodes such that the bin will be exactly [-1, 1] x [-1, 1] x [-1, 1]
    coords[-1, :] = [1.0, 1.0, 1.0]
    coords[-2, :] = [-1.0, -1.0, -1.0]

    # Add nodes between the bins
    coords[-3, :] = [0.0, 0.0, 0.0]
    coords[-4, :] = [0.0, 0.0, 0.0]
    coords[-5, :] = [0.0, 0.0, 0.5]
    coords[-6, :] = [0.0, 0.0, 0.5]
    coords[-7, :] = [0.0, 0.5, 0.0]
    coords[-8, :] = [0.0, 0.5, 0.0]
    coords[-9, :] = [0.5, 0.0, 0.0]
    coords[-10, :] = [0.5, 0.0, 0.0]
    coords[-11, :] = [0.0 + eps_medium_factor, 0.0, 0.0]
    coords[-12, :] = [0.0, 0.0 + eps_medium_factor, 0.0]
    coords[-13, :] = [0.0, 0.0, 0.0 + eps_medium_factor]
    coords[-14, :] = [0.0 - eps_medium_factor, 0.0, 0.0]
    coords[-15, :] = [0.0, 0.0 - eps_medium_factor, 0.0]
    coords[-16, :] = [0.0, 0.0, 0.0 - eps_medium_factor]

    # Test the number of partners and list of partners
    point_partners, n_partners = find_close_points(
        coords, algorithm=algorithm, tol=eps_medium
    )
    partner_indices = point_partners_to_partner_indices(point_partners, n_partners)

    # Compare to the reference solution.
    point_partners_reference = -1 * np.ones([n_nodes_total], dtype=int)
    # fmt: off
    index_vector = [0, 1, 3, 6, 8, 9, 10, 11, 12, 13, 16, 19, 20, 21, 22,
        25, 26, 29, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41, 43, 44, 48, 49,
        50, 51, 55, 59, 61, 62, 63, 65, 66, 67, 68, 69, 72, 73, 74, 77, 78,
        79, 80, 82, 83, 84, 85, 86, 88, 89, 92, 94, 95, 96, 98, 103, 104,
        106, 110, 111, 115, 119, 120, 121, 123, 125, 127, 128, 130, 132,
        133, 134, 136, 137, 138, 139, 140, 141, 143, 144, 146, 147, 149,
        151, 154, 156, 157, 160, 165, 168, 170, 172, 175, 176, 179, 180,
        181, 183, 185, 188, 189, 190, 193, 196, 197, 198, 202, 203, 206,
        209, 211, 212, 214, 215, 219, 221, 222, 225, 229, 230, 232, 239,
        241, 242, 245, 247, 248, 250, 253, 254, 255, 256, 260, 261, 265,
        270, 273, 276, 278, 281, 288, 289, 295, 301, 302, 305, 308, 312,
        313, 315, 318, 322, 328, 337, 343, 346, 347, 352, 354, 356, 359,
        361, 363, 365, 367, 368, 369, 373, 376, 377, 379, 381, 384, 391,
        392, 393, 397, 398, 400, 401, 405, 417, 419, 421, 424, 429, 431,
        435, 437, 440, 441, 442, 444, 446, 456, 459, 462, 469, 470, 471,
        473, 474, 475, 476, 480, 484, 486, 490, 498, 501, 506, 513, 518,
        522, 523, 524, 525, 526, 536, 539, 541, 544, 545, 548, 560, 567,
        570, 571, 572, 576, 577, 596, 598, 600, 601, 603, 606, 611, 615,
        616, 621, 622, 625, 632, 633, 637, 639, 643, 646, 647, 649, 650,
        653, 672, 675, 676, 678, 679, 680, 682, 691, 693, 695, 700, 701,
        715, 719, 721, 725, 730, 736, 738, 742, 746, 749, 750, 756, 759,
        761, 762, 767, 769, 777, 778, 781, 783, 785, 787, 789, 792, 795,
        806, 811, 817, 826, 827, 830, 833, 834, 835, 839, 840, 841, 842,
        847, 850, 853, 854, 857, 859, 861, 864, 867, 869, 872, 875, 887,
        890, 895, 902, 904, 905, 912, 913, 914, 915, 926, 928, 932, 934,
        935, 944, 956, 958, 965, 979, 985, 986, 987, 988, 989, 990, 991,
        992, 993, 994, 995, 996, 997, 998]
    val_vector = [0, 1, 2, 3, 4, 5, 5, 6, 2, 7, 8, 9, 9, 10, 11, 12, 13,
        11, 14, 2, 15, 16, 17, 18, 19, 20, 15, 21, 22, 22, 23, 11, 10, 16,
        0, 5, 24, 2, 25, 8, 16, 26, 26, 27, 15, 28, 9, 25, 29, 30, 31, 32,
        33, 20, 34, 35, 36, 19, 5, 37, 38, 39, 26, 40, 41, 42, 16, 4, 43,
        44, 37, 45, 46, 24, 35, 47, 48, 49, 44, 50, 51, 49, 52, 53, 5, 54,
        55, 18, 42, 12, 56, 57, 1, 58, 59, 51, 60, 39, 61, 62, 63, 64, 65,
        27, 0, 66, 67, 68, 69, 70, 71, 50, 63, 63, 72, 73, 74, 75, 76, 77,
        6, 21, 73, 44, 23, 78, 24, 76, 75, 22, 69, 79, 80, 54, 81, 16, 82,
        59, 50, 83, 84, 85, 3, 75, 86, 35, 87, 66, 53, 48, 13, 88, 31, 73,
        85, 67, 89, 90, 39, 47, 17, 21, 26, 87, 3, 62, 91, 78, 75, 7, 60,
        92, 61, 93, 94, 54, 95, 96, 60, 97, 98, 86, 28, 99, 68, 100, 101,
        56, 100, 102, 67, 103, 104, 80, 105, 57, 103, 79, 106, 46, 82, 58,
        107, 69, 50, 108, 90, 109, 47, 92, 53, 91, 84, 110, 111, 30, 112,
        106, 98, 113, 114, 82, 115, 45, 116, 62, 117, 34, 35, 118, 32, 4,
        39, 109, 32, 33, 81, 115, 107, 10, 119, 64, 118, 120, 52, 108, 119,
        70, 102, 121, 23, 117, 5, 97, 122, 123, 124, 99, 111, 95, 125, 83,
        126, 127, 123, 114, 126, 107, 128, 117, 129, 130, 65, 116, 75, 72,
        131, 93, 132, 40, 129, 133, 131, 18, 134, 95, 124, 135, 77, 58,
        105, 128, 38, 136, 21, 41, 130, 137, 120, 132, 36, 138, 127, 110,
        136, 22, 96, 88, 29, 121, 74, 112, 133, 58, 42, 139, 64, 113, 117,
        125, 135, 137, 63, 14, 59, 83, 138, 140, 141, 94, 21, 140, 122, 89,
        43, 71, 101, 1, 139, 55, 134, 63, 104, 141, 142, 142, 142, 142,
        142, 142, 143, 143, 144, 144, 145, 145, 142, 142]
    unique_indices_ref = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16,
        17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36,
        37, 38, 39, 41, 42, 43, 45, 46, 47, 48, 52, 53, 54, 56, 57, 58, 60,
        61, 63, 64, 67, 69, 70, 71, 73, 75, 76, 78, 79, 80, 81, 82, 83, 85,
        86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104,
        105, 106, 107, 108, 109, 112, 113, 114, 115, 116, 117, 118, 119, 121,
        122, 123, 124, 126, 128, 129, 130, 131, 132, 134, 135, 136, 138, 139,
        141, 142, 143, 145, 148, 149, 150, 151, 152, 153, 155, 156, 157, 158,
        159, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174,
        175, 176, 177, 178, 179, 182, 183, 184, 185, 186, 187, 188, 189, 190,
        191, 192, 193, 194, 195, 199, 200, 201, 202, 203, 204, 205, 206, 207,
        208, 209, 210, 211, 212, 213, 216, 217, 218, 220, 223, 224, 225, 226,
        227, 228, 231, 233, 234, 235, 236, 237, 238, 240, 242, 243, 244, 245,
        246, 248, 249, 251, 252, 253, 256, 257, 258, 259, 260, 261, 262, 263,
        264, 266, 267, 268, 269, 271, 272, 273, 274, 275, 277, 278, 279, 280,
        282, 283, 284, 285, 286, 287, 290, 291, 292, 293, 294, 296, 297, 298,
        299, 300, 301, 303, 304, 306, 307, 309, 310, 311, 313, 314, 315, 316,
        317, 319, 320, 321, 323, 324, 325, 326, 327, 329, 330, 331, 332, 333,
        334, 335, 336, 338, 339, 340, 341, 342, 344, 345, 348, 349, 350, 351,
        353, 354, 355, 357, 358, 360, 362, 364, 365, 366, 368, 369, 370, 371,
        372, 374, 375, 376, 377, 378, 380, 381, 382, 383, 384, 385, 386, 387,
        388, 389, 390, 393, 394, 395, 396, 398, 399, 400, 402, 403, 404, 406,
        407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 420, 421,
        422, 423, 424, 425, 426, 427, 428, 430, 431, 432, 433, 434, 436, 438,
        439, 441, 443, 445, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456,
        457, 458, 460, 461, 463, 464, 465, 466, 467, 468, 469, 471, 472, 477,
        478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489, 491, 492, 493,
        494, 495, 496, 497, 498, 499, 500, 502, 503, 504, 505, 507, 508, 509,
        510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 523, 525,
        527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 540, 542,
        543, 544, 546, 547, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558,
        559, 561, 562, 563, 564, 565, 566, 568, 569, 573, 574, 575, 578, 579,
        580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593,
        594, 595, 597, 598, 599, 602, 603, 604, 605, 607, 608, 609, 610, 612,
        613, 614, 617, 618, 619, 620, 622, 623, 624, 626, 627, 628, 629, 630,
        631, 634, 635, 636, 638, 639, 640, 641, 642, 643, 644, 645, 646, 648,
        651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664,
        665, 666, 667, 668, 669, 670, 671, 673, 674, 675, 676, 677, 681, 683,
        684, 685, 686, 687, 688, 689, 690, 691, 692, 694, 695, 696, 697, 698,
        699, 700, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713,
        714, 716, 717, 718, 720, 722, 723, 724, 725, 726, 727, 728, 729, 731,
        732, 733, 734, 735, 736, 737, 739, 740, 741, 743, 744, 745, 746, 747,
        748, 751, 752, 753, 754, 755, 756, 757, 758, 760, 762, 763, 764, 765,
        766, 768, 770, 771, 772, 773, 774, 775, 776, 779, 780, 782, 783, 784,
        786, 788, 790, 791, 792, 793, 794, 796, 797, 798, 799, 800, 801, 802,
        803, 804, 805, 807, 808, 809, 810, 812, 813, 814, 815, 816, 817, 818,
        819, 820, 821, 822, 823, 824, 825, 828, 829, 831, 832, 836, 837, 838,
        843, 844, 845, 846, 848, 849, 851, 852, 854, 855, 856, 858, 860, 862,
        863, 865, 866, 868, 870, 871, 873, 874, 876, 877, 878, 879, 880, 881,
        882, 883, 884, 885, 886, 888, 889, 891, 892, 893, 894, 896, 897, 898,
        899, 900, 901, 902, 903, 904, 906, 907, 908, 909, 910, 911, 916, 917,
        918, 919, 920, 921, 922, 923, 924, 925, 927, 929, 930, 931, 933, 936,
        937, 938, 939, 940, 941, 942, 943, 945, 946, 947, 948, 949, 950, 951,
        952, 953, 954, 955, 957, 959, 960, 961, 962, 963, 964, 966, 967, 968,
        969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 980, 981, 982, 983,
        984, 985, 991, 993, 995, 999, 1000]
    inverse_indices_ref = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 3, 11, 12, 13,
        14, 15, 16, 17, 17, 18, 19, 20, 21, 22, 23, 24, 25, 19, 26, 3, 27, 28,
        29, 30, 31, 32, 33, 34, 27, 35, 36, 37, 37, 38, 39, 40, 41, 19, 18, 29,
        42, 43, 44, 0, 45, 46, 47, 9, 48, 49, 3, 50, 51, 14, 29, 52, 52, 53, 54,
        55, 27, 56, 17, 57, 58, 50, 59, 60, 61, 62, 63, 64, 34, 65, 66, 67, 68,
        33, 69, 70, 9, 71, 72, 73, 74, 75, 52, 76, 77, 78, 79, 80, 81, 82, 83,
        84, 85, 86, 29, 8, 87, 88, 89, 90, 91, 92, 93, 94, 72, 95, 96, 97, 98,
        49, 99, 66, 100, 101, 102, 103, 104, 94, 105, 106, 107, 104, 108, 109,
        9, 110, 111, 112, 32, 113, 83, 22, 114, 115, 116, 117, 118, 119, 1, 120,
        121, 122, 123, 124, 107, 125, 126, 127, 128, 129, 130, 131, 74, 132, 133,
        134, 135, 136, 137, 138, 139, 140, 141, 142, 53, 0, 143, 144, 145, 146,
        147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 105, 138, 138, 157, 158,
        159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 10, 35,
        172, 173, 174, 161, 175, 94, 41, 176, 177, 178, 179, 180, 181, 49, 169,
        182, 167, 183, 184, 185, 186, 187, 188, 37, 189, 150, 190, 191, 192, 193,
        194, 110, 195, 196, 29, 197, 198, 199, 122, 105, 200, 201, 202, 203, 204,
        205, 206, 207, 208, 6, 209, 210, 211, 212, 167, 213, 214, 215, 216, 217,
        66, 218, 219, 220, 221, 144, 222, 223, 224, 225, 226, 227, 109, 102, 228,
        229, 230, 231, 232, 23, 233, 234, 235, 236, 237, 238, 61, 239, 240, 161,
        241, 242, 205, 243, 244, 245, 146, 246, 247, 248, 249, 250, 74, 251, 252,
        253, 100, 254, 255, 256, 257, 258, 30, 259, 260, 261, 262, 263, 264, 265,
        266, 35, 267, 268, 269, 270, 271, 52, 272, 273, 219, 6, 274, 275, 276, 277,
        135, 278, 279, 280, 178, 281, 282, 167, 283, 11, 284, 129, 285, 286, 287,
        133, 288, 289, 290, 291, 292, 110, 293, 294, 295, 296, 297, 129, 298, 299,
        300, 301, 302, 303, 304, 305, 306, 307, 308, 215, 56, 309, 310, 311, 312,
        149, 313, 314, 315, 115, 316, 317, 318, 313, 319, 320, 321, 322, 323, 324,
        325, 326, 327, 328, 329, 330, 331, 146, 332, 333, 334, 335, 336, 337, 338,
        339, 340, 193, 341, 342, 343, 344, 345, 117, 346, 333, 347, 348, 190, 349,
        97, 350, 199, 351, 121, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361,
        362, 363, 150, 364, 365, 105, 366, 367, 368, 369, 370, 371, 372, 248, 373,
        374, 100, 286, 109, 279, 375, 376, 377, 204, 378, 379, 380, 381, 382, 383,
        384, 385, 386, 60, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 349,
        397, 398, 399, 400, 302, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410,
        411, 412, 413, 414, 415, 199, 416, 95, 417, 135, 418, 419, 420, 421, 422,
        423, 424, 425, 426, 427, 428, 429, 65, 430, 66, 431, 432, 433, 63, 434,
        435, 8, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 74, 447,
        448, 449, 450, 451, 452, 373, 453, 454, 63, 64, 195, 455, 456, 457, 416,
        361, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471,
        472, 473, 474, 475, 18, 476, 477, 478, 139, 433, 479, 480, 481, 482, 108,
        483, 484, 485, 486, 372, 487, 488, 489, 477, 151, 490, 491, 492, 493, 330,
        494, 495, 496, 41, 497, 498, 499, 500, 501, 502, 427, 9, 503, 504, 505,
        299, 506, 507, 508, 509, 510, 511, 512, 513, 514, 309, 515, 383, 295, 516,
        517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531,
        532, 533, 534, 535, 536, 200, 537, 538, 539, 540, 541, 511, 412, 539, 542,
        361, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 427, 553, 554, 555,
        556, 557, 558, 559, 142, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569,
        570, 571, 572, 417, 573, 574, 575, 167, 576, 160, 577, 578, 579, 580, 581,
        582, 583, 584, 288, 585, 586, 587, 588, 589, 590, 591, 80, 592, 593, 594,
        554, 595, 596, 597, 598, 599, 600, 580, 32, 601, 602, 603, 604, 605, 606,
        607, 608, 295, 609, 514, 610, 611, 612, 613, 614, 170, 615, 121, 616, 617,
        618, 619, 620, 621, 622, 342, 551, 623, 624, 73, 625, 626, 627, 35, 628,
        81, 629, 559, 630, 631, 632, 633, 634, 480, 635, 636, 637, 638, 639, 640,
        641, 642, 643, 644, 590, 645, 646, 647, 648, 68, 649, 650, 651, 652, 653,
        654, 655, 656, 657, 658, 659, 660, 661, 662, 540, 381, 663, 664, 626, 665,
        666, 37, 296, 238, 667, 668, 669, 59, 494, 164, 394, 670, 671, 672, 673,
        598, 674, 675, 121, 676, 677, 83, 678, 679, 680, 139, 681, 407, 682, 427,
        683, 684, 518, 685, 686, 610, 687, 632, 688, 689, 138, 690, 691, 26, 692,
        693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 122, 703, 704, 200, 705,
        706, 707, 708, 654, 709, 710, 711, 712, 713, 714, 715, 716, 717, 289, 718,
        719, 720, 721, 722, 723, 35, 715, 507, 246, 724, 725, 726, 727, 728, 729,
        730, 731, 732, 733, 90, 734, 154, 735, 736, 737, 315, 738, 1, 678, 739,
        740, 741, 742, 743, 744, 745, 746, 112, 747, 748, 749, 750, 751, 752, 753,
        754, 755, 756, 757, 606, 758, 138, 759, 760, 761, 762, 763, 764, 336, 765,
        766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 717, 778, 779,
        780, 781, 782, 783, 783, 783, 783, 783, 783, 784, 784, 785, 785, 786, 786,
        783, 783, 787, 788]
    # fmt: on
    point_partners_reference[index_vector] = val_vector
    assert len(partner_indices) == 146
    assert_results_close(point_partners, point_partners_reference)

    # Test unique IDs
    assert_unique_id_coordinates(
        coords,
        point_partners,
        algorithm,
        eps_medium,
        unique_indices_ref,
        inverse_indices_ref,
        assert_results_close,
    )


@pytest.mark.parametrize(*PYTEST_GEOMETRIC_SEARCH_PARAMETRIZE)
def test_find_close_points_binning_flat(
    algorithm, assert_results_close, get_default_test_beam_material
):
    """Test case for coupling of points, when the nodes are all on a plane.

    This is challenging for a binning based approach. However, this test
    case can also be used for all find_close_point algorithms.
    """

    # Dummy material.
    material = get_default_test_beam_material(material_type="reissner")

    def create_flat_mesh():
        """Create a flat honeycomb mesh."""
        mesh = Mesh()
        create_beam_mesh_honeycomb_flat(
            mesh, Beam3rHerm2Line3, material, 1, 5, 5, create_couplings=False
        )
        return mesh

    # Get a reference solution for the coupling nodes.
    # fmt: off
    reference_partners_list = [0, 1, 1, 2, 0, 3, 2, 4, 4, 5, 2, 6, 5, 7, 7,
        8, 5, 9, 8, 10, 10, 11, 8, 12, 11, 13, 13, 14, 11, 15, 14, 16, 3,
        17, 17, 6, 17, 18, 6, 19, 19, 9, 19, 20, 9, 21, 21, 12, 21, 22, 12,
        23, 23, 15, 23, 24, 15, 25, 25, 16, 25, 26, 27, 18, 18, 28, 27, 29,
        28, 20, 20, 30, 28, 31, 30, 22, 22, 32, 30, 33, 32, 24, 24, 34, 32,
        35, 34, 26, 26, 36, 34, 37, 36, 38, 29, 39, 39, 31, 39, 40, 31, 41,
        41, 33, 41, 42, 33, 43, 43, 35, 43, 44, 35, 45, 45, 37, 45, 46, 37,
        47, 47, 38, 47, 48, 49, 40, 40, 50, 49, 51, 50, 42, 42, 52, 50, 53,
        52, 44, 44, 54, 52, 55, 54, 46, 46, 56, 54, 57, 56, 48, 48, 58, 56,
        59, 58, 60, 51, 61, 61, 53, 53, 62, 62, 55, 55, 63, 63, 57, 57, 64,
        64, 59, 59, 65, 65, 60]
    unique_indices_ref = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25,
        27, 29, 31, 33, 37, 39, 43, 45, 49, 51, 55, 57, 61, 62, 65, 67, 71,
        73, 77, 79, 83, 85, 89, 91, 93, 95, 99, 101, 105, 107, 111, 113, 117,
        119, 123, 124, 127, 129, 133, 135, 139, 141, 145, 147, 151, 153, 155,
        157, 161, 165, 169, 173]
    inverse_indices_ref = [0, 1, 1, 2, 0, 3, 2, 4, 4, 5, 2, 6, 5, 7, 7, 8, 5,
        9, 8, 10, 10, 11, 8, 12, 11, 13, 13, 14, 11, 15, 14, 16, 3, 17, 17,
        6, 17, 18, 6, 19, 19, 9, 19, 20, 9, 21, 21, 12, 21, 22, 12, 23, 23,
        15, 23, 24, 15, 25, 25, 16, 25, 26, 27, 18, 18, 28, 27, 29, 28, 20,
        20, 30, 28, 31, 30, 22, 22, 32, 30, 33, 32, 24, 24, 34, 32, 35, 34,
        26, 26, 36, 34, 37, 36, 38, 29, 39, 39, 31, 39, 40, 31, 41, 41, 33,
        41, 42, 33, 43, 43, 35, 43, 44, 35, 45, 45, 37, 45, 46, 37, 47, 47,
        38, 47, 48, 49, 40, 40, 50, 49, 51, 50, 42, 42, 52, 50, 53, 52, 44,
        44, 54, 52, 55, 54, 46, 46, 56, 54, 57, 56, 48, 48, 58, 56, 59, 58,
        60, 51, 61, 61, 53, 53, 62, 62, 55, 55, 63, 63, 57, 57, 64, 64, 59,
        59, 65, 65, 60]
    # fmt: on

    reference_partners = point_partners_to_partner_indices(reference_partners_list, 66)

    # Apply different rotations and compare the partner results.
    rotations = [
        Rotation([1, 0, 0], 0.0),
        Rotation([1, 0, 0], np.pi * 0.5),
        Rotation([0, 1, 0], np.pi * 0.5),
        Rotation([0, 0, 1], np.pi * 0.5),
        Rotation([1, 3, -4], 25.21561 * np.pi * 0.5),
    ]
    for rotation in rotations:
        # Create and rotate the mesh
        mesh = create_flat_mesh()
        mesh.rotate(rotation)

        # The reference data was created for the nodes without the middle
        # nodes, therefore we filter the middle nodes here
        coords = get_nodal_coordinates(filter_nodes(mesh.nodes, middle_nodes=False))
        partners = find_close_points(coords, algorithm=algorithm)

        has_partners, n_partner = find_close_points(
            get_nodal_coordinates(
                filter_nodes(create_flat_mesh().nodes, middle_nodes=False)
            ),
            algorithm=algorithm,
        )

        # Apply the result conversion, so we check this functionality too.
        n_points = len(has_partners)
        partners = point_partners_to_partner_indices(has_partners, n_partner)
        has_partners, n_partner = partner_indices_to_point_partners(partners, n_points)

        # Compare results with the reference.
        assert partners == reference_partners
        assert list(has_partners) == reference_partners_list
        assert n_partner == 66

        # Test unique IDs
        assert_unique_id_coordinates(
            coords,
            has_partners,
            algorithm,
            1e-5,
            unique_indices_ref,
            inverse_indices_ref,
            assert_results_close,
        )


@pytest.mark.parametrize(*PYTEST_GEOMETRIC_SEARCH_PARAMETRIZE)
def test_find_close_points_single_dimension(algorithm, assert_results_close):
    """Test that the find_close_points function works properly with a 1D
    dimensional array (internally a n x 1 array is required)"""

    # Create array with coordinates
    eps = 1e-10
    coords = np.array(
        [[0, 1, 2, 3 + eps, 4, 5, 3, 6, 7 + eps, 2, 8, 7 - eps, 9, 10, 10 + eps, 7]]
    ).transpose()

    # Expected results
    has_partner_expected = [-1, -1, 0, 1, -1, -1, 1, -1, 2, 0, -1, 2, -1, 3, 3, 2]
    partner_expected = 4

    # Get results
    has_partner, partner = find_close_points(coords, algorithm=algorithm, tol=10 * eps)

    # Check the results
    assert_results_close(has_partner_expected, has_partner)
    assert partner_expected == partner

    # Test unique IDs
    unique_indices_ref = [0, 1, 2, 3, 4, 5, 7, 8, 10, 12, 13]
    inverse_indices_ref = [0, 1, 2, 3, 4, 5, 3, 6, 7, 2, 8, 7, 9, 10, 10, 7]
    assert_unique_id_coordinates(
        coords,
        has_partner,
        algorithm,
        10 * eps,
        unique_indices_ref,
        inverse_indices_ref,
        assert_results_close,
    )


@pytest.mark.parametrize(*PYTEST_GEOMETRIC_SEARCH_PARAMETRIZE)
def test_find_close_points_multi_dimension(algorithm, assert_results_close):
    """Test that the find_close_points function also works properly with
    multidimensional points."""

    # Create array with coordinates.
    coords = np.array(
        [
            [0.0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [-1, 1, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 0],
            [-1, -1, 1, 0, 0, 0],
            [1, 1, -1, 0, 0, 0],
            [-1, 1, -1, 0, 0, 0],
            [1, -1, -1, 0, 0, 0],
            [-1, -1, -1, 0, 0, 0],
            [1, 1, 1, 0, 0, 1],
            [-1, 1, 1, 0, 0, 1],
            [1, -1, 1, 0, 0, 1],
            [-1, -1, 1, 0, 0, 0],
            [1, 1, -1, 0, 0, 0],
            [-1, 1, -1, 0, 0, 0],
            [1, -1, -1, 0, 0, 0],
            [-1, -1, -1, 0, 0, 0],
        ]
    )

    # Expected results
    # fmt: off
    has_partner_expected = [-1, -1, -1, -1, 0, 1, 2, 3, 4, -1, -1, -1, 0,
        1, 2, 3, 4]
    # fmt: on
    partner_expected = 5

    # Get results
    has_partner, partner = find_close_points(coords, algorithm=algorithm)

    # Check the results
    assert_results_close(has_partner_expected, has_partner)
    assert partner_expected == partner

    # Test unique IDs
    # fmt: off
    unique_indices_ref = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    inverse_indices_ref = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8]
    # fmt: on
    assert_unique_id_coordinates(
        coords,
        has_partner,
        algorithm,
        1e-5,
        unique_indices_ref,
        inverse_indices_ref,
        assert_results_close,
    )


@pytest.mark.parametrize(*PYTEST_GEOMETRIC_SEARCH_PARAMETRIZE)
def test_find_close_points_tolerance_precision(algorithm, assert_results_close):
    """Test that the find_close_points tolerance works with a precision of at
    least 12."""

    n_points = 4
    delta = 1e-12
    coords = np.ones([n_points, 3])
    coords[1, 0] += delta
    coords[2, 0] += 4 * delta
    coords[3, 0] += 5 * delta

    has_partner, _ = find_close_points(coords, algorithm=algorithm, tol=1.1 * delta)
    assert_results_close(has_partner, [0, 0, 1, 1])


@pytest.mark.performance
def test_performance_find_close_points_brute_force_cython(
    evaluate_execution_time,
):
    """Test the performance of finding close points using brute force Cython
    algorithm."""

    def repeat_find_random_close_points(n_points, n_runs, algorithm):
        """Repeat finding close points with random points."""
        np.random.seed(seed=1)
        points = np.random.rand(n_points, 3)

        for _ in range(n_runs):
            find_close_points(points, algorithm=algorithm)

    evaluate_execution_time(
        "BeamMe: Find close points (brute force Cython)",
        repeat_find_random_close_points,
        kwargs={
            "n_points": 100,
            "n_runs": 1000,
            "algorithm": FindClosePointAlgorithm.brute_force_cython,
        },
        expected_time=0.025,
    )
