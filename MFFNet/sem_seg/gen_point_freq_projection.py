import os
import sys

import numpy as np
import math as math_tool
import cv2

import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

# ATTENTION:
#       The global variable below is used to determine whether to use the threshold for aggregation or not.
#       If the users don't want to use the threshold to aggregate the contour, please set it False.
#       The aggregation described above is just used for the clear visualization.
#       Thus, the processed data after the aggregation definitely cannot be used as train or test data.
FLAG4THRESHOLD = False

# ATTENTION:
#       The global variable below is ued to determine whether to use the float data not the integer data.
#       Data with the type of float is more accurate than that with the type of integer.
FLAG4FLOAT = True


def load_data_files(data_filename):
    """Function that loads data file into memory

    :param data_filename: str, the object that represents the original data file name
    :return: data_source: ndarray, the object that stores the data
    """
    if data_filename[-3:] == 'txt':
        data_source = np.loadtxt(data_filename)
        return data_source
    elif data_filename[-3:] == 'npy':
        data_source = np.load(data_filename)
        return data_source
    else:
        print('Unknown file type: {}. Please check your input!'.format(data_filename))
        exit()


def round2int(value, boundary=-1):
    """Function that is used to obtain rounding value

    :param value: float32, an original value
    :param boundary: int, an object that represents the boundary of rounded value
    :return:
    """

    if value - int(value) >= 0.5:
        if boundary == -1 or (int(value)+1) <= boundary:
            return int(value)+1
        else:
            return boundary
    else:
        return int(value)


def normalize_xyz(xyz_data):
    """Function that is used to normalize the xyz data

    :param xyz_data:
    :return:
    """

    if np.max(xyz_data) != 0:
        xyz_data /= np.max(xyz_data)

    xyz_data -= np.min(xyz_data)
    return xyz_data


def point_freq_projection_plus_normalized(data_source, resolution, block_size=3, alpha=0.1, other_op='011'):
    """Function that is used to obtain the point frequency projection result

    :param data_source: np.ndarray, an object that stores original data
    :param resolution: int, an object that represents the resolution of the RGB image
    :param block_size: int, an object that represents the size of the block
    :param other_op: string, an object that tells what operation will be carried out for the following processing
    :return:
    """

    # Scale all points to the range (0,1)
    # Normalization
    data = data_source[:, 0:6]

    data[:, 0], data[:, 1], data[:, 2] = normalize_xyz(xyz_data=data[:, 0]),\
                                         normalize_xyz(xyz_data=data[:, 1]),\
                                         normalize_xyz(xyz_data=data[:, 2])

    # Obtain the different projection result
    point_freq_result_xy = generate_point_freq_2opposite_projection(data=data, resolution=resolution,
                                                                    plane_name='xy',
                                                                    block_size=block_size, other_op=other_op,
                                                                    alpha=alpha)
    point_freq_result_yz = generate_point_freq_2opposite_projection(data=data, resolution=resolution,
                                                                    plane_name='yz',
                                                                    block_size=block_size, other_op=other_op,
                                                                    alpha=alpha)
    point_freq_result_xz = generate_point_freq_2opposite_projection(data=data, resolution=resolution,
                                                                    plane_name='xz',
                                                                    block_size=block_size, other_op=other_op,
                                                                    alpha=alpha)

    return point_freq_result_xy.astype(np.uint8), point_freq_result_yz.astype(np.uint8), point_freq_result_xz.astype(np.uint8)


def apply_average_filter2remove_noise(image, resolution=128, kernel_size=3):
    """Funciton that is used to remove the white noise from the result image

    :param image: [[int]], an object that stores the RGB matrix
    :param resolution: int, an object that represents the resolution of the RGB image
    :param kernel_size: int, an object that is used in the filter
    :return:
    """

    # Assert that resolution equals to the shape of the matrix
    assert((resolution == image.shape[0]) and (resolution == image.shape[1]))

    # Generate the average filter
    kernel = np.ones((kernel_size, kernel_size), np.float32)/(kernel_size**2)
    kernel = kernel[::-1, ::-1]

    # Zero-padding
    padding_size4row, padding_size4col = int(kernel.shape[0]/2), int(kernel.shape[1]/2)
    padding_image = np.pad(image, ((padding_size4row, padding_size4row), (padding_size4col, padding_size4col)),
                           'constant', constant_values=(0, 0))

    # Calculate the convolution on the given point frequency image matrix
    kernel_flatten = kernel.flatten()

    # In every batch, we calculate the result by multiplying pixels by the filter and adding up the multiplications
    def conv2d_in_batch(centerx, centery):
        # Obtain the batch around the center
        image_batch = padding_image[centerx: centerx+kernel.shape[0], centery: centery+kernel.shape[1]]

        return np.dot(image_batch.flatten(), kernel_flatten)

    # Find white pixels and use the average filter
    for idx1 in range(image.shape[0]):
        for idx2 in range(image.shape[1]):
            # Aim only at the white pixels
            # if image[idx1, idx2] == 0:
            #     image[idx1, idx2] = int(conv2d_in_batch(idx1, idx2))
            image[idx1, idx2] = int(conv2d_in_batch(idx1, idx2))

    return image


def generate_point_freq_2opposite_projection(data, resolution, plane_name,
                                             block_size=3, alpha=0.05, other_op='011'):
    """Function that is used to generate the matrix of point frequency

    :param data: np.ndarray, an object that stores original data
    :param resolution: int, an object that represents the resolution of the RGB image
    :param plane_name: str, an object that represents the projection plane
    :param block_size: int, an object that determine the block for kNN
    :param alpha: float, an object that represents the proportion for threshold
    :param other_op: string, an object that tells what operation will be carried out for the following processing
    :return:
    """

    # Obtain the xyz data
    xyz_data = np.zeros((data.shape[0], 3))
    xyz_data[:, :] = data[:, 0:3]

    # Storage for data pre-processing
    distance2plane_data = np.zeros(data.shape[0])
    plane_data = np.zeros((data.shape[0], 2))

    if plane_name == 'xy':
        distance2plane_data[:] = xyz_data[:, 2]
        plane_data[:, 0], plane_data[:, 1] = xyz_data[:, 0], xyz_data[:, 1]
    elif plane_name == 'yz':
        distance2plane_data[:] = data[:, 0]
        plane_data[:, 0], plane_data[:, 1] = xyz_data[:, 1], xyz_data[:, 2]
    elif plane_name == 'xz':
        distance2plane_data[:] = data[:, 1]
        plane_data[:, 0], plane_data[:, 1] = xyz_data[:, 0], xyz_data[:, 2]
    else:
        print('Unknown plane name: {}. Please check the existing plane!'.format(plane_name))
        exit()

    # Allocate plane data into xy plane and distance data into z axis
    xyz_data[:, 0], xyz_data[:, 1], xyz_data[:, 2] = plane_data[:, 0], plane_data[:, 1], distance2plane_data[:]
    point_freq_projected_result = generate_point_freq_matrix(xyz_data=xyz_data, resolution=resolution, alpha=alpha)

    # Apply the average filter to the matrix
    if other_op[0] == '1':
        point_freq_projected_result = apply_average_filter2remove_noise(image=point_freq_projected_result,
                                                                        resolution=resolution,
                                                                        kernel_size=block_size)

    # Apply the histogram equalization to the matrix
    if other_op[1] == '1':
        point_freq_projected_result = cv2.equalizeHist(src=point_freq_projected_result)

    # Apply the Gaussian filter to the matrix
    if other_op[2] == '1':
        point_freq_projected_result = cv2.GaussianBlur(src=point_freq_projected_result,
                                                       ksize=(block_size, block_size),
                                                       sigmaX=0, sigmaY=0)

    # Apply the edge detection method to aggregating the image (use Sobel operator)
    if other_op[2] == '2':
        point_freq_resx= cv2.Sobel(point_freq_projected_result, cv2.CV_16S, 1, 0)
        point_freq_resy = cv2.Sobel(point_freq_projected_result, cv2.CV_16S, 0, 1)

        point_freq_resx = cv2.convertScaleAbs(point_freq_resx)
        point_freq_resy = cv2.convertScaleAbs(point_freq_resy)

        point_freq_projected_result = cv2.addWeighted(src1=point_freq_resx, alpha=0.5,
                                                      src2=point_freq_resy, beta=0.5,
                                                      gamma=0.5)

    # Apply the edge detection method to aggregating the image (use Canny operator)
    if other_op[2] == '3':
        point_freq_projected_result = cv2.Canny(image=point_freq_projected_result, threshold1=180, threshold2=240)

    # Adjust the point frequency image to the right angle
    if plane_name == 'yz':
        point_freq_projected_result = np.rot90(point_freq_projected_result, 1)
    if plane_name == 'xz':
        point_freq_projected_result = np.rot90(point_freq_projected_result, 1)

    return point_freq_projected_result


def generate_point_freq_matrix(xyz_data, resolution, alpha=0.15):
    """Function that is used to generate one projection matrix

    :param xyz_data: np.ndarray, an object that stores xyz information only
    :param resolution: int, an object that represents the size of the grid
    :param alpha: float32, an object that represents the scale proportion
    :return:
    """

    point_freq_projected_result = np.zeros((resolution, resolution))

    # Create a container to calculate the frequency and obtain the rgb at the same time
    locations = {}

    # Choose the nearest point and project those points into blocks
    for idx in range(xyz_data.shape[0]):
        # Shift 0.5/resolution to right and scale
        shifted_x = (xyz_data[idx, 0] + 0.5 / resolution) * resolution
        shifted_y = (xyz_data[idx, 1] + 0.5 / resolution) * resolution

        # Allocate the point to the specific block
        block_x = round2int(value=shifted_x, boundary=resolution)
        block_y = round2int(value=shifted_y, boundary=resolution)

        # Append to the list
        # Every element in the array is like: [idx, [x, y, z], dist2center]
        # Actually, dist2center is a placeholder for the following kNN calculating
        loc4pt = tuple([block_x - 1, block_y - 1])
        if loc4pt not in locations:
            locations[loc4pt] = 0
        locations[loc4pt] += 1

    # Assign the matrix with number of points in a given block
    for idx1 in range(resolution):
        for idx2 in range(resolution):
            if (idx1, idx2) in locations:
                point_freq_projected_result[idx1, idx2] = locations[(idx1, idx2)]
            else:
                point_freq_projected_result[idx1, idx2] = 0

    # Normalize the point frequency matrix,
    # Then aggregate the the value of every element in the matrix to the range of (0, 255)
    max_point_number = point_freq_projected_result.max()
    if max_point_number != 0:
        point_freq_projected_result /= max_point_number

    point_freq_projected_result *= 255.0

    '''Aggregation just for clear visualization'''
    if FLAG4THRESHOLD is True:
        print("Execute aggregating transformation for clear visualization...")
        # Set the threshold for every element in the matrix (just for visualization)
        threshold = generate_threshold4aggregate(method='median', matrix=point_freq_projected_result,
                                                 alpha=alpha)
        for idx1 in range(resolution):
            for idx2 in range(resolution):
                point_freq_projected_result[idx1, idx2] = round2int(point_freq_projected_result[idx1, idx2])
                # Aggregate just for clear visualization
                if point_freq_projected_result[idx1, idx2] > threshold:
                    point_freq_projected_result[idx1, idx2] = 255

    '''Save original data with the type of float32'''
    if FLAG4FLOAT is True:
        return point_freq_projected_result.astype(np.float32)

    # Obtain the integer version of the matrix
    for idx1 in range(resolution):
        for idx2 in range(resolution):
            point_freq_projected_result[idx1, idx2] = round2int(point_freq_projected_result[idx1, idx2])

    return 255-point_freq_projected_result.astype(np.uint8)


def generate_threshold4aggregate(method, matrix, alpha=0.05):
    """Function that is used to generate the threshold

    :param method:
    :param matrix:
    :param alpha:
    :return:
    """

    if method == 'median':
        return round2int(np.median(matrix))
    elif method == 'median_bias':
        diff_normalized = matrix.max() - matrix.min()
        threshold = round2int(np.median(matrix) + (diff_normalized*alpha))
        return threshold
    elif method == 'mean':
        return round2int(np.mean(matrix))
    elif method == 'mean_bias':
        diff_normalized = matrix.max() - matrix.min()
        threshold = round2int(np.mean(matrix) + (diff_normalized*alpha))
        return threshold
    else:
        diff_normalized = matrix.max() - matrix.min()
        threshold = round2int(np.min(matrix) + (diff_normalized*alpha))
        return threshold


def main():
    """Function that obtains the arguments and generates data

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./../data/stanford_indoor3d',
                        help='Data directory path to use [default: ./../data/standford_indoor3d]')
    parser.add_argument('--res_dir', type=str, default='./../data/indoor3d_sem_seg_pf_projection',
                        help='Result directory [default: ./../data/indoor3d_sem_seg_pf_projection]')
    parser.add_argument('--alpha', type=float, default=0.007,
                        help='Threshold involving in aggregating edges(just for clear visualization) [default: 0.007]')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution of the projection [default: 128]')
    parser.add_argument('--block_size', type=int, default=5,
                        help='The size of a block that is used to apply the filter to(must be odd) [default: 5]')
    parser.add_argument('--other_operation', type=str, default='011',
                        help='Some other operations to process the result. The length of the given string should be '
                             'exactly three. FIRST: 0->Not, 1->Do, for blurring. SECOND: 0->Not, 1->Do, for '
                             'histogram equalization. THIRD: 0->Not, 1->Gaussian, 2->Sobel, 3->Canny, for blurring'
                             '[default: 011]')

    config = parser.parse_args()

    # Process every file in the data directory
    data_list = os.listdir(config.data_dir)
    for data_file in data_list:
        data_source = load_data_files(config.data_dir + '/' + data_file)
        pf_result_xy, pf_result_yz, pf_result_xz = point_freq_projection_plus_normalized(data_source=data_source,
                                                                                         resolution=config.resolution,
                                                                                         block_size=config.block_size,
                                                                                         alpha=config.alpha,
                                                                                         other_op=config.other_operation)
        # File name for point frequency to be stored
        result_file_name = str(config.res_dir + '/' + data_file[:-4]) + '_'
        # Generate the point frequency matrix
        cv2.imwrite(filename=result_file_name+'xy_pf_bsize{}.jpg'.format(config.block_size), img=pf_result_xy)
        np.save(result_file_name+'xy_pf_bsize{}.npy'.format(config.block_size), pf_result_xy)
        cv2.imwrite(filename=result_file_name+'xz_pf_bsize{}.jpg'.format(config.block_size), img=pf_result_xz)
        np.save(result_file_name + 'xz_pf_bsize{}.npy'.format(config.block_size), pf_result_xy)
        cv2.imwrite(filename=result_file_name+'yz_pf_bsize{}.jpg'.format(config.block_size), img=pf_result_yz)
        np.save(result_file_name + 'yz_pf_bsize{}.npy'.format(config.block_size), pf_result_xy)

        print('FINISH: ' + data_file[:-4])


if __name__ == '__main__':
    main()
