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
    # Skip the data file that has incorrect format and exit the program
    else:
        print('Unknown file type: {}. Please check your input!'.format(data_filename))
        exit()


def partition(array, low, high):
    """Function that gives out the pivot index

    :param array: list, an object that stores the indices
    :param low: int, an object that represents the lower boundary
    :param high: int, an object that represents the upper boundary
    :return:
    """

    pivot = array[high][2]
    pivot_index = low
    for idx in range(low, high):
        if array[idx][2] < pivot:
            array[idx], array[pivot_index] = array[pivot_index], array[idx]
            pivot_index += 1
    array[pivot_index], array[high] = array[high], array[pivot_index]
    return pivot_index


def quicksort(array, low, high):
    """Function that sorts the points in the given block

    :param array: list, an object that stores the indices
    :param low: int, an object that represents the lower boundary
    :param high: int, an object that represents the upper boundary
    :return:
    """

    if low < high:
        pivot_index = partition(array, low, high)
        quicksort(array=array, low=low, high=pivot_index-1)
        quicksort(array=array, low=pivot_index+1, high=high)


def normalize_xyz_warping(xyz_data):
    """Function that is used to normalize the xyz data, which is warped

    :param xyz_data:
    :return:
    """

    if np.min(xyz_data) < 0:
        xyz_data -= np.min(xyz_data)

    if np.max(xyz_data) != 0:
        xyz_data /= np.max(xyz_data)

    xyz_data -= np.min(xyz_data)
    return xyz_data


def normalize_xyz(xyz_data, area_size=1.0):
    """Function that is used to normalize the xyz data

    :param xyz_data:
    :param area_size:
    :return:
    """

    # Size of the given block of the point cloud should be larger than zero
    assert(area_size > 0)

    if np.min(xyz_data) < 0:
        xyz_data -= np.min(xyz_data)

    xyz_data /= area_size
    xyz_data -= np.min(xyz_data)

    return xyz_data


def rgb_projection_plus_warped_normalized(data_source, resolution, radius=0.5,
                                          kernel_size=3, sample_num=10, block_size=3):
    """Function that projects the data to different planes

    :param data_source: np.ndarray, an object that stores original data
    :param resolution: int, an object that represents the resolution of the RGB image
    :param sample_num: int, an object that represents the number of points in our consideration
    :param block_size: int, an object that represents the size of the block
    :param radius: int, an object that represents the bias for the coordinates to deal with the transformation
    :param kernel_size: int, an object that represents the size of the kernel used in filtering
    :return:
    """

    # Scale all points to the range (0,1)
    # Normalization
    data = data_source[:, 0:6]

    data[:, 0], data[:, 1], data[:, 2] = normalize_xyz_warping(xyz_data=data[:, 0]), \
                                         normalize_xyz_warping(xyz_data=data[:, 1]), \
                                         normalize_xyz_warping(xyz_data=data[:, 2])

    # Generates RGB data in different projection
    rgb_result_xy = generate_rgb_projection_data(data=data,
                                                 resolution=resolution, plane_name='xy',
                                                 sample_num=sample_num, block_size=block_size,
                                                 kernel_size=kernel_size, radius=radius)
    rgb_result_xz = generate_rgb_projection_data(data=data,
                                                 resolution=resolution, plane_name='xz',
                                                 sample_num=sample_num, block_size=block_size,
                                                 kernel_size=kernel_size, radius=radius)
    rgb_result_yz = generate_rgb_projection_data(data=data,
                                                 resolution=resolution, plane_name='yz',
                                                 sample_num=sample_num, block_size=block_size,
                                                 kernel_size=kernel_size, radius=radius)

    return rgb_result_xy.astype(np.uint8), rgb_result_xz.astype(np.uint8), rgb_result_yz.astype(np.uint8)


def rgb_projection_plus_normalized(data_source, resolution, radius=0.5,
                                   kernel_size=3, sample_num=10, block_size=3, area_size=1.0):
    """Function that projects the data to different planes

    :param data_source: np.ndarray, an object that stores original data
    :param resolution: int, an object that represents the resolution of the RGB image
    :param sample_num: int, an object that represents the number of points in our consideration
    :param block_size: int, an object that represents the size of the block
    :param radius: int, an object that represents the bias for the coordinates to deal with the transformation
    :param kernel_size: int, an object that represents the size of the kernel used in filtering
    :param area_size: float, an object that represents the boundary of the given point cloud block
    :return:
    """

    # Scale all points to the range (0,1)
    # Normalization
    data = data_source[:, 0:6]

    data[:, 0], data[:, 1], data[:, 2] = normalize_xyz(xyz_data=data[:, 0], area_size=area_size), \
                                         normalize_xyz(xyz_data=data[:, 1], area_size=area_size), \
                                         normalize_xyz(xyz_data=data[:, 2], area_size=area_size)

    # Generates RGB data in different projection
    rgb_result_xy = generate_rgb_projection_data(data=data,
                                                 resolution=resolution, plane_name='xy',
                                                 sample_num=sample_num, block_size=block_size,
                                                 kernel_size=kernel_size, radius=radius)
    rgb_result_xz = generate_rgb_projection_data(data=data,
                                                 resolution=resolution, plane_name='xz',
                                                 sample_num=sample_num, block_size=block_size,
                                                 kernel_size=kernel_size, radius=radius)
    rgb_result_yz = generate_rgb_projection_data(data=data,
                                                 resolution=resolution, plane_name='yz',
                                                 sample_num=sample_num, block_size=block_size,
                                                 kernel_size=kernel_size, radius=radius)

    return rgb_result_xy.astype(np.uint8), rgb_result_xz.astype(np.uint8), rgb_result_yz.astype(np.uint8)


def round2int(value, boundary=-1):
    """Function that is used to obtain rounding value

    :param value: float32, an original value
    :param boundary: int, an object that represents the boundary of rounded value or no boundary
    :return:
    """

    if value - int(value) >= 0.5:
        if boundary == -1 or (int(value) + 1) <= boundary:
            return int(value) + 1
        else:
            return boundary
    else:
        return int(value)


def generate_2rgb_opposite_projection(data, xyz_data, resolution, radius=0.5,
                                      sample_num=10, block_size=3):
    """Function that is used to generate two opposite projection matrices

    :param data: np.ndarray, an object that stores original data
    :param xyz_data: np.ndarray, an object that stores xyz information only
    :param resolution: int, an object that represents the size of the grid
    :param sample_num: int, an object that represents the number to sample from the block
    :param block_size: int, an object that represents the size of block
    :param radius: int, an object that represents the bias for the coordinates to deal with the transformation
    :return:
    """

    rgb_projected_result = np.zeros((resolution, resolution, 3))

    # Create a container to calculate the frequency and obtain the rgb at the same time
    locations = {}

    # Choose the nearest point and project those points into blocks
    for idx in range(data.shape[0]):
        # Shift 0.5/resolution to right and scale
        shifted_x = (xyz_data[idx, 0] + radius / resolution) * resolution
        shifted_y = (xyz_data[idx, 1] + radius / resolution) * resolution

        # Allocate the point to the specific block
        block_x = round2int(value=shifted_x, boundary=resolution)
        block_y = round2int(value=shifted_y, boundary=resolution)

        # Append to the list
        # Every element in the array is like: [idx, [x, y, z], dist2center]
        # Actually, dist2center is a placeholder for the following kNN calculating
        loc4pt = tuple([block_x-1, block_y-1])
        if loc4pt not in locations:
            locations[loc4pt] = []
        locations[loc4pt].append([idx, xyz_data[idx, :], -1])

    # Sort all points in the blocks and obtain the RGB value of every block
    for idx1 in range(resolution):
        for idx2 in range(resolution):
            # Sort all points in one block

            point_array, array_length = generate_point_array4block(block_dict=locations, center_index=[idx1, idx2],
                                                                   resolution=resolution, block_size=block_size)
            quicksort(array=point_array, low=0, high=array_length-1)

            # Vote for the RGB selected
            selected_r, selected_g, selected_b = 255, 255, 255
            if array_length >= sample_num:
                selected_r, selected_g, selected_b = knn_vote4rgb_in_block(data=data,
                                                                           array=point_array,
                                                                           sample_num=sample_num)
            elif 0 < array_length < sample_num:
                selected_r, selected_g, selected_b = knn_vote4rgb_in_block(data=data,
                                                                           array=point_array,
                                                                           sample_num=array_length)

            # Store the RGB value
            rgb_projected_result[idx1, idx2, :] = np.array([selected_r, selected_g, selected_b], dtype=np.uint8)

    return rgb_projected_result


def calculate_distance2center(array4pixel, centerx, centery, array4block):
    """Function that is used to calculate the distance between the given point to the center point

    :param array4pixel: [], an object that stores every point that describes the same pixel
    :param centerx: int, an object that represents x coordinate of the center
    :param centery: int, an object that represents y coordinate of the center
    :param array4block: [], an object that stores every point in a given block
    :return:
    """

    for idx in range(len(array4pixel)):
        x_dist = array4pixel[idx][1][0] - centerx
        y_dist = array4pixel[idx][1][1] - centery
        z_dist = array4pixel[idx][1][2]
        # Separate two space along the 0.5 axis
        z_dist = 0.5 - z_dist if z_dist <= 0.5 else z_dist + 0.5
        # Vertical distance matters most
        z_dist = np.exp(z_dist*np.pi)

        array4pixel[idx][2] = pow(x_dist ** 2 + y_dist ** 2 + z_dist ** 2, 1. / 3.)
        array4block.append(array4pixel[idx])


def generate_point_array4block(block_dict, center_index, resolution=256, block_size=3):
    """Function that is used to generate point array for blocks around the center

    :param block_dict: dictionary, an object that stores every point in dataset
    :param center_index: [int], an array that stores current coordinates
    :param resolution: int, an object that represents the size of the grid
    :param block_size: int, an object that represents the size of the block
    :return:
    """

    centerx = center_index[0]
    centery = center_index[1]
    point_array4block = []

    # Four corners
    # left-top corner
    if centerx <= block_size/2 and centery <= block_size/2:
        for idx1 in range(0, centerx+int(block_size/2)+1):
            for idx2 in range(0, centery+int(block_size/2)+1):
                if (idx1, idx2) in block_dict:
                    calculate_distance2center(array4pixel=block_dict[(idx1, idx2)],
                                              array4block=point_array4block,
                                              centerx=centerx, centery=centery)
    # right-top corner
    elif centerx <= block_size/2 and centery >= (resolution-block_size/2):
        for idx1 in range(0, centerx+int(block_size/2)+1):
            for idx2 in range(centery-int(block_size/2), resolution):
                if (idx1, idx2) in block_dict:
                    calculate_distance2center(array4pixel=block_dict[(idx1, idx2)],
                                              array4block=point_array4block,
                                              centerx=centerx, centery=centery)
    # left-down corner
    elif centerx >= (resolution-block_size/2) and centery <= block_size/2:
        for idx1 in range(centerx-int(block_size/2), resolution):
            for idx2 in range(0, centery+int(block_size/2)):
                if (idx1, idx2) in block_dict:
                    calculate_distance2center(array4pixel=block_dict[(idx1, idx2)],
                                              array4block=point_array4block,
                                              centerx=centerx, centery=centery)
    # right-down corner
    elif centerx >= (resolution-block_size/2) and centery >= (resolution-block_size/2):
        for idx1 in range(centerx-int(block_size/2), resolution):
            for idx2 in range(centery-int(block_size/2), resolution):
                if (idx1, idx2) in block_dict:
                    calculate_distance2center(array4pixel=block_dict[(idx1, idx2)],
                                              array4block=point_array4block,
                                              centerx=centerx, centery=centery)
    # Four edges
    # top edge
    elif centerx <= block_size/2:
        for idx1 in range(0, centerx+int(block_size/2)+1):
            for idx2 in range(centery-int(block_size/2), centery-int(block_size/2)+1):
                if (idx1, idx2) in block_dict:
                    calculate_distance2center(array4pixel=block_dict[(idx1, idx2)],
                                              array4block=point_array4block,
                                              centerx=centerx, centery=centery)
    # down edge
    elif centerx >= (resolution-block_size/2):
        for idx1 in range(centerx-int(block_size/2), resolution):
            for idx2 in range(centery-int(block_size/2), centery-int(block_size/2)+1):
                if (idx1, idx2) in block_dict:
                    calculate_distance2center(array4pixel=block_dict[(idx1, idx2)],
                                              array4block=point_array4block,
                                              centerx=centerx, centery=centery)
    # left edge
    elif centery <= block_size/2:
        for idx1 in range(centerx-int(block_size/2), centerx+int(block_size/2)+1):
            for idx2 in range(0, centery+int(block_size/2)+1):
                if (idx1, idx2) in block_dict:
                    calculate_distance2center(array4pixel=block_dict[(idx1, idx2)],
                                              array4block=point_array4block,
                                              centerx=centerx, centery=centery)
    # right edge
    elif centery >= (resolution-block_size/2):
        for idx1 in range(centerx-int(block_size/2), centerx+int(block_size/2)+1):
            for idx2 in range(centery-int(block_size/2), resolution):
                if (idx1, idx2) in block_dict:
                    calculate_distance2center(array4pixel=block_dict[(idx1, idx2)],
                                              array4block=point_array4block,
                                              centerx=centerx, centery=centery)
    # Other areas
    else:
        for idx1 in range(centerx-int(block_size/2), centerx+int(block_size/2)+1):
            for idx2 in range(centery-int(block_size/2), centery-int(block_size/2)+1):
                if (idx1, idx2) in block_dict:
                    calculate_distance2center(array4pixel=block_dict[(idx1, idx2)],
                                              array4block=point_array4block,
                                              centerx=centerx, centery=centery)

    return point_array4block, len(point_array4block)


def apply_average_filter2remove_noise(image, resolution=256, kernel_size=3):
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

    # Calculate the convolution on the given RGB image matrix
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


def generate_rgb_projection_data(data, resolution, plane_name, radius=0.5,
                                 kernel_size=3, sample_num=10, block_size=3):
    """Function that generates RGB data in one projection plane

    :param data: np.ndarray, an object that stores original data
    :param resolution: int, an object that represents the resolution of the RGB image
    :param plane_name: str, an object that represents the projection plane
    :param block_size: int, an object that determine the block for kNN
    :param sample_num: int, an object that represents the number of points in our consideration
    :param radius: int, an object that represents the bias for the coordinates to deal with the transformation
    :param kernel_size: int, an object that represents the size of the kernel used in filtering
    :return:
    """

    rgb_2projected_result = np.zeros((2, resolution, resolution, 3))
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
    # Obtain the RGB image result
    rgb_2projected_result[0, :, :, :] = generate_2rgb_opposite_projection(data=data, xyz_data=xyz_data,
                                                                          resolution=resolution,
                                                                          sample_num=sample_num,
                                                                          radius=radius,
                                                                          block_size=block_size)
    # Remove the white pixels in the matrix
    if kernel_size != -1:
        for idx in range(3):
            # rgb_2projected_result[0, :, :, idx] = \
            #     apply_average_filter2remove_noise(image=rgb_2projected_result[0, :, :, idx],
            #                                       resolution=resolution,
            #                                       kernel_size=kernel_size)
            rgb_2projected_result[0, :, :, idx] = cv2.GaussianBlur(src=rgb_2projected_result[0, :, :, idx],
                                                                   ksize=(kernel_size, kernel_size),
                                                                   sigmaX=0, sigmaY=0)

    # Allocate plane data into xy plane and distance data into z axis, yet opposite direction
    xyz_data[:, 0], xyz_data[:, 1], xyz_data[:, 2] = plane_data[:, 0], plane_data[:, 1], 1-distance2plane_data[:]
    # Obtain the RGB image result
    rgb_2projected_result[1, :, :, :] = generate_2rgb_opposite_projection(data=data, xyz_data=xyz_data,
                                                                          resolution=resolution,
                                                                          sample_num=sample_num,
                                                                          radius=radius,
                                                                          block_size=block_size)
    # Remove the white pixels in the matrix
    if kernel_size != -1:
        for idx in range(3):
            # rgb_2projected_result[1, :, :, idx] = \
            #     apply_average_filter2remove_noise(image=rgb_2projected_result[1, :, :, idx],
            #                                       resolution=resolution,
            #                                       kernel_size=kernel_size)
            rgb_2projected_result[1, :, :, idx] = cv2.GaussianBlur(src=rgb_2projected_result[1, :, :, idx],
                                                                   ksize=(kernel_size, kernel_size),
                                                                   sigmaX=0, sigmaY=0)

    # Adjust the RGB image to the right angle
    if plane_name == 'yz':
        rgb_2projected_result[0, :, :, :] = np.rot90(rgb_2projected_result[0, :, :, :], 1)
        rgb_2projected_result[1, :, :, :] = np.rot90(rgb_2projected_result[1, :, :, :], 1)
    if plane_name == 'xz':
        rgb_2projected_result[0, :, :, :] = np.rot90(rgb_2projected_result[0, :, :, :], 1)
        rgb_2projected_result[1, :, :, :] = np.rot90(rgb_2projected_result[1, :, :, :], 1)

    return rgb_2projected_result


def distance_weight4voting(distance):
    """Function that is used to calculate the weight of the distance for RGB voting

    :param distance:
    :return:
    """

    weight = 1./(0.5*math_tool.pi)*np.exp((-0.5)*(distance*distance/0.25))
    return weight


def knn_vote4rgb_in_block(data, array, sample_num=10):
    """Function that counts total amount of points that belongs to a specific RGB in every block

    :param data: np.ndarray, an object that stores original data
    :param array: list, an object that stores sorted points in a block
    :param sample_num: int, an object that represents the number of sampled points
    :return:
    """

    # Limit the range
    index_end_voting = sample_num

    # Vote for the represented RGB
    voting_number_in_block_dict = {}
    for idx in range(0, index_end_voting):
        index = array[idx][0]
            
        # Count the number of points in different RGB
        point_r = data[index, 3]
        point_g = data[index, 4]
        point_b = data[index, 5]
        point_rgb = tuple([point_r, point_g, point_b])

        if point_rgb in voting_number_in_block_dict.keys():
            voting_number_in_block_dict[point_rgb] += distance_weight4voting(array[idx][2])
        else:
            voting_number_in_block_dict[point_rgb] = distance_weight4voting(array[idx][2])

    # Obtain the RGB that has largest number of voting
    if len(voting_number_in_block_dict) > 0:
        selected_rgb = max(voting_number_in_block_dict, key=voting_number_in_block_dict.get)
        return float(selected_rgb[0]), float(selected_rgb[1]), float(selected_rgb[2])
    else:
        print('Error Happens: {}->{}, which should not be zero'.format('voting_dictionary_length',
                                                                       len(voting_number_in_block_dict)))
        exit()


def main():
    """Function that obtains the arguments and generates data

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./../data/stanford_indoor3d',
                        help='Data directory path to use [default: ./../data/standford_indoor3d]')
    parser.add_argument('--res_dir', type=str, default='./../data/indoor3d_sem_seg_rgb_projection',
                        help='Result directory [default: ./../data/indoor3d_sem_seg_rgb_projection]')
    parser.add_argument('--sample_num', type=int, default=15,
                        help='Point number involving in voting in our consideration [default: 31]')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution of the projection [default: 256]')
    parser.add_argument('--block_size', type=int, default=3,
                        help='The size of a block that is used to count nearest points(must be odd) [default: 3]')
    parser.add_argument('--kernel_size', type=int, default=3, help='The size of the kernel in filtering [default: 3]')
    parser.add_argument('--radius', type=float, default=0.5, help='The bias for the coordinates [default: 0.5]')

    config = parser.parse_args()
    
    # Process every file in the data directory
    data_list = os.listdir(config.data_dir)
    for data_file in data_list:
        data_source = load_data_files(config.data_dir+'/'+data_file)
        rgb_result_xy, rgb_result_xz, rgb_result_yz = \
            rgb_projection_plus_warped_normalized(data_source=data_source, resolution=config.resolution,
                                                  sample_num=config.sample_num, block_size=config.block_size,
                                                  kernel_size=config.kernel_size, radius=config.radius)

        result_file_name = str(config.res_dir+'/'+data_file[:-4])+'_'
        # Generate the RGB matrix
        for idx in range(2):
            # Save xy plane
            xy_filename = 'xy_rgb{}_bsize{}_ksize{}'.format(idx+1, config.block_size, config.kernel_size)
            cv2.imwrite(filename=result_file_name+xy_filename+'.jpg',
                        img=rgb_result_xy[idx, :, :, ::-1])
            np.save(result_file_name+xy_filename+'.npy', rgb_result_xy[idx, :, :, ::-1])

            # Save xz plane
            xz_filename = 'xz_rgb{}_bsize{}_ksize{}'.format(idx + 1, config.block_size, config.kernel_size)
            cv2.imwrite(filename=result_file_name + xz_filename + '.jpg',
                        img=rgb_result_xz[idx, :, :, ::-1])
            np.save(result_file_name + xz_filename + '.npy', rgb_result_xz[idx, :, :, ::-1])

            # Save yz plane
            yz_filename = 'yz_rgb{}_bsize{}_ksize{}'.format(idx + 1, config.block_size, config.kernel_size)
            cv2.imwrite(filename=result_file_name + yz_filename + '.jpg',
                        img=rgb_result_yz[idx, :, :, ::-1])
            np.save(result_file_name + yz_filename + '.npy', rgb_result_yz[idx, :, :, ::-1])

        print('FINISH: ' + data_file[:-4])


if __name__ == '__main__':
    main()


