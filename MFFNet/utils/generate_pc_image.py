import copy
import numpy as np
import open3d as o3d


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


def main():
    data_source = load_data_files('./../data/stanford_indoor3d' + '/' + 'Area_1_conferenceRoom_1.npy')

    data_source[:, 0] = data_source[:, 0]/np.max(data_source[:, 0])
    data_source[:, 1] = data_source[:, 1]/np.max(data_source[:, 1])
    data_source[:, 2] = data_source[:, 2]/np.max(data_source[:, 2])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_source[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(data_source[:, 3:6]/255)

    o3d.io.write_point_cloud('./test.ply', pcd)

    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
