import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
from model import *


parser = argparse.ArgumentParser()

# Arguments relevant to network for point cloud data
parser.add_argument('--gpu', type=int, default=2, help='GPU to use [default: GPU 2]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=121, help='Epoch to run [default: 121]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=3, help='Which area to use for test, option: 1-6 [default: 3]')

# Arguments relevant to network for projection data
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--num_pf_views', type=int, default=3)
parser.add_argument('--num_rgb_views', type=int, default=6)

FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
IMAGE_SIZE = FLAGS.image_size
NUM_PF_VIEWS = FLAGS.num_pf_views
NUM_RGB_VIEWS = FLAGS.num_rgb_views

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 13

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# RGB projection data path
RGB_DATA_PATH = './../data/indoor3d_sem_seg_rgb_projection'

# Point frequency projection data path
PF_DATA_PATH = './../data/indoor3d_sem_seg_pf_projection'

ALL_FILES = provider.getDataFiles('./../data/indoor3d_sem_seg_hdf5_data/all_files.txt')
room_filelist = [line.rstrip() for line in open('./../data/indoor3d_sem_seg_hdf5_data/room_filelist.txt')]

# Load ALL data
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
print('Data Batches Shape: {}'.format(data_batches.shape))
print('Label Batches Shape: {}'.format(label_batches.shape))


def load_rgb_pf_data(path, type_of_projection=True):
    """Function that is used to load rgb and point frequency projection data

    :param path: string, an object that stores the path for data
    :param type_of_projection: bool, an object that tells which data to be loaded, rgb or point frequency
    :return:
    """

    room_filelist = [line.rstrip() for line in open(os.path.join(path, 'room_filelist.txt'))]

    data_list = []
    for _, block_name in enumerate(room_filelist):
        room_name = block_name[block_name.find('_') + 1:]
        block_num = block_name.split('_')[0]
        block_data_dict = {}
        for dir in os.listdir(os.path.join(path, room_name)):

            # Obtain different data of RGB or Point Frequency
            if type_of_projection is True:
                block_data_path = os.path.join(path, room_name, dir, 'rgb{}_{}.npy'.format(block_num, dir))
                if not os.path.exists(block_data_path):
                    for idx in range(1, 121):
                        block_data_path = os.path.join(path, room_name, dir, 'rgb{}_{}.npy'.format(str(int(block_num)+idx), dir))
                        if os.path.exists(block_data_path):
                            break
            else:
                block_data_path = os.path.join(path, room_name, dir, 'pf{}_{}.npy'.format(block_num, dir))
                if not os.path.exists(block_data_path):
                    for idx in range(1, 121):
                        block_data_path = os.path.join(path, room_name, dir, 'pf{}_{}.npy'.format(str(int(block_num)+idx), dir))
                        if os.path.exists(block_data_path):
                            break

            block_data = np.load(block_data_path)/255.0
            block_data = np.expand_dims(block_data, 0)
            block_data_dict[dir] = block_data
        data_list.append(block_data_dict)

    multiview_data_list = []
    for key in sorted(data_list[0].keys()):
        oneview_data_list = [data_list[idx][key] for idx in range(len(data_list))]
        oneview_data = np.concatenate(oneview_data_list, 0)
        multiview_data_list.append(oneview_data)

    return multiview_data_list


# Load RGB and Point Frequency data
rgb_data_batches_list = load_rgb_pf_data(path=RGB_DATA_PATH, type_of_projection=True)
pf_data_batches_list = load_rgb_pf_data(path=PF_DATA_PATH, type_of_projection=False)

# Split the data into train data and test data
test_area = 'Area_'+str(FLAGS.test_area)
train_idxs = []
test_idxs = []
for i, room_name in enumerate(room_filelist):
    if test_area in room_name:
        test_idxs.append(i)
    else:
        train_idxs.append(i)

# Generate train data and label
train_data = data_batches[train_idxs,...]
rgb_train_data = []
pf_train_data = []

for rgb_data_batches in rgb_data_batches_list:
    rgb_train_data.append(rgb_data_batches[train_idxs, ...])

for pf_data_batches in pf_data_batches_list:
    pf_train_data.append(pf_data_batches[train_idxs, ...])

train_label = label_batches[train_idxs]

# Generate test data and label
test_data = data_batches[test_idxs,...]
rgb_test_data = []
pf_test_data = []

for rgb_data_batches in rgb_data_batches_list:
    rgb_test_data.append(rgb_data_batches[test_idxs, ...])

for pf_data_batches in pf_data_batches_list:
    pf_test_data.append(pf_data_batches[test_idxs, ...])

test_label = label_batches[test_idxs]

print('Point Cloud -> Train Data Shape: {}, Train Label Shape: {}'.format(train_data.shape, train_label.shape))
print('Point Cloud -> Test Data Shape: {}, Test Label Shape: {}'.format(test_data.shape, test_label.shape))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, rgb_projection_pl, pf_projection_pl, labels_pl = placeholder_inputs(batch_size=BATCH_SIZE,
                                                                                                num_point=NUM_POINT,
                                                                                                image_size=IMAGE_SIZE,
                                                                                                num_pf_views=NUM_PF_VIEWS,
                                                                                                num_rgb_views=NUM_RGB_VIEWS)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred = get_model(point_cloud=pointclouds_pl, projection_data=[pf_projection_pl, rgb_projection_pl],
                             is_training=is_training_pl, image_size=IMAGE_SIZE, num_class=NUM_CLASSES,
                             bn_decay=bn_decay)
            loss = get_loss(pred=pred, label=labels_pl)

            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'rgb_projection_pl': rgb_projection_pl,
               'pf_projection_pl': pf_projection_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    # current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label)
    rgb_current_data, pf_current_data, pc_current_data, current_label, _ = provider.shuffle_rgb_pf_pc_data(rgb_data=rgb_train_data,
                                                                                                            pf_data=pf_train_data,
                                                                                                            pc_data=train_data,
                                                                                                            labels=train_label)

    # Obtain the pre-processed RGB and point frequency data
    rgb_current_data = np.concatenate(rgb_current_data, 0)
    pf_current_data = np.concatenate(pf_current_data, 0)
    pf_current_data = np.reshape(pf_current_data,
                                 (pf_current_data.shape[0], pf_current_data.shape[1], IMAGE_SIZE, IMAGE_SIZE, 1))

    file_size = pc_current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print(('Current batch/total batch num: %d/%d'%(batch_idx,num_batches)))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        feed_dict = {ops['pointclouds_pl']: pc_current_data[start_idx:end_idx, :, :],
                     ops['rgb_projection_pl']: rgb_current_data[:, start_idx:end_idx, :, :, :],
                     ops['pf_projection_pl']: pf_current_data[:, start_idx:end_idx, :, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
    
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('-----------')

    rgb_current_data = np.concatenate(rgb_test_data, 0)
    pf_current_data = np.concatenate(pf_test_data, 0)
    pf_current_data = np.reshape(pf_current_data,
                                 (pf_current_data.shape[0].pf_curent_data.shape[1], IMAGE_SIZE, IMAGE_SIZE, 1))

    pc_current_data = test_data[:, 0:NUM_POINT, :]
    current_label = np.squeeze(test_label)
    
    file_size = pc_current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: pc_current_data[start_idx:end_idx, :, :],
                     ops['rgb_projection_pl']: rgb_current_data[:, start_idx:end_idx, :, :, :],
                     ops['pf_projection_pl']: pf_current_data[:, start_idx:end_idx, :, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
         


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
