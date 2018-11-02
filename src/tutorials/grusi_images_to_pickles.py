'''
copyright @prwl_nght

A simple Rnn example to understand the workings of rnn

using image data from GRUSI

input: color
'''

import os
import tensorflow as tf
import shutil
from PIL import Image
import pickle
import datetime
import random
import platform
import datetime as dt

if platform.system() == 'Windows':
    import resources_windows as resources
else:
    import resources_unix as resources

# resources
input_dir = os.path.join(resources.workspace_dir, 'data', 'grusi', 'figures_100x100')
dir_image_splits = os.path.join(resources.workspace_dir, 'data', 'grusi', 'split')
tmp_dir = os.path.join(resources.workspace_dir, 'tmp', 'grusi')
log_file = os.path.join(tmp_dir, 'run_logs.log')
train_test_split_percent = 10
pickle_split_size = 1000
is_color_images = True

train_pickle_name = 'train_data'
test_pickle_name = 'test_data'
img_size = 100

# booleans
to_split_train_test = False  # this is not necessary anymore, creating pickles now
to_create_train_test_pickles = True


def pickle_me(x, y, file_name):
    try:
        with open(file_name, 'wb') as m_dump:
            pickle.dump([x, y], m_dump)
    except Exception as e:
        print(str(e))
        print('Failed to create pickle')


def split_train_test_dirs(m_input_dir=input_dir):
    # create a test / train split
    if not os.path.exists(dir_image_splits):
        os.mkdir(dir_image_splits)

    train_dir = os.path.join(dir_image_splits, 'train')
    test_dir = os.path.join(dir_image_splits, 'test')

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    train_counter = 0
    test_counter = 0
    this_users = os.listdir(input_dir)
    for folder in os.listdir(input_dir):
        if folder.startswith('.'):
            continue
        for file in os.listdir(os.path.join(input_dir, folder)):
            file_path = os.path.join(input_dir, folder, file)
            if file.startswith('.'):
                continue
            # renames file
            if 'sG02' in file:
                shutil.copy(file_path, test_dir)
                test_counter += 1
            else:
                shutil.copy(file_path, train_dir)
                train_counter += 1
        print('moved ', str(train_counter + test_counter), ' files')


def get_classes_and_users(m_input_dir):
    all_classes = []
    all_users = []
    for folder in os.listdir(m_input_dir):
        this_class = folder
        if this_class not in all_classes:
            all_classes.append(this_class)
        for img in os.listdir(os.path.join(m_input_dir, folder)):
            this_user = img.split('_')[2]
            if this_user not in all_users:
                all_users.append(this_user)
    return all_classes, all_users


def create_train_test_pickles(m_input_dir=input_dir, hm_train=1000, hm_test=100):
    print(m_input_dir)
    global img_size
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_counter = 0
    test_counter = 0
    train_users = []
    test_users = []
    all_classes, all_users = get_classes_and_users(m_input_dir)
    hm_test_users = int(train_test_split_percent / 100 * len(all_users))
    test_user_counter = 0

    # using the first few as test users
    for m_user in all_users:
        if test_user_counter < hm_test_users:
            test_users.append(m_user)
            test_user_counter += 1
        else:
            train_users.append(m_user)

    print('all_users: ', len(all_users), 'train_users: ', len(train_users), 'test users: ', len(test_users))
    assert len(all_users) == len(train_users) + len(test_users)

    start = datetime.datetime.now()

    pickle_path = os.path.join(tmp_dir, str(img_size) + '_pickles')
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)

    all_files_with_path = []
    m_classes = []

    # all the img files in one list
    for dirs, subdirs, file_list in os.walk(m_input_dir):
        for file in file_list:
            if file.endswith('.jpeg') or file.endswith('.png'):
                all_files_with_path.append(os.path.join(dirs, file))
                m_classes.append(file.split('_')[0])

    combined = list(zip(all_files_with_path, m_classes))
    random.shuffle(combined)



    for im_file, im_class in combined:
        this_y = [0] * len(all_classes)
        this_y[all_classes.index(im_class)] += 1 # 1-hot

        this_user = im_file.split('_')[-3]

        img = Image.open(im_file)
        this_x = list(img.getdata())
        if is_color_images:
            r, g, b = zip(*this_x)
            this_x = list(r) + list(g) + list(b)
        img_size = len(this_x)
        if this_user in train_users:
            train_counter += 1
            train_x.append(this_x)
            train_y.append(this_y)
            if len(train_x) >= pickle_split_size:
                pickle_file_name = os.path.join(pickle_path, train_pickle_name + str(train_counter) + '.pickle')
                pickle_me(train_x, train_y, pickle_file_name)
                train_x = []
                train_y = []
        else:
            test_counter += 1
            test_x.append(this_x)
            test_y.append(this_y)

    # remaining trailing files
    if len(train_x) > 0:
        pickle_file_name = os.path.join(pickle_path, train_pickle_name + str(train_counter) + '.pickle')
        pickle_me(train_x, train_y, pickle_file_name)

    # test_pickle
    pickle_file_name = os.path.join(pickle_path, test_pickle_name + '.pickle')
    pickle_me(test_x, test_y, pickle_file_name)

    end = datetime.datetime.now()
    print('train_counter: ', train_counter, 'test_counter: ', test_counter)
    print('train_pickle_length: ', len(train_x))
    print('Preprocessing time: ', end - start)

    with open(log_file, 'a') as m_log:
        m_log.write('-------------------------------------------------------------------------\n')
        m_log.write(str(dt.datetime.now()) + '\n')
        m_log.write('In file ' + str(os.path.basename(__file__)) + '\n')
        m_log.write(' train_counter: ' + str(train_counter) + ' test_counter: ' + str(test_counter))
        m_log.write(' train_pickle_length: ' + str(len(train_x)))
        m_log.write(' Preprocessing time: ' + str(end - start) + '\n')
        m_log.write('-------------------------------------------------------------------------\n')


if __name__ == '__main__':
    if to_split_train_test:
        split_train_test_dirs()

    if to_create_train_test_pickles:
        create_train_test_pickles()

# todo implement the following as solution to list bottlenecks
# todo understand how to deal with color bands
'''
ImageSize = (512, 512)
NChannelsPerImage = 3
images = [ Image.open(f, 'r') for f in batch ]
for i in images :
    assert i.size == ImageSize
    assert len(i.getbands()) == NChannelsPerImage
 
ImageShape =  (1,) + ImageSize + (NChannelsPerImage,)
allImages = [ numpy.fromstring(i.tostring(), dtype='uint8', count=-1, sep='') for i in images ]
allImages = [ numpy.rollaxis(a.reshape(ImageShape), 3, 1) for a in allImages ]
allImages = numpy.concatenate(allImages)

'''
