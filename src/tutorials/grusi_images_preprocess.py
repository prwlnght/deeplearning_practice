'''
copyright @prwl_nght

Input: rgb image examples per class
Output: greysclaed and resized images

'''

import platform, os, datetime as dt

from PIL import Image
from PIL import ImageOps

# import cv2
import random, pickle
import numpy as np

if platform.system() == 'Windows':
    import resources_windows as resources
else:
    import resources_unix as resources

# hyper parameters
train_test_split_percent = 10
channels = 3
pickle_split_size = 1000
resize_shape = (28, 28)

# booleans
to_invert_image = True
to_rgb_to_grayscale = True
to_create_intermediate_images = True
to_crop = False

input_dir = os.path.join(resources.workspace_dir, 'data', 'grusi', 'figures_all_smoothing5')

tmp_dir = os.path.join(resources.workspace_dir, 'tmp', 'grusi')
output_dir_images = os.path.join(tmp_dir,
                                 'figures_temp_' + str(resize_shape[0]) + '_' + str(resize_shape[1]) + '_' + str(
                                     channels))
log_file = os.path.join(tmp_dir, 'run_logs.log')

train_pickle_name = 'train_data'
test_pickle_name = 'test_data'

if not os.path.exists(output_dir_images):
    os.mkdir(output_dir_images)

'''grayscaling'''


def pickle_me(x, y, file_name):
    try:
        with open(file_name, 'wb') as m_dump:
            pickle.dump([x, y], m_dump)
    except Exception as e:
        print(str(e))
        print('Failed to create pickle')


def get_classes_and_users(m_input_dir):
    all_classes = []
    all_users = []
    for folder in os.listdir(m_input_dir):
        if folder.startswith('.'):
            continue
        this_class = folder
        if this_class not in all_classes:
            all_classes.append(this_class)
        for img in os.listdir(os.path.join(m_input_dir, folder)):
            if img.startswith('.'):
                continue
            this_user = img.split('_')[2]
            if this_user not in all_users:
                all_users.append(this_user)
    return all_classes, all_users


def img_to_pickles(m_input_dir=input_dir, m_resize_shape=resize_shape):
    # img = Image.open(input_image).convert('LA')
    counter = 0

    # for pickles

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

    pickle_path = os.path.join(tmp_dir,
                               'pickles_' + str(resize_shape[0]) + '_' + str(resize_shape[1]) + '_' + str(channels))

    if not os.path.exists(pickle_path):
        print('Creating directory for pickles', pickle_path)
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

    start = dt.datetime.now()
    tmp_file_counter = 0
    for im_file, im_class in combined:
        this_y = np.zeros(len(all_classes))
        this_y[all_classes.index(im_class)] += 1  # 1-hot

        this_user = im_file.split('_')[-3]

        img = Image.open(im_file)

        # resize operations
        area = (0, 0, img.size[0], int(img.size[1] / 2))
        # gray_image = gray_image.crop(area)
        img = img.resize(m_resize_shape)
        area_2 = (0.2 * img.size[0], 0.3 * img.size[1], 0.8 * img.size[0], img.size[1])
        # gray_image = gray_image.crop(area_2)
        m_downsize = (int(m_resize_shape[0] / 2), int(m_resize_shape[1] / 2))
        # gray_image = gray_image.resize(m_downsize)
        if to_invert_image:
            img = ImageOps.invert(img)
        if to_create_intermediate_images:
            this_output_dir = os.path.join(output_dir_images, im_class)
            if not os.path.exists(this_output_dir):
                os.mkdir(this_output_dir)
            tmp_file_counter += 1
            out_file = os.path.join(this_output_dir, str(tmp_file_counter) + '.jpeg')
            img.save(out_file)

        this_x = np.array(img.getdata())
        # this_x = np.array(this_x)
        img_size = len(this_x)
        if this_user in train_users:
            train_counter += 1
            train_x.append(this_x)
            train_y.append(this_y)
            if len(train_x) >= pickle_split_size:
                pickle_file_name = os.path.join(pickle_path, train_pickle_name + str(train_counter) + '.pickle')
                pickle_me(train_x, train_y, pickle_file_name)
                print('Created pickle file for ', train_counter, ' files')
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
        print('Created pickle file for ', train_counter, ' files')

    # test_pickle
    pickle_file_name = os.path.join(pickle_path, test_pickle_name + '.pickle')
    pickle_me(test_x, test_y, pickle_file_name)
    print('Created pickle file for test file')

    end = dt.datetime.now()
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

    # for folder in os.listdir(m_input_dir):
    #     if folder.startswith('.DS'):
    #         continue
    #     for file in os.listdir(os.path.join(m_input_dir, folder)):
    #         if file.startswith('.'):
    #             continue
    #
    #         this_output_dir = os.path.join(output_dir, folder)
    #         if not os.path.exists(this_output_dir):
    #             os.mkdir(this_output_dir)
    #         if file.endswith('jpeg'):
    #             in_file = os.path.join(m_input_dir, folder, file)
    #             # gray_image = Image.open(in_file).convert('L')
    #             gray_image = Image.open(in_file)
    #
    #             area = (0, 0 , gray_image.size[0], int(gray_image.size[1]/2) )
    #
    #             #gray_image = gray_image.crop(area)
    #             gray_image = gray_image.resize(m_resize_shape)
    #             area_2 = (0.2*gray_image.size[0], 0.3*gray_image.size[1], 0.8*gray_image.size[0], gray_image.size[1] )
    #             #gray_image = gray_image.crop(area_2)
    #             m_downsize = (int(m_resize_shape[0]/2), int(m_resize_shape[1]/2))
    #             # gray_image = gray_image.resize(m_downsize)
    #             if to_invert_image:
    #                 gray_image = ImageOps.invert(gray_image)
    #             out_file = os.path.join(this_output_dir, file)
    #             gray_image.save(out_file)
    #             counter += 1
    #     print('Processed, ', counter, ' files')
    # with open(log_file, 'a') as m_log:
    #     m_log.write(str(os.path.basename(__file__)) + ' processed: ' + str(counter) + ' files\n')


if __name__ == '__main__':
    if to_rgb_to_grayscale:
        img_to_pickles()
