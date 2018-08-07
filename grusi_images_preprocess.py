'''
copyright @prwl_nght

Input: rgb image examples per class
Output: greysclaed and resized images

'''


import os
from PIL import Image
from PIL import ImageOps
import platform

if platform.system() == 'Windows':
    import resources_windows as resources
else:
    import resources_unix as resources

to_rgb_to_grayscale = True

input_dir = os.path.join(resources.workspace_dir, 'data', 'grusi', 'figures_all_smoothing5')
output_dir = os.path.join(resources.workspace_dir, 'data', 'grusi', 'figures_grayscale')
tmp_dir = os.path.join(resources.workspace_dir, 'tmp')
log_file = os.path.join(tmp_dir, 'grusi', 'run_logs.log')

resize_shape = (28, 28)

# booleans
to_invert_image = True

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

'''grayscaling'''


def rgb_to_grayscale(m_input_dir=input_dir, m_resize_shape=resize_shape):
    # img = Image.open(input_image).convert('LA')
    counter = 0
    if __name__ == '__main__':
        if to_rgb_to_grayscale:
            for folder in os.listdir(m_input_dir):
                if folder.startswith('.DS'):
                    continue
                for file in os.listdir(os.path.join(m_input_dir, folder)):
                    if file.startswith('.'):
                        continue

                    this_output_dir = os.path.join(output_dir, folder)
                    if not os.path.exists(this_output_dir):
                        os.mkdir(this_output_dir)
                    if file.endswith('jpeg'):
                        in_file = os.path.join(m_input_dir, folder, file)
                        gray_image = Image.open(in_file).convert('L')
                        gray_image = gray_image.resize(m_resize_shape)
                        if to_invert_image:
                            gray_image = ImageOps.invert(gray_image)
                        out_file = os.path.join(this_output_dir, file)
                        gray_image.save(out_file)
                        counter += 1
                print('Processed, ', counter, ' files')
    with open(log_file, 'a') as m_log:
        m_log.write(str(os.path.basename(__file__)) + ' processed: ' + str(counter) + ' files\n')


if __name__ == '__main__':
    rgb_to_grayscale()
