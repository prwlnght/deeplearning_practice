'''

Sub-project: EAI, how does RNN work? and why does RNN work?

Goals:

1. Implement a classifier based on Recurrent Neural Network and understand its compoments
2. Implement an auto-encoder to 'predict' the next time-stamp
3. Train on the entire network, then animate
4. Switch datasets.


Input:
x,y coodinate data for a bunch of bone locations > learn2Sign


Ouput:
Varies by step

'''

import os, platform, sys


if platform.system() == 'Windows':
    import resources_windows as resources

    tmp_dir = os.path.join(resources.workspace_dir, 'tmp', 'grusi')
    log_file = os.path.join(tmp_dir, 'run_logs.log')
else:
    import resources_unix as resources

    tmp_dir = os.path.join(resources.workspace_dir, 'tmp', 'grusi')
    log_file = os.path.join(tmp_dir, 'run_server_logs.log')