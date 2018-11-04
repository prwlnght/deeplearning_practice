import tensorflow as tf
import platform
import datetime as dt
import os

if platform.system() == 'Windows':
    import resources_windows as resources
else:
    import resources_unix as resources

tmp_dir = os.path.join(resources.workspace_dir, 'tmp')
log_file = os.path.join(tmp_dir, 'grusi', 'run_logs.log')

print('Tensorflow imported')

with open(log_file, 'a') as m_log:
    m_log.write('-------------------------------------------------------------------------')
    m_log.write(str(dt.datetime.now()) + '\n')
    m_log.write('testing the log file')
    m_log.write('-------------------------------------------------------------------------')