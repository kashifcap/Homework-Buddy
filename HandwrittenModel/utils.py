import os
import shutil


def next_experiment_path():
    """
    creates paths for new experiment
    returns path for next experiment
    """

    i = 0
    path = os.path.join('HandwrittenModel/summary', 'experiment-{}')
    while os.path.exists(path.format(i)):
        i += 1
    path = path.format(i)
    os.makedirs(os.path.join(path, 'models'))
    os.makedirs(os.path.join(path, 'backup'))
    for file in filter(lambda x: x.endswith('.py'), os.listdir('.')):
        shutil.copy2(file, os.path.join(path, 'backup'))
    return path
