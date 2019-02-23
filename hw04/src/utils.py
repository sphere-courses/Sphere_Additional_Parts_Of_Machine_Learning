import os
import math
import errno
import subprocess
import multiprocessing

import numpy as np
from natsort import natsorted


def split(in_path, out_path, n_lines, names=None, prefix='', suffix=''):
    idx = 0
    out_file = None
    if out_path[-1] != '/':
        out_path += '/'
    with open(in_path, 'r') as in_file:
        for lineno, line in enumerate(in_file):
            if lineno % n_lines == 0:
                if out_file:
                    out_file.close()
                if names is None:
                    out_file_path = out_path + prefix + '{0:d}'.format(lineno // n_lines + 1) + suffix
                else:
                    out_file_path = out_path + names[idx]
                    idx += 1
                if not os.path.exists(os.path.dirname(out_file_path)):
                    try:
                        os.makedirs(os.path.dirname(out_file_path))
                    except OSError as exc:  # Guard against race condition
                        if exc.errno != errno.EEXIST:
                            raise
                out_file = open(out_file_path, 'w')
            out_file.write(line)
        if out_file:
            out_file.close()


def merge(in_path, out_path, condition=None):
    if not os.path.exists(os.path.dirname(out_path)):
        try:
            os.makedirs(os.path.dirname(out_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    in_paths = (in_path + '/' + path for path in natsorted(os.listdir(in_path)))
    if condition is not None:
        in_paths = (in_path for in_path in in_paths if condition(in_path))
    with open(out_path, 'w') as out_file:
        for in_file in (open(path, 'r') for path in in_paths):
            for line in in_file:
                out_file.write(line)
            in_file.close()


def remove(path):
    if not os.path.exists(path):
        return
    if os.path.isfile(path):
        os.remove(path)
        return
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(path)


def parallelize(in_path, condition, func, n_workers, *args, **kwargs):
    in_paths = (in_path + '/' + path for path in os.listdir(in_path) if condition(path))
    with multiprocessing.Pool(n_workers) as executor:
        futures = {in_path: executor.apply_async(func, (in_path,) + args, kwargs) for in_path in in_paths}
        for (in_path, future) in futures.items():
            try:
                result = future.get()
            except Exception as exc:
                print('Processing path %r generated an exception: %s' % (in_path, exc))
            else:
                print(in_path + ' was processed successfully')
                print('Result: ' + str(result))


def count_lines(path, *args, **kwargs):
    result = 0
    with open(path, 'r') as file:
        for _ in file:
            result += 1
    return result


def train_validation_split(in_path, out_path, names, ratio=0.6):
    n_lines = count_lines(in_path)
    ext_idx = in_path.rfind('/') + 1
    tmp_path = in_path[:ext_idx] + names[0]
    os.rename(in_path, tmp_path)
    split(tmp_path, out_path, int(ratio * n_lines), names[1:])


def run_script(script):
    return_code = subprocess.call(script, shell=True)
    return return_code


def make_submission(in_paths, out_path, weights, subsample_ratio=1.):
    fix_ratio = (
        lambda proba:
            1. / (1. - 1. / subsample_ratio + 1. / (proba * subsample_ratio)) if not math.isclose(proba, 0.) else 0.
    )
    with open(out_path, 'w') as out_file:
        out_file.write('Id,Click\n')
        in_files = [open(in_path, 'r') for in_path in in_paths]
        idx = 0
        weights = [weight / sum(weights) for weight in weights]
        while True:
            lines = [in_file.readline() for in_file in in_files]
            if lines[0] == '':
                break
            else:
                idx += 1
            all_probas = [list(map(float, line.strip().split())) for line in lines]
            predictions = [sum(probas) / len(probas) for probas in all_probas]

            predictions = [
                fix_ratio(prediction) for prediction in predictions
            ]
            weighted_predictions = [weight * predictions[idx] for idx, weight in enumerate(weights)]
            prediction = sum(weighted_predictions)
            out_file.write('{0:d},{1:.5f}\n'.format(idx, prediction))


def evaluate_loss(in_path, probas, subsample_ratio=1.):
    loss = 0.
    n_obj = 0.
    with open(in_path, 'r') as in_file, open(probas, 'r') as probas_file:
        for line in in_file:
            n_obj += 1
            label = int(line.strip().split()[0])
            probas = list(map(float, probas_file.readline().strip().split()))
            prediction = sum(probas) / len(probas)
            prediction = 1. / (1. - 1. / subsample_ratio + 1. / (prediction * subsample_ratio))
            if label == 1:
                loss += -math.log(prediction)
            else:
                loss += -math.log(1 - prediction)
    return loss / n_obj


def get_subsample(in_path, out_path, n_positive, n_negative, extract_label=None):
    positive_idxs = []
    negative_idxs = []
    with open(in_path, 'r') as in_file:
        for idx, line in enumerate(in_file):
            label = extract_label(line.strip())
            if label == 1:
                positive_idxs.append(idx)
            else:
                negative_idxs.append(idx)
    positive_idxs = np.array(positive_idxs)
    negative_idxs = np.array(negative_idxs)

    positive_idxs = positive_idxs[np.random.permutation(positive_idxs.shape[0])[:n_positive]]
    negative_idxs = negative_idxs[np.random.permutation(negative_idxs.shape[0])[:n_negative]]

    chosen_idxs = set(positive_idxs).union(set(negative_idxs))

    with open(in_path, 'r') as in_file:
        with open(out_path, 'w') as out_file:
            for idx, line in enumerate(in_file):
                if idx in chosen_idxs:
                    out_file.write(line)
