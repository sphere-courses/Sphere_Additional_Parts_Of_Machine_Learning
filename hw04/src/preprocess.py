import os
import math

import data_params as dp
from utils import split, parallelize, merge, run_script, train_validation_split


def convert_to_vw_format(in_path, out_path=None, test=False, line_processor=None):
    out_path = in_path + '.vw' if out_path is None else out_path
    with open(in_path, 'r') as input_file:
        with open(out_path, 'w') as output_file:
            # skip header
            maybe_header = input_file.readline().strip()
            if maybe_header != dp.head_line:
                output_file.write(line_processor(maybe_header.strip(), test))
            for line in input_file:
                output_file.write(line_processor(line.strip(), test))


def try_1_line_processor(line, test=False):
    objects = line.split(';')

    result = ''
    if not test:
        if objects[dp.idx_label] == '0':
            result += '-1 '
        else:
            result += '1 '

    # integer features
    result += '|L '
    for pos, idx in enumerate(dp.idx_integer):
        value_0 = float(objects[idx])
        value_1 = math.log(1.0 + value_0)
        value_2 = math.log(1.0 + value_1) ** 2

        result += 'C1.{0:d}.{1:d} C2.{0:d}.{2:d} C3.{0:d}.{3:d} '.format(
            pos, math.floor(value_0), math.floor(value_1), math.floor(value_2)
        )
        result += 'I1.{0:d}:{1:.3f} I2.{0:d}:{2:.3f} I3.{0:d}:{3:.3f} '.format(
            pos, value_0, value_1, value_2
        )

    # categorical features
    result += '|C '
    for pos, idx in enumerate(dp.idx_categorical):
        result += '{0:d}.{1} '.format(pos, objects[idx])

    # group features
    for pos, idx in enumerate(dp.idx_group):
        group = objects[idx].split(',')
        weight = 1.0 / math.sqrt(len(group) + 1.0)
        result += '|G{0:d}:{1:.3f} '.format(pos, weight)
        for value in group:
            result += '{0} '.format(value)

    return result + '\n'


if __name__ == '__main__':
    print('#######  Split test file  #######')
    if not os.path.exists('./data/tmp'):
        split('./data/test.csv', './data/tmp/test', 2_800_000)
        split('./data/train_full.csv', './data/tmp/train', 3_800_000)
    print('#######  Convert chunks to vw format  #######')
    parallelize(
        './data/tmp/test', lambda path: path.find('.') == -1,
        convert_to_vw_format, 8, test=True, line_processor=try_1_line_processor
    )
    parallelize(
        './data/tmp/train', lambda path: path.find('.') == -1,
        convert_to_vw_format, 8, test=False, line_processor=try_1_line_processor
    )
    print('#######  Merge chunks  #######')
    merge('./data/tmp/test', './data/test.vw', lambda path: path.endswith('.vw'))
    run_script('rm -rf ./data/tmp/test/*.vw')
    merge('./data/tmp/train', './data/train_full.vw', lambda path: path.endswith('.vw'))
    run_script('rm -rf ./data/tmp/train/*.vw')
    print('#######  Split train into train and validation parts  #######')
    train_validation_split('./data/train_full.vw', './data', ['train_full.vw', 'train.vw', 'validation.vw'], 0.7)
