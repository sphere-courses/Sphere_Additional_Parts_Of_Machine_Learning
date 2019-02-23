import numpy as np
from sklearn.metrics import log_loss

from utils import run_script

if __name__ == '__main__':
    print('#######  Train model (for validation)  #######')
    run_script('rm -rf ./data/model_1.val.vw')
    run_script('rm -rf ./data/*cache*')
    run_script(
        'vw --data ./data/train.vw  \
        --loss_function logistic \
        --link logistic \
        --kill_cache \
        --cache \
        --holdout_off \
        --learning_rate 0.1 \
        --bit_precision 28 \
        --passes 10 \
        --l1 1e-10 \
        --power_t 0.15 \
        --final_regressor ./data/model_1.val.vw \
        '
    )

    run_script('rm -rf ./data/model_2.val.vw')
    run_script('rm -rf ./data/*cache*')
    run_script(
        'vw --data ./data/train.vw  \
        --loss_function logistic \
        --link logistic \
        --holdout_off \
        --kill_cache \
        --cache \
        --quadratic CC \
        --bit_precision 28 \
        --passes 10 \
        --l1 0.1 \
        --l2 0.1 \
        --final_regressor ./data/model_2.val.vw \
        --ftrl --ftrl_alpha 0.1 --ftrl_beta 1.5 \
        '
    )

    run_script('rm -rf ./data/model_3.val.vw')
    run_script('rm -rf ./data/*cache*')
    run_script(
        'vw --data ./data/train.vw  \
        --loss_function logistic \
        --link logistic \
        --holdout_off \
        --kill_cache \
        --cache \
        --bit_precision 28 \
        --passes 10 \
        --l1 1.0 \
        --l2 1.0 \
        --power_t 0.15 \
        --final_regressor ./data/model_3.val.vw \
        --ftrl --ftrl_alpha 0.1 --ftrl_beta 1.5 \
        '
    )

    run_script('rm -rf ./data/model_4.val.vw')
    run_script('rm -rf ./data/*cache*')
    run_script(
        'vw --data ./data/train.vw  \
        --loss_function logistic \
        --link logistic \
        --holdout_off \
        --kill_cache \
        --cache \
        --quadratic GG \
        --bit_precision 28 \
        --passes 5 \
        --l1 0. \
        --l2 0. \
        --final_regressor ./data/model_4.val.vw \
        --ftrl --ftrl_alpha 0.1 --ftrl_beta 1.5 \
        '
    )

    run_script('rm -rf ./data/model_5.val.vw')
    run_script('rm -rf ./data/*cache*')
    run_script(
        'vw --data ./data/train_full.vw  \
        --loss_function logistic \
        --link logistic \
        --holdout_off \
        --kill_cache \
        --nn 3 \
        --dropout \
        --cache \
        --bit_precision 28 \
        --passes 5 \
        --l1 0. \
        --l2 0. \
        --final_regressor ./data/model_5.val.vw \
        --ftrl --ftrl_alpha 0.1 --ftrl_beta 1.5 \
        '
    )

    print('#######  Evaluate probas on validation set  #######')
    run_script(
        'vw -d ./data/test.vw -t -i ./data/model_1.val.vw -p ./data/probas_1.val.txt --loss_function logistic'
    )
    run_script(
        'vw -d ./data/test.vw -t -i ./data/model_2.val.vw -p ./data/probas_2.val.txt --loss_function logistic'
    )
    run_script(
        'vw -d ./data/test.vw -t -i ./data/model_3.val.vw -p ./data/probas_3.val.txt --loss_function logistic'
    )
    run_script(
        'vw -d ./data/test.vw -t -i ./data/model_4.val.vw -p ./data/probas_4.val.txt --loss_function logistic'
    )
    run_script(
        'vw -d ./data/test.vw -t -i ./data/model_5.val.vw -p ./data/probas_5.val.txt --loss_function logistic'
    )

    print('#######  Extract targets from validation set  #######')
    with open('./data/validation.vw', 'r') as in_file:
        with open('./data/validation.classes', 'w') as out_file:
            for line in in_file:
                out_file.write(line.strip().split(' ')[0] + '\n')

    print('#######  Search mixing coefficients  #######')
    y, pr_1, pr_2, pr_3, pr_4, pr_5 = [], [], [], [], [], []

    with open('./data/validation.classes', 'r') as file:
        for line in file:
            y.append(int(line.strip()))

    with open('./data/probas_1.val.txt', 'r') as file:
        for line in file:
            pr_1.append(float(line.strip()))

    with open('./data/probas_2.val.txt', 'r') as file:
        for line in file:
            pr_2.append(float(line.strip()))

    with open('./data/probas_3.val.txt', 'r') as file:
        for line in file:
            pr_3.append(float(line.strip()))

    with open('./data/probas_4.val.txt', 'r') as file:
        for line in file:
            pr_4.append(float(line.strip()))

    with open('./data/probas_5.val.txt', 'r') as file:
        for line in file:
            pr_5.append(float(line.strip()))

    with open('./data/validation.classes', 'r') as file:
        for line in file:
            y.append(int(line.strip()))

    y = np.array(y)
    pr_1 = np.array(pr_1)
    pr_2 = np.array(pr_2)
    pr_3 = np.array(pr_3)
    pr_4 = np.array(pr_4)
    pr_5 = np.array(pr_5)

    for idx in range(100):
        w_1, w_2, w_3, w_4, w_5 = np.random.uniform(0., 1., 5)
        w = w_1 + w_2 + w_3 + w_4 + w_5 + 0.001
        w_1, w_2, w_3, w_4, w_5 = w_1 / w, w_2 / w, w_3 / w, w_4 / w, w_5 / w
        ll = log_loss(y, w_1 * pr_1 + w_2 * pr_2 + w_3 * pr_3 + w_4 * pr_4 + w_5 * pr_5)
        print('{0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}, {4:.3f}, {5:.6f}'.format(w_1, w_2, w_3, w_4, w_5, ll))
