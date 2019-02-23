from utils import run_script


if __name__ == '__main__':
    print('#######  Train model (for making predictions)  #######')
    run_script('rm -rf ./data/model_1.vw')
    run_script('rm -rf ./data/*cache*')
    run_script(
        'vw --data ./data/train_full.vw  \
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
        --final_regressor ./data/model_1.vw \
        '
    )

    run_script('rm -rf ./data/model_2.vw')
    run_script('rm -rf ./data/*cache*')
    run_script(
        'vw --data ./data/train_full.vw  \
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
        --final_regressor ./data/model_2.vw \
        --ftrl --ftrl_alpha 0.1 --ftrl_beta 1.5 \
        '
    )

    run_script('rm -rf ./data/model_3.vw')
    run_script('rm -rf ./data/*cache*')
    run_script(
        'vw --data ./data/train_full.vw  \
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
        --final_regressor ./data/model_3.vw \
        --ftrl --ftrl_alpha 0.1 --ftrl_beta 1.5 \
        '
    )

    run_script('rm -rf ./data/model_4.vw')
    run_script('rm -rf ./data/*cache*')
    run_script(
        'vw --data ./data/train_full.vw  \
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
        --final_regressor ./data/model_4.vw \
        --ftrl --ftrl_alpha 0.1 --ftrl_beta 1.5 \
        '
    )

    run_script('rm -rf ./data/model_5.vw')
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
        --final_regressor ./data/model_5.vw \
        --ftrl --ftrl_alpha 0.1 --ftrl_beta 1.5 \
        '
    )

    print('#######  Evaluate probas on test set  #######')
    run_script(
        'vw -d ./data/test.vw -t -i ./data/model_1.vw -p ./data/probas_1.txt --loss_function logistic'
    )
    run_script(
        'vw -d ./data/test.vw -t -i ./data/model_2.vw -p ./data/probas_2.txt --loss_function logistic'
    )
    run_script(
        'vw -d ./data/test.vw -t -i ./data/model_3.vw -p ./data/probas_3.txt --loss_function logistic'
    )
    run_script(
        'vw -d ./data/test.vw -t -i ./data/model_4.vw -p ./data/probas_4.txt --loss_function logistic'
    )
    run_script(
        'vw -d ./data/test.vw -t -i ./data/model_5.vw -p ./data/probas_5.txt --loss_function logistic'
    )
