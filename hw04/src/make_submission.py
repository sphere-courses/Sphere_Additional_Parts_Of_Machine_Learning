from utils import make_submission, run_script

if __name__ == '__main__':
    weights = [
        [0.254, 0.256, 0.224, 0.255, 0.013]
    ]

    for idx, weight in enumerate(weights):
        print('#######  Convert model output to the kaggle format  #######')
        make_submission(
            [
                './data/probas_1.txt',
                './data/probas_2.txt',
                './data/probas_3.txt',
                './data/probas_4.txt',
                './data/probas_5.txt'
            ],
            './data/submission.txt', weight
        )

        print('#######  Submit new result to kaggle  #######')
        run_script(
            '~/.local/bin/kaggle competitions submit online-advertising-challenge-fall-2018\
             -f ./data/submission.txt -m "{0:d}"'.format(idx + 1)
        )

    print('#######  Get result from kaggle  #######')
    run_script(
        '~/.local/bin/kaggle competitions submissions online-advertising-challenge-fall-2018'
    )
