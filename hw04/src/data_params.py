"""
    timestamp   0
    label       1
    --------------
    C1          2
    C2          3
    C3          4
    C4          5
    C5          6
    C6          7
    C7          8
    C8          9
    C9          10
    C10         11
    --------------
    CG1         12
    CG2         13
    CG3         14
    --------------
    l1          15
    l2          16
    --------------
    C11         17
    C12         18
"""

head_line = 'timestamp;label;C1;C2;C3;C4;C5;C6;C7;C8;C9;C10;CG1;CG2;CG3;l1;l2;C11;C12'
idx_timestamp = 0
idx_label = 1
idx_categorical = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18]
idx_group = [12, 13, 14]
idx_integer = [15, 16]

n_positive = 76302
n_negative = 29913450
n_total = n_positive + n_negative
