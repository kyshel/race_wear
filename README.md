# race_wear

## basic info
raw/train.tgz > train:val 0.8:0.2

## steps
- if manA in manB, remove manA that score <= 0.5
- if man has no label, label it 'no'


## submit
- s8 0.89
- s7 0.7415 remove man_score <= 0.5
- s6 0.7463 
- s3 0.71
- s2 0.17


## info
- s7 seems like overlapped iou do not affect result, remove overlapped man will decrease scpre


