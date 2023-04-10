PROJ_PATH=/data/bhuiyan/bee/rusty/thorax_detector/rustypatch/color

rm $PROJ_PATH/annotations/*.record

~/.conda/envs/effi_env/bin/python generate_tfrecord.py \
    -x $PROJ_PATH/images/train \
    -l $PROJ_PATH/annotations/label_map.pbtxt \
    -o $PROJ_PATH/annotations/train.record

~/.conda/envs/effi_env/bin/python generate_tfrecord.py \
    -x $PROJ_PATH/images/test \
    -l $PROJ_PATH/annotations/label_map.pbtxt \
    -o $PROJ_PATH/annotations/test.record