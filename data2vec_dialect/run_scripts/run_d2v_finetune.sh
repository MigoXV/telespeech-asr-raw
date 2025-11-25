. ./path.sh || exit 1

pretrained_model=/path/to/model
python  "$FAIRSEQ_DIR/fairseq_cli/hydra_train.py" -m --config-dir "$DATA2VEC_DIALECT_DIR/config/v2_dialect_asr" \
    --config-name base_audio_finetune_140h \
    common.user_dir="$DATA2VEC_DIALECT_DIR" \
    model.w2v_path=${pretrained_model} \
    task.data=/path/to/data
