#for col in student teacher
#do
#    python run_classifier.py \
#    --train \
#    --train_data=data/paired_annotations.csv \
#    --dev_split_size=0.2 \
#    --num_train_epochs=10 \
#    --text_cols="${col}_text" \
#    --label_col="${col}_on_task" \
#    --balance_labels
#done

for col in high_uptake focusing_question
do
    python run_classifier.py \
    --train \
    --train_data=data/paired_annotations.csv \
    --dev_split_size=0.2 \
    --num_train_epochs=10 \
    --text_cols=student_text,teacher_text \
    --label_col="${col}" \
    --balance_labels
done

python run_classifier.py \
--train \
--train_data=data/student_reasoning.csv \
--dev_split_size=0.2 \
--num_train_epochs=10 \
--text_cols=text \
--label_col=student_reasoning \
--balance_labels \
--predict \
--predict_data=data/student_reasoning_candidates.csv \
--predict_index_col=comb_idx