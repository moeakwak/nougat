export target="standard"
python -m nougat.dataset.create_index --dir data/paired_$target --out data/index_$target.jsonl
python split_dataset.py -I data/index_$target.jsonl -O data/$target -P 90:5:5 --shuffle
python -m nougat.dataset.gen_seek data/$target/train.jsonl data/$target/test.jsonl data/$target/validation.jsonl