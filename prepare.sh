python -m nougat.dataset.create_index --dir data/paired --out data/index.jsonl
python split_dataset.py -I data/index_standard.jsonl -O data -P 90:5:5 --shuffle
python -m nougat.dataset.gen_seek data/train.jsonl data/test.jsonl data/validation.jsonl