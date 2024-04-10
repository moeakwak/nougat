"""
将 arxiv_data/src/yymm/arXiv_src_yymm_xxx.tar 解压到 data/tex/yymm/arxiv_id 中
"""

import os
import argparse
import tarfile
from tqdm import tqdm


def process_tar_file(tar_file, tar_size, output_dir):
    yymm = os.path.basename(tar_file)[10:14]
    with tarfile.open(tar_file, "r") as tar:
        with tqdm(total=tar_size, unit="B", unit_scale=True, leave=False) as pbar:
            for member in tar.getmembers():
                pbar.set_description(os.path.basename(member.name))
                if member.name.endswith(".gz"):
                    arxiv_id = os.path.splitext(os.path.basename(member.name))[0]
                    tar.extract(member, output_dir)  # extract to data/text/YYMM
                    gz_path = os.path.join(output_dir, member.name)
                    folder_path = os.path.join(output_dir, yymm, arxiv_id)
                    # extract gz_file (as tar.gz) into folder_path
                    try:
                        with tarfile.open(gz_path, "r:gz") as gz_tar:
                            os.makedirs(folder_path, exist_ok=True)
                            gz_tar.extractall(folder_path)
                    except tarfile.ReadError:
                        pass
                    # remove gz_file
                    os.remove(gz_path)
                    pbar.update(member.size)
                else:
                    tar.extract(member, output_dir)
                    pbar.update(member.size)


def main():
    parser = argparse.ArgumentParser(description="Process arXiv tar files.")
    parser.add_argument("--input", "-I", help="Input directory", default="/mnt/nas_media/arxiv_data/src")
    parser.add_argument("--output", "-O", help="Output directory", default="./data/tex")
    parser.add_argument("start", help="Start of the range (YYMM_xxx format)")
    parser.add_argument("end", help="End of the range (YYMM_xxx format)")
    args = parser.parse_args()

    start_parts = args.start.split("_")
    end_parts = args.end.split("_")
    start_yymm = start_parts[0]
    start_xxx = int(start_parts[1])
    end_yymm = end_parts[0]
    end_xxx = int(end_parts[1])

    tar_files = []
    print(f"Processing files from {args.start} to {args.end}")
    for filename in os.listdir(args.input):
        if filename.startswith("arXiv_src_") and filename.endswith(".tar"):
            yymm_xxx = filename[10:-4]
            yymm = yymm_xxx[:4]
            xxx = int(yymm_xxx[5:])
            if (yymm > start_yymm or (yymm == start_yymm and xxx >= start_xxx)) and (yymm < end_yymm or (yymm == end_yymm and xxx <= end_xxx)):
                tar_files.append(os.path.join(args.input, filename))

    total_files = len(tar_files)
    total_size = 0
    with tqdm(total=total_files) as pbar:
        for tar_file in tar_files:
            pbar.set_description("total | " + os.path.basename(os.path.basename(tar_file)))
            tar_size = os.path.getsize(tar_file)
            process_tar_file(tar_file, tar_size, args.output)
            pbar.update(1)
            total_size += tar_size
    print(f"Processed {total_files} files with total size {total_size / 1024 / 1024:.2f} MB.")


if __name__ == "__main__":
    main()
