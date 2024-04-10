import datetime
import os
import argparse
import time
from tqdm import tqdm
import subprocess
import json
from multiprocessing import Pool, Manager

from common import LatexError, find_main_tex, get_file_list


def write_error_log(logs_dir, arxiv_id, error_name, tex_dir, error_detail, lock):
    yymm = arxiv_id[:4]
    log_path = os.path.join(logs_dir, "html", f"{yymm}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_entry = {
        "arxiv_id": arxiv_id,
        "datetime": datetime.datetime.now().strftime("%Y%m%d %H:%M:%S"),
        "source": error_name,
        "tex_dir": tex_dir,
        "detail": error_detail,
    }
    with lock:
        with open(log_path, "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")


def export_to_html(args):
    tex_dir, html_path, logs_dir, lock = args
    arxiv_id = os.path.splitext(os.path.basename(html_path))[0]
    try:
        main_tex_file = find_main_tex(tex_dir)
        main_tex_path = os.path.join(tex_dir, main_tex_file)

        process = subprocess.Popen(
            ["latexml", "--dest=" + html_path, main_tex_path],
            # cwd=tex_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            raise LatexError("export_to_html", "LaTeXML process timed out.")

        if not os.path.exists(html_path):
            raise LatexError("export_to_html", "No generated .html file found.")

        return True
    except LatexError as e:
        write_error_log(
            logs_dir,
            arxiv_id,
            e.source,
            tex_dir,
            e.message,
            lock,
        )
        return False


def main():
    parser = argparse.ArgumentParser(description="Export arXiv TeX files to HTML.")
    parser.add_argument("--input", "-I", help="Input directory", default="./data/tex")
    parser.add_argument("--output", "-O", help="Output directory", default="./data/html")
    parser.add_argument("--logs_dir", "-L", help="Logs directory", default="./logs")
    parser.add_argument("--skip", "-S", help="Skip existing HTML files", action="store_true")
    parser.add_argument("--start", default="0000_000", help="Start of the range (YYMM_xxx format)")
    parser.add_argument("--end", default="2404_000", help="End of the range (YYMM_xxx format)")
    parser.add_argument("--arxiv_id", help="Process a single arXiv ID")
    parser.add_argument("--workers", "-W", type=int, default=-1, help="Number of worker processes")
    args = parser.parse_args()

    file_list = get_file_list(args, ".html")

    with Manager() as manager:
        lock = manager.Lock()
        export_args = [(tex_dir, html_path, args.logs_dir, lock) for tex_dir, html_path in file_list]

        workers = args.workers
        if workers == -1:
            workers = os.cpu_count()
        with Pool(processes=workers) as pool:
            error_count = 0
            with tqdm(total=len(file_list), unit="file") as pbar:
                for result in pool.imap_unordered(export_to_html, export_args):
                    if not result:
                        error_count += 1
                    pbar.update(1)
                    pbar.set_postfix(errors=error_count)

        print(f"Processing completed with {error_count} errors.")


if __name__ == "__main__":
    main()
