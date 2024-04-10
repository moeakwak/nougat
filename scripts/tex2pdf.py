import datetime
import glob
import os
import argparse
import time
from tqdm import tqdm
import subprocess
import json
from multiprocessing import Pool, Manager


class LatexError(Exception):
    def __init__(self, source, message):
        self.source = source
        self.message = message

    def __str__(self):
        return f"{self.source}: {self.message}"


def find_main_tex(folder):
    tex_files = [f for f in os.listdir(folder) if f.endswith(".tex")]
    if not tex_files:
        raise LatexError("find_main_tex", "No .tex files found in the folder.")
    if len(tex_files) == 1:
        return tex_files[0]
    preferred_names = ["main.tex", "paper.tex", "arxiv.tex", "sample.tex"]
    for name in preferred_names:
        if name in tex_files:
            return name
    document_class_files = []
    for tex_file in tex_files:
        with open(os.path.join(folder, tex_file), "r") as f:
            content = f.read()
            if "\\documentclass" in content:
                document_class_files.append(tex_file)
    if len(document_class_files) == 1:
        return document_class_files[0]
    raise LatexError("find_main_tex", "Unable to determine the main .tex file.")


def write_error_log(logs_dir, arxiv_id, error_name, tex_dir, error_detail, lock):
    yymm = arxiv_id[:4]
    log_path = os.path.join(logs_dir, f"{yymm}.jsonl")
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


def export_to_pdf(args):
    tex_dir, pdf_path, logs_dir, lock = args
    arxiv_id = os.path.splitext(os.path.basename(pdf_path))[0]
    try:
        # Get the directory containing main.tex
        main_tex_path = find_main_tex(tex_dir)
        start_time = time.time()

        # Run pdflatex to generate the PDF
        process = subprocess.Popen(
            ["pdflatex", "-interaction=nonstopmode", main_tex_path],
            cwd=tex_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.communicate(timeout=60)  # 设置超时时间为60秒
        except subprocess.TimeoutExpired:
            process.kill()  # 超时后强行终止进程
            raise LatexError("export_to_pdf", "PdfLaTeX process timed out.")

        # Find the generated PDF file by timestamp
        pdf_files = [f for f in os.listdir(tex_dir) if f.endswith(".pdf")]
        expected_pdf_path = None
        for pdf_file in pdf_files:
            file_path = os.path.join(tex_dir, pdf_file)
            if os.path.getmtime(file_path) > start_time:
                expected_pdf_path = file_path
                break
        if expected_pdf_path is None:
            raise LatexError("export_to_pdf", "No newly generated .pdf file found.")
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        os.rename(expected_pdf_path, pdf_path)

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
    parser = argparse.ArgumentParser(description="Export arXiv TeX files to PDF.")
    parser.add_argument("--input", "-I", help="Input directory", default="./data/tex")
    parser.add_argument("--output", "-O", help="Output directory", default="./data/pdf")
    parser.add_argument("--logs_dir", "-L", help="Logs directory", default="./logs")
    parser.add_argument("--skip", "-S", help="Skip existing PDFs", action="store_true")
    parser.add_argument("--start", default="0000_000", help="Start of the range (YYMM_xxx format)")
    parser.add_argument("--end", default="2404_000", help="End of the range (YYMM_xxx format)")
    parser.add_argument("--arxiv_id", help="Process a single arXiv ID")
    parser.add_argument("--workers", "-W", type=int, default=-1, help="Number of worker processes")
    args = parser.parse_args()

    start_parts = args.start.split("_")
    end_parts = args.end.split("_")
    start_yymm = start_parts[0]
    start_xxx = int(start_parts[1])
    end_yymm = end_parts[0]
    end_xxx = int(end_parts[1])

    file_list = []
    if args.arxiv_id:
        tex_dir = os.path.join(args.input, args.arxiv_id[:4], args.arxiv_id)
        if not os.path.exists(tex_dir):
            raise FileNotFoundError(f"Directory for {args.arxiv_id} not found.")
        pdf_path = os.path.join(args.output, args.arxiv_id[:4], args.arxiv_id + ".pdf")
        file_list.append((tex_dir, pdf_path))
    else:
        for yymm in os.listdir(args.input):
            if yymm >= start_yymm and yymm <= end_yymm:
                yymm_dir = os.path.join(args.input, yymm)
                for arxiv_id in os.listdir(yymm_dir):
                    if arxiv_id.endswith(".pdf"):
                        continue
                    tex_dir = os.path.join(yymm_dir, arxiv_id)
                    pdf_path = os.path.join(args.output, yymm, arxiv_id + ".pdf")
                    if args.skip and os.path.exists(pdf_path):
                        continue
                    file_list.append((tex_dir, pdf_path))

    file_list.sort(key=lambda x: x[0])

    with Manager() as manager:
        lock = manager.Lock()
        export_args = [(tex_dir, pdf_path, args.logs_dir, lock) for tex_dir, pdf_path in file_list]

        workers = args.workers
        if workers == -1:
            workers = os.cpu_count()
        with Pool(processes=workers) as pool:
            error_count = 0
            with tqdm(total=len(file_list), unit="file") as pbar:
                for result in pool.imap_unordered(export_to_pdf, export_args):
                    if not result:
                        error_count += 1
                    pbar.update(1)
                    pbar.set_postfix(errors=error_count)

        print(f"Processing completed with {error_count} errors.")


if __name__ == "__main__":
    main()
