import datetime
import json
import os


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


def get_file_list(args, out_prefix):
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
        out_path = os.path.join(args.output, args.arxiv_id[:4], args.arxiv_id + out_prefix)
        file_list.append((tex_dir, out_path))
    else:
        for yymm in os.listdir(args.input):
            if yymm >= start_yymm and yymm <= end_yymm:
                yymm_dir = os.path.join(args.input, yymm)
                for arxiv_id in os.listdir(yymm_dir):
                    if arxiv_id.endswith(".pdf"):
                        continue
                    tex_dir = os.path.join(yymm_dir, arxiv_id)
                    if not os.path.isdir(tex_dir):
                        continue
                    out_path = os.path.join(args.output, yymm, arxiv_id + out_prefix)
                    if args.skip and os.path.exists(out_path):
                        continue
                    file_list.append((tex_dir, out_path))
    file_list.sort(key=lambda x: x[0])
    return file_list