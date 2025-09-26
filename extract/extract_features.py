import sys
import os
from datasets import load_dataset
from call_api_local.call_api import call_gpt4
import random
import json
import concurrent.futures
import json
import os
import threading
from datetime import datetime
from utils.file_operation import jsonl2json, sort_jsonl_file



lock = threading.Lock()
random.seed(0)
DEFAULT_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rb",
    ".rs",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".ps1",
    ".sql",
    # ".json",
    # ".yaml",
    # ".yml",
    # ".toml",
    # ".ini",
    # ".cfg",
    # ".txt",
    # ".md"
}
def get_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def get_full_content(d):
    keys=["full_content", "instruction", "output", "prompt", "text", "code"]
    content=""
    for k in keys:
        if k in d.keys():
            content+=str(d[k])
    return content

def get_data_from_file(file_path):
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data=json.load(f)
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
    for d in data:
        d["full_content"]=get_full_content(d)
    return data

# def get_data_from_file(file_path):
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             json_line = json.loads(line.strip())
#             keys_to_keep = ["full_content"]
#             filtered_data = {key: json_line[key] for key in keys_to_keep if key in json_line}
#             data.append(filtered_data)
#     return data


def get_source_data(file_paths):
    data = []
    for file_path in file_paths:
        data.extend(get_data_from_file(file_path))
    return data


def extract_fields_from_formatted_text(text):
    """
    Extracts fields from a formatted text.

    Parameters:
    - text (str): Multiline string containing the formatted text.

    Returns:
    - dict: A dictionary where each key is a category name and the value is a list of items extracted from that category.
    """
    import re

    json_start = text.find("<begin>") + len("<begin>")
    json_end = text.find("<end>")
    json_string = text[json_start:json_end].strip()
    json_string = json_string.replace("\n", "").replace("\r", "")
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(e)
        data = {"parse_error_str": json_string}

    return data


def extract_single_feature(d, idx, base_prompt, save_path):
    try:
        code = d["full_content"]
        prompt = base_prompt.replace("{source_code}", code)
        output, response = call_gpt4(prompt, model='gpt-4')

        if idx < 10:
            print(f"prompt=\n{prompt}")
            print(f"output=\n{output}")

        extract_output = extract_fields_from_formatted_text(output)
        assert len(extract_output) > 0, f"Extract output failed. Output: {output}"

        with lock:
            partial_data = {
                "idx": idx,
                "original_code": code,
                "features": extract_output
            }
            with open(save_path, 'a', encoding='utf-8') as file:
                json.dump(partial_data, file)
                file.write('\n')

        if idx % 100 == 0:
            with lock:
                with open(save_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                data = [json.loads(line) for line in lines]
                sorted_data = sorted(data, key=lambda x: x['idx'])

                with open(save_path, 'w', encoding='utf-8') as file:
                    for entry in sorted_data:
                        json.dump(entry, file)
                        file.write('\n')
        return partial_data
    except Exception as e:
        print(f"Error processing idx {idx}: {e}")
        return None


def extract_features(base_prompt, seed_data_paths, end_idx, output_dir='./output',begin_idx=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset = get_source_data(seed_data_paths)
    save_path = f"{output_dir}/extract_{begin_idx}-{end_idx}.jsonl"
    ext_data = []
    print(f"end_idx={end_idx}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, d in enumerate(dataset):
            if i < begin_idx:
                continue
            if i >= end_idx:
                break
            print(f"submit{i}")
            futures.append(executor.submit(extract_single_feature, d, i, base_prompt, save_path))

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                ext_data.append(result)
    sort_jsonl_file(save_path)
    jsonl2json(save_path)


def load_code_files_from_directory(
    directory: str,
    allowed_extensions: set[str] | None = None,
    encoding: str = "utf-8",
) -> list[dict]:
    """遍历目录，读取所有符合后缀条件的代码文件并返回标准化样本列表。"""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"目录不存在: {directory}")

    extensions = allowed_extensions or DEFAULT_CODE_EXTENSIONS
    samples: list[dict] = []
    for root, _dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            _, ext = os.path.splitext(file_name)
            if ext.lower() not in extensions:
                continue
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            samples.append(
                {
                    "path": os.path.relpath(file_path, directory),
                    "full_content": content,
                }
            )
    return samples


def extract_features_from_dir(
    directory: str,
    base_prompt: str,
    output_dir: str,
    allowed_extensions: set[str] | None = None,
    begin_idx: int = 0,
    end_idx: int | None = None,
    encoding: str = "utf-8",
):
    """将目录下的代码文件视作样本输入，复用 extract_features 逻辑提取特征。"""
    samples = load_code_files_from_directory(
        directory,
        allowed_extensions=allowed_extensions,
        encoding=encoding,
    )
    if not samples:
        raise ValueError(f"目录 {directory} 中未发现可用代码文件")

    # 将 samples 写入临时 jsonl 以复用现有流程
    tmp_jsonl_path = os.path.join(output_dir, "_tmp_dir_samples.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(tmp_jsonl_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            record = {
                "idx": idx,
                "path": sample.get("path"),
                "full_content": sample.get("full_content", ""),
            }
            json.dump(record, f)
            f.write("\n")

    try:
        end_idx = end_idx if end_idx is not None else len(samples)
        extract_features(
            base_prompt=base_prompt,
            seed_data_paths=[tmp_jsonl_path],
            end_idx=end_idx,
            output_dir=output_dir,
            begin_idx=begin_idx,
        )
    finally:
        try:
            os.remove(tmp_jsonl_path)
        except OSError:
            pass


if __name__ == "__main__":
    root_dir = "."
    prompt_idx = 12
    prompt_file = f"{root_dir}/prompt/extract/prompt{prompt_idx}.txt"
    base_prompt = get_text(prompt_file)

    source_directory = "/home/yangyang/workspace/fclib/src"
    data_name = "workspace_fclib"
    output_dir = f"{root_dir}/output/extract/{data_name}/prompt{prompt_idx}/features"
    os.makedirs(output_dir, exist_ok=True)

    extract_features_from_dir(
        directory=source_directory,
        base_prompt=base_prompt,
        output_dir=output_dir,
        begin_idx=0,
        end_idx=100000,
    )
