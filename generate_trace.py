import sys
from copy import deepcopy

def generate_trace_file (file_path: str, out_dir: str, batch_size: int):

    # read raw trace file
    with open(file_path, 'r') as f:
        data = f.readlines()

    outer_list = []
    inner_list = []

    for line in data:
        if line.startswith('Begin'):
            continue;
        if line.startswith('end'):
            outer_list.append(deepcopy(inner_list))
            inner_list = []
            continue;
        inner_list.append(line)

    for i, inn in enumerate(outer_list[1:]):
        with open(f'data/traces/resnet/{i}_batch_{batch_size}.tsv', 'w') as f:
            f.writelines(inn)

if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print("Script requires 3 arguments: input file path, output dir path and batch size")

    generate_trace_file(sys.argv[1], sys.argv[2], sys.argv[3])
    