import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("test_output_file", type=str, default=None)
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()

    test_output_lines = open(args.test_output_file, 'r', encoding='utf8').readlines()

    (file, ext) = os.path.splitext(args.test_output_file)

    output_path = file + "_clean" + ext

    with open(output_path, 'w', encoding='utf8') as writer:
        for line in test_output_lines:
            writer.write(line[:line.find('.')].strip()+'.\n')

if __name__ == "__main__":
    main()
