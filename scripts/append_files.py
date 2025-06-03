import sys
import os

def write_file_contents_to_output(file_paths, output_path="scripts/out.txt"):
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            out_file.write(f"=== {file_name} ====\n")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    out_file.write(f.read())
            except Exception as e:
                out_file.write(f"Error reading {file_path}: {e}")
            out_file.write("\n\n")  # Spacing between file sections

        out_file.write("=== Your task ====\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file1> <file2> ... <fileN>")
    else:
        write_file_contents_to_output(sys.argv[1:])
        print("Output written to out.txt")
