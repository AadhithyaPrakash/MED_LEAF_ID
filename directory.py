import os

def write_directory_structure(base_path, output_file):
    def should_exclude_dir(dirname):
      
        return dirname in {'.git', 'base'}

    def format_dir_name(dirname):
        
        if dirname in {'dataset', 'data'}:
            return f"{dirname}/[...]"
        return dirname

    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(base_path):
           
            dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
           
            level = root.replace(base_path, '').count(os.sep)
            indent = ' ' * 4 * level
            dir_name = os.path.basename(root)
            if dir_name:  
                f.write(f"{indent}{format_dir_name(dir_name)}/\n")


base_path = 'D:\MED_LEAF_ID-1'
output_file = 'D:\MED_LEAF_ID-1\directory.txt'
write_directory_structure(base_path, output_file)