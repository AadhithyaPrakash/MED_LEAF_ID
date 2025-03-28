import os

repo_path = r"D:\MED_LEAF_ID"
ignore_dirs = {"base", ".git"}
data_folders = {
    "D:\\MED_LEAF_ID\\data\\augmented",
    "D:\\MED_LEAF_ID\\data\\cnn",
    "D:\\MED_LEAF_ID\\data\\cnn\\augmented",
    "D:\\MED_LEAF_ID\\data\\cnn\\original",
    "D:\\MED_LEAF_ID\\data\\preprocessed_glcm",
}
data_file = "D:\\MED_LEAF_ID\\data\\glcm_features.csv"
dataset_folder = (
    r"D:\MED_LEAF_ID\dataset\Medicinal Leaf dataset"  # Define dataset directory
)

output_file = "repo_structure.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for root, dirs, files in os.walk(repo_path):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        # Hide all subdirectories inside "dataset/Medicinal Leaf dataset"
        if root == dataset_folder:
            f.write(f"\n📂 Directory: {root}/ (Only showing folder name)\n")
            dirs.clear()  # Prevent subdirectories from being listed
            continue

        # Show only specific folders in data/
        elif root in data_folders or root == os.path.dirname(data_file):
            dirs[:] = [d for d in dirs if os.path.join(root, d) in data_folders]
            files = [file for file in files if os.path.join(root, file) == data_file]
            f.write(f"\n📂 Directory: {root}/ (Filtered view)\n")

        # Standard listing for everything else
        else:
            f.write(f"\n📂 Directory: {root}\n")

        for d in dirs:
            f.write(f"   📁 {d}/\n")
        for file in files:
            f.write(f"   ├── {file}\n")

print(f"✅ Repo structure saved to {output_file}")
print("📁 Directory structure:" )