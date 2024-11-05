import os

def print_directory_tree(root_dir, indent=""):
    # 列出目錄中的所有檔案和資料夾
    items = os.listdir(root_dir)
    for index, item in enumerate(items):
        path = os.path.join(root_dir, item)
        is_last = index == len(items) - 1
        # 根據是否是最後一個項目來選擇不同的符號
        prefix = "└── " if is_last else "├── "
        print(indent + prefix + item)
        # 如果是資料夾，遞迴調用並增加縮排
        if os.path.isdir(path):
            new_indent = indent + ("    " if is_last else "│   ")
            print_directory_tree(path, new_indent)

# 使用當前目錄
if __name__ == "__main__":
    print_directory_tree(".")