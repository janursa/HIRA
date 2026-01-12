#!/bin/bash

BASE_DIR="/Users/jno24/Documents/projs/ongoing/hiara/base_folder"

echo "=================================================="
echo "Dataset File Renaming Script"
echo "=================================================="
echo ""

# Function to rename a single file
rename_file() {
    local dir=$1
    local old_name=$2
    local new_name=$3
    local pattern=$4
    
    for file in "$dir"/${old_name}${pattern}; do
        if [ -f "$file" ]; then
            new_file="${file//${old_name}/${new_name}}"
            echo "  Renaming: $(basename "$file") -> $(basename "$new_file")"
            mv "$file" "$new_file"
        fi
    done
}

# Function to rename files in a directory
rename_files_in_dir() {
    local dir=$1
    local pattern=$2
    
    if [ ! -d "$dir" ]; then
        echo "⚠️  Directory not found: $dir"
        return
    fi
    
    echo "Processing directory: $dir"
    
    rename_file "$dir" "data1" "onek1k" "$pattern"
    rename_file "$dir" "data7_allTPs_jalil" "abf300" "$pattern"
    rename_file "$dir" "data12" "zhang" "$pattern"
    rename_file "$dir" "data13_Korean" "aida_korean" "$pattern"
    rename_file "$dir" "data13_Japanese" "aida_japanese" "$pattern"
    
    echo ""
}

# Rename files in different directories
rename_files_in_dir "$BASE_DIR/datasets/bulk" "_bulk.h5ad"
rename_files_in_dir "$BASE_DIR/datasets/sc" "_sc.h5ad"
rename_files_in_dir "$BASE_DIR/prior" "_*.h5ad"

# Also check for any other file patterns that might exist
for subdir in "$BASE_DIR/datasets"/*; do
    if [ -d "$subdir" ]; then
        rename_files_in_dir "$subdir" "*"
    fi
done

echo "=================================================="
echo "✓ Renaming complete!"
echo "=================================================="
