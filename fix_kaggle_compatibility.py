"""
Fix compatibility issue for Kaggle notebooks.
Updates pip install to use compatible versions of numpy, scipy, albumentations.
"""
import json

print("Fixing compatibility issues in notebooks...")

# Fix both notebooks
for notebook_file in ['notebooks/train_optimized.ipynb', 'notebooks/train_with_visualizations.ipynb']:
    with open(notebook_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    fixed = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if '!pip install' in source and 'albumentations' in source:
                # Fix: Downgrade albumentations to compatible version, fix scipy
                cell['source'] = [
                    "# Install required packages with compatible versions\n",
                    "# Note: albumentations 1.3.x is more stable on Kaggle\n",
                    "!pip install -q numpy==1.24.3\n",
                    "!pip install -q scipy==1.11.4\n", 
                    "!pip install -q albumentations==1.3.1\n",
                    "!pip install -q timm onnx onnxruntime seaborn scikit-learn"
                ]
                fixed = True
                break
    
    if fixed:
        with open(notebook_file, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Fixed {notebook_file}")
    else:
        print(f"  - {notebook_file} (no changes needed)")

print("\n✅ Fixed version compatibility!")
print("   numpy==1.24.3")
print("   scipy==1.11.4") 
print("   albumentations==1.3.1")
