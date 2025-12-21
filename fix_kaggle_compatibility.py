"""
Fix Kaggle compatibility by proper installation order.
Must reinstall sklearn AFTER numpy to fix binary compatibility.
"""
import json

print("Fixing Kaggle compatibility with proper install order...")

# Fix both notebooks
for notebook_file in ['notebooks/train_optimized.ipynb', 'notebooks/train_with_visualizations.ipynb']:
    with open(notebook_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    fixed = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if '!pip install' in source and ('albumentations' in source or 'numpy' in source):
                # Fix: Use --force-reinstall for sklearn after numpy change
                cell['source'] = [
                    "# Install packages with proper order to avoid binary incompatibility\n",
                    "# Step 1: Upgrade numpy first\n",
                    "!pip install -q --upgrade numpy\n",
                    "\n",
                    "# Step 2: Force reinstall sklearn to match new numpy\n",
                    "!pip install -q --force-reinstall scikit-learn\n",
                    "\n",
                    "# Step 3: Install other packages\n",
                    "!pip install -q timm albumentations onnx onnxruntime seaborn"
                ]
                fixed = True
                break
    
    if fixed:
        with open(notebook_file, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Fixed {notebook_file}")
    else:
        print(f"  - {notebook_file} (no changes needed)")

print("\n✅ Fixed installation order!")
print("   1. Upgrade numpy first")
print("   2. Force reinstall scikit-learn (fixes binary compatibility)")
print("   3. Install other packages")
print("\n⚠️ IMPORTANT: After running Step 1 & 2, RESTART KERNEL before importing!")
