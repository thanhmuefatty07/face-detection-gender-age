"""
Sync train_with_visualizations.ipynb from train_optimized.ipynb
Ensures 100% accuracy for report by copying training sections exactly
and adding only approved visualization sections.
"""
import json

print("Reading original train_optimized.ipynb...")
with open('notebooks/train_optimized.ipynb', 'r', encoding='utf-8') as f:
    original = json.load(f)

print(f"Original has {len(original['cells'])} cells")

# Copy all original cells
new_nb = {
    'cells': original['cells'].copy(),
    'metadata': original['metadata'].copy(),
    'nbformat': original['nbformat'],
    'nbformat_minor': original['nbformat_minor']
}

# Fix pip install to include seaborn and sklearn (required for visualizations)
for cell in new_nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '!pip install' in source and 'timm' in source:
            cell['source'] = [
                "# Install required packages\n",
                "!pip install -q timm albumentations onnx onnxruntime seaborn scikit-learn"
            ]
            print("Added seaborn and sklearn to pip install")
            break

# Add visualization sections that use data from training
visualization_cells = [
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 14. Confusion Matrix Visualization']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'import seaborn as sns\n',
        'from sklearn.metrics import confusion_matrix, classification_report\n',
        '\n',
        'def plot_confusion_matrices(model, val_loader, device):\n',
        '    """Generate confusion matrices for gender and age"""\n',
        '    model.eval()\n',
        '    all_gender_preds, all_gender_true = [], []\n',
        '    all_age_preds, all_age_true = [], []\n',
        '    \n',
        '    with torch.no_grad():\n',
        '        for images, genders, age_groups, _ in tqdm(val_loader, desc="Evaluating"):\n',
        '            images = images.to(device)\n',
        '            pred_gender, pred_age = model(images)\n',
        '            all_gender_preds.extend(pred_gender.argmax(1).cpu().numpy())\n',
        '            all_gender_true.extend(genders.numpy())\n',
        '            all_age_preds.extend(pred_age.argmax(1).cpu().numpy())\n',
        '            all_age_true.extend(age_groups.numpy())\n',
        '    \n',
        '    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n',
        '    \n',
        '    # Gender\n',
        '    cm_gender = confusion_matrix(all_gender_true, all_gender_preds)\n',
        '    sns.heatmap(cm_gender, annot=True, fmt="d", cmap="Blues", ax=axes[0],\n',
        '                xticklabels=["Male", "Female"], yticklabels=["Male", "Female"])\n',
        '    axes[0].set_title("Gender Confusion Matrix", fontsize=14)\n',
        '    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")\n',
        '    \n',
        '    # Age\n',
        '    cm_age = confusion_matrix(all_age_true, all_age_preds)\n',
        '    sns.heatmap(cm_age, annot=True, fmt="d", cmap="Oranges", ax=axes[1],\n',
        '                xticklabels=cfg.AGE_GROUPS, yticklabels=cfg.AGE_GROUPS)\n',
        '    axes[1].set_title("Age Group Confusion Matrix", fontsize=14)\n',
        '    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")\n',
        '    axes[1].tick_params(axis="x", rotation=45)\n',
        '    \n',
        '    plt.tight_layout()\n',
        '    plt.savefig(f"{cfg.OUTPUT_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")\n',
        '    plt.show()\n',
        '    \n',
        '    print("\\nGENDER REPORT:")\n',
        '    print(classification_report(all_gender_true, all_gender_preds, target_names=["Male", "Female"]))\n',
        '    print("\\nAGE GROUP REPORT:")\n',
        '    print(classification_report(all_age_true, all_age_preds, target_names=cfg.AGE_GROUPS))\n',
        '    \n',
        '    return all_gender_preds, all_gender_true, all_age_preds, all_age_true\n',
        '\n',
        '# Run\n',
        '_, val_dataset = create_datasets()\n',
        'val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)\n',
        'gender_preds, gender_true, age_preds, age_true = plot_confusion_matrices(model, val_loader, device)'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 15. Age Distribution Analysis']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'def plot_age_distribution(age_true, age_preds):\n',
        '    """Plot age distribution with 3 subplots"""\n',
        '    fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n',
        '    \n',
        '    # 1. Dataset distribution\n',
        '    axes[0].bar(range(len(cfg.AGE_GROUPS)), np.bincount(age_true, minlength=10), color="steelblue")\n',
        '    axes[0].set_xticks(range(len(cfg.AGE_GROUPS)))\n',
        '    axes[0].set_xticklabels(cfg.AGE_GROUPS, rotation=45, ha="right")\n',
        '    axes[0].set_title("Dataset Age Distribution")\n',
        '    axes[0].set_xlabel("Age Group"); axes[0].set_ylabel("Count")\n',
        '    \n',
        '    # 2. Predicted vs Actual\n',
        '    x = np.arange(len(cfg.AGE_GROUPS))\n',
        '    width = 0.35\n',
        '    axes[1].bar(x - width/2, np.bincount(age_true, minlength=10), width, label="Actual", color="steelblue")\n',
        '    axes[1].bar(x + width/2, np.bincount(age_preds, minlength=10), width, label="Predicted", color="coral")\n',
        '    axes[1].set_xticks(x)\n',
        '    axes[1].set_xticklabels(cfg.AGE_GROUPS, rotation=45, ha="right")\n',
        '    axes[1].set_title("Predicted vs Actual")\n',
        '    axes[1].legend()\n',
        '    \n',
        '    # 3. Error distribution\n',
        '    errors = np.array(age_preds) - np.array(age_true)\n',
        '    axes[2].hist(errors, bins=range(-5, 6), color="salmon", edgecolor="black")\n',
        '    axes[2].axvline(x=0, color="green", linestyle="--", linewidth=2)\n',
        '    axes[2].set_title("Prediction Error")\n',
        '    axes[2].set_xlabel("Error (groups)")\n',
        '    \n',
        '    plt.tight_layout()\n',
        '    plt.savefig(f"{cfg.OUTPUT_DIR}/age_distribution.png", dpi=150, bbox_inches="tight")\n',
        '    plt.show()\n',
        '    \n',
        '    print(f"Age Exact: {np.mean(np.array(age_preds)==np.array(age_true))*100:.1f}%")\n',
        '    print(f"Age ±1: {np.mean(np.abs(errors)<=1)*100:.1f}%")\n',
        '    print(f"MAE: {np.mean(np.abs(errors)):.2f} groups")\n',
        '\n',
        'plot_age_distribution(age_true, age_preds)'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 16. Per-Class Accuracy Report']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'def per_class_report(gender_true, gender_preds, age_true, age_preds):\n',
        '    g_true, g_pred = np.array(gender_true), np.array(gender_preds)\n',
        '    a_true, a_pred = np.array(age_true), np.array(age_preds)\n',
        '    \n',
        '    print("GENDER BREAKDOWN:")\n',
        '    print(f"  Male: {np.mean(g_pred[g_true==0]==0)*100:.1f}% (n={np.sum(g_true==0)})")\n',
        '    print(f"  Female: {np.mean(g_pred[g_true==1]==1)*100:.1f}% (n={np.sum(g_true==1)})")\n',
        '    \n',
        '    print("\\nAGE BREAKDOWN:")\n',
        '    results = []\n',
        '    for i, group in enumerate(cfg.AGE_GROUPS):\n',
        '        mask = a_true == i\n',
        '        if np.sum(mask) > 0:\n',
        '            exact = np.mean(a_pred[mask] == i) * 100\n',
        '            within1 = np.mean(np.abs(a_pred[mask] - i) <= 1) * 100\n',
        '            print(f"  {group}: {exact:.1f}% exact, {within1:.1f}% ±1 (n={np.sum(mask)})")\n',
        '            results.append({"Group": group, "N": np.sum(mask), "Exact": exact, "±1": within1})\n',
        '    \n',
        '    pd.DataFrame(results).to_csv(f"{cfg.OUTPUT_DIR}/per_class_accuracy.csv", index=False)\n',
        '    print(f"\\nSaved: per_class_accuracy.csv")\n',
        '\n',
        'per_class_report(gender_true, gender_preds, age_true, age_preds)'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 17. Sample Predictions Grid']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'def show_samples(model, val_dataset, device, n=16):\n',
        '    model.eval()\n',
        '    fig, axes = plt.subplots(4, 4, figsize=(14, 14))\n',
        '    np.random.seed(42)\n',
        '    indices = np.random.choice(len(val_dataset), n, replace=False)\n',
        '    \n',
        '    for idx, ax in zip(indices, axes.flatten()):\n',
        '        img, g_true, a_true, _ = val_dataset[idx]\n',
        '        with torch.no_grad():\n',
        '            g_out, a_out = model(img.unsqueeze(0).to(device))\n',
        '            g_pred, a_pred = g_out.argmax(1).item(), a_out.argmax(1).item()\n',
        '            g_conf = F.softmax(g_out, dim=1).max().item()\n',
        '            a_conf = F.softmax(a_out, dim=1).max().item()\n',
        '        \n',
        '        img_np = img.numpy().transpose(1,2,0)\n',
        '        img_np = img_np * [0.229,0.224,0.225] + [0.485,0.456,0.406]\n',
        '        ax.imshow(np.clip(img_np, 0, 1))\n',
        '        \n',
        '        ok = g_pred==g_true and a_pred==a_true\n',
        '        color = "green" if ok else "red"\n',
        '        ax.set_title(f"{[\'M\',\'F\'][g_pred]} ({g_conf:.0%})\\n{cfg.AGE_GROUPS[a_pred]} ({a_conf:.0%})", color=color, fontsize=10)\n',
        '        for spine in ax.spines.values():\n',
        '            spine.set_edgecolor(color)\n',
        '            spine.set_linewidth(3)\n',
        '        ax.set_xticks([]); ax.set_yticks([])\n',
        '    \n',
        '    plt.suptitle("Sample Predictions (Green=Correct, Red=Wrong)", fontsize=14)\n',
        '    plt.tight_layout()\n',
        '    plt.savefig(f"{cfg.OUTPUT_DIR}/sample_predictions.png", dpi=150, bbox_inches="tight")\n',
        '    plt.show()\n',
        '\n',
        'show_samples(model, val_dataset, device)'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 18. Model Architecture Summary']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'def model_info(model):\n',
        '    total = sum(p.numel() for p in model.parameters())\n',
        '    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n',
        '    \n',
        '    print("MODEL SUMMARY:")\n',
        '    print(f"  Total params: {total:,}")\n',
        '    print(f"  Trainable: {trainable:,}")\n',
        '    print(f"  Size (FP32): {total*4/1e6:.2f} MB")\n',
        '    print(f"  Size (INT8): {total/1e6:.2f} MB")\n',
        '    \n',
        '    print("\\nLAYER BREAKDOWN:")\n',
        '    for name, m in model.named_children():\n',
        '        p = sum(x.numel() for x in m.parameters())\n',
        '        print(f"  {name}: {p:,}")\n',
        '    \n',
        '    print(f"\\nCompression vs Teacher: {21000000/total:.1f}x smaller")\n',
        '\n',
        'model_info(model)'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 19. Final Summary Table']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'def final_summary(model, history, g_true, g_pred, a_true, a_pred):\n',
        '    g_acc = np.mean(np.array(g_pred)==np.array(g_true))*100\n',
        '    a_exact = np.mean(np.array(a_pred)==np.array(a_true))*100\n',
        '    a_pm1 = np.mean(np.abs(np.array(a_pred)-np.array(a_true))<=1)*100\n',
        '    mae = np.mean(np.abs(np.array(a_pred)-np.array(a_true)))\n',
        '    total = sum(p.numel() for p in model.parameters())\n',
        '    \n',
        '    data = {\n',
        '        "Metric": ["Gender Acc", "Age Exact", "Age ±1", "Age MAE", "Params", "Size", "Backbone", "Input"],\n',
        '        "Value": [f"{g_acc:.1f}%", f"{a_exact:.1f}%", f"{a_pm1:.1f}%", f"{mae:.2f}", \n',
        '                  f"{total:,}", f"{total*4/1e6:.2f} MB", cfg.STUDENT_BACKBONE, f"{cfg.IMG_SIZE}x{cfg.IMG_SIZE}"]\n',
        '    }\n',
        '    df = pd.DataFrame(data)\n',
        '    print(df.to_string(index=False))\n',
        '    df.to_csv(f"{cfg.OUTPUT_DIR}/final_summary.csv", index=False)\n',
        '    print("\\nSaved: final_summary.csv")\n',
        '\n',
        'final_summary(model, history, gender_true, gender_preds, age_true, age_preds)'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 📦 All Output Files']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'print("OUTPUT FILES:")\n',
        'files = ["best_model.pth", "gender_age_optimized.onnx", "gender_age_optimized_int8.onnx",\n',
        '         "training_history.png", "confusion_matrices.png", "age_distribution.png",\n',
        '         "sample_predictions.png", "per_class_accuracy.csv", "final_summary.csv"]\n',
        'for f in files:\n',
        '    path = os.path.join(cfg.OUTPUT_DIR, f)\n',
        '    if os.path.exists(path):\n',
        '        print(f"  ✓ {f}: {os.path.getsize(path)/1e6:.2f} MB")\n',
        '    else:\n',
        '        print(f"  ✗ {f}: MISSING")\n',
        'print("\\nDownload these for your report!")'
    ]}
]

new_nb['cells'].extend(visualization_cells)

# Save
with open('notebooks/train_with_visualizations.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, indent=2, ensure_ascii=False)

print(f"\n✅ Synced train_with_visualizations.ipynb")
print(f"   Training sections: {len(original['cells'])} cells (from original)")
print(f"   Visualization sections: {len(visualization_cells)} cells (added)")
print(f"   Total: {len(new_nb['cells'])} cells")
print("\n📊 REQUIRED DATASETS ON KAGGLE:")
print("   1. utkface-new - UTKFace dataset for training")
print("      Path: /kaggle/input/utkface-new/UTKFace")
print("   2. gender-age-model - Teacher model checkpoint")
print("      Path: /kaggle/input/gender-age-model/gender_age_model.pth")
