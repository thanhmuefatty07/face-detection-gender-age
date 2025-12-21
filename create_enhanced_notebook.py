"""Script to create enhanced notebook with visualizations"""
import json

# Read original notebook
with open('notebooks/train_optimized.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New cells to add
new_cells = [
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 14. Confusion Matrix Visualization']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'import seaborn as sns\n',
        'from sklearn.metrics import confusion_matrix, classification_report\n',
        '\n',
        'def plot_confusion_matrices(model, val_loader, device):\n',
        '    model.eval()\n',
        '    all_gender_preds, all_gender_true = [], []\n',
        '    all_age_preds, all_age_true = [], []\n',
        '    with torch.no_grad():\n',
        '        for images, genders, age_groups, _ in tqdm(val_loader):\n',
        '            images = images.to(device)\n',
        '            pred_gender, pred_age = model(images)\n',
        '            all_gender_preds.extend(pred_gender.argmax(1).cpu().numpy())\n',
        '            all_gender_true.extend(genders.numpy())\n',
        '            all_age_preds.extend(pred_age.argmax(1).cpu().numpy())\n',
        '            all_age_true.extend(age_groups.numpy())\n',
        '    \n',
        '    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n',
        '    cm_gender = confusion_matrix(all_gender_true, all_gender_preds)\n',
        '    sns.heatmap(cm_gender, annot=True, fmt="d", cmap="Blues", ax=axes[0])\n',
        '    axes[0].set_title("Gender Confusion Matrix")\n',
        '    cm_age = confusion_matrix(all_age_true, all_age_preds)\n',
        '    sns.heatmap(cm_age, annot=True, fmt="d", cmap="Oranges", ax=axes[1])\n',
        '    axes[1].set_title("Age Group Confusion Matrix")\n',
        '    plt.savefig(f"{cfg.OUTPUT_DIR}/confusion_matrices.png", dpi=150)\n',
        '    plt.show()\n',
        '    print(classification_report(all_gender_true, all_gender_preds, target_names=["Male", "Female"]))\n',
        '    print(classification_report(all_age_true, all_age_preds, target_names=cfg.AGE_GROUPS))\n',
        '    return all_gender_preds, all_gender_true, all_age_preds, all_age_true\n',
        '\n',
        '_, val_dataset = create_datasets()\n',
        'val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)\n',
        'gender_preds, gender_true, age_preds, age_true = plot_confusion_matrices(model, val_loader, device)'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 15. Age Distribution Analysis']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'def plot_age_distribution(age_true, age_preds):\n',
        '    fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n',
        '    axes[0].bar(range(len(cfg.AGE_GROUPS)), np.bincount(age_true, minlength=10))\n',
        '    axes[0].set_xticklabels(cfg.AGE_GROUPS, rotation=45)\n',
        '    axes[0].set_title("Dataset Distribution")\n',
        '    errors = np.array(age_preds) - np.array(age_true)\n',
        '    axes[2].hist(errors, bins=range(-5, 6), color="salmon")\n',
        '    axes[2].set_title("Error Distribution")\n',
        '    plt.savefig(f"{cfg.OUTPUT_DIR}/age_distribution.png", dpi=150)\n',
        '    plt.show()\n',
        '    print(f"Age Exact: {np.mean(np.array(age_preds)==np.array(age_true))*100:.1f}%")\n',
        '    print(f"Age +-1: {np.mean(np.abs(errors)<=1)*100:.1f}%")\n',
        '\n',
        'plot_age_distribution(age_true, age_preds)'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 16. Per-Class Accuracy']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'for i, group in enumerate(cfg.AGE_GROUPS):\n',
        '    mask = np.array(age_true) == i\n',
        '    if np.sum(mask) > 0:\n',
        '        acc = np.mean(np.array(age_preds)[mask] == i) * 100\n',
        '        print(f"{group}: {acc:.1f}%")'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 17. Sample Predictions']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'fig, axes = plt.subplots(4, 4, figsize=(14, 14))\n',
        'indices = np.random.choice(len(val_dataset), 16, replace=False)\n',
        'for idx, ax in zip(indices, axes.flatten()):\n',
        '    img, g_true, a_true, _ = val_dataset[idx]\n',
        '    with torch.no_grad():\n',
        '        g_logits, a_logits = model(img.unsqueeze(0).to(device))\n',
        '        g_pred, a_pred = g_logits.argmax(1).item(), a_logits.argmax(1).item()\n',
        '    img_np = img.numpy().transpose(1,2,0)\n',
        '    img_np = img_np * [0.229,0.224,0.225] + [0.485,0.456,0.406]\n',
        '    ax.imshow(np.clip(img_np, 0, 1))\n',
        '    color = "green" if g_pred==g_true and a_pred==a_true else "red"\n',
        '    ax.set_title(f"{[\"M\",\"F\"][g_pred]} {cfg.AGE_GROUPS[a_pred]}", color=color)\n',
        '    ax.axis("off")\n',
        'plt.savefig(f"{cfg.OUTPUT_DIR}/sample_predictions.png", dpi=150)\n',
        'plt.show()'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 18. Model Summary']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'total = sum(p.numel() for p in model.parameters())\n',
        'print(f"Parameters: {total:,}")\n',
        'print(f"Size: {total*4/1e6:.2f} MB")'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 19. Final Summary']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'summary = {\n',
        '    "Gender Acc": f"{np.mean(np.array(gender_preds)==np.array(gender_true))*100:.1f}%",\n',
        '    "Age Exact": f"{np.mean(np.array(age_preds)==np.array(age_true))*100:.1f}%",\n',
        '    "Age +-1": f"{np.mean(np.abs(np.array(age_preds)-np.array(age_true))<=1)*100:.1f}%",\n',
        '    "Params": f"{sum(p.numel() for p in model.parameters()):,}"\n',
        '}\n',
        'print(pd.DataFrame([summary]).T)\n',
        'pd.DataFrame([summary]).T.to_csv(f"{cfg.OUTPUT_DIR}/final_summary.csv")\n',
        'print("Saved final_summary.csv")'
    ]},
    {'cell_type': 'markdown', 'metadata': {}, 'source': ['## 📦 All Output Files']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
        'for f in sorted(os.listdir(cfg.OUTPUT_DIR)):\n',
        '    if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, f)):\n',
        '        print(f"  {f}: {os.path.getsize(os.path.join(cfg.OUTPUT_DIR, f))/1e6:.2f} MB")'
    ]}
]

nb['cells'].extend(new_cells)

with open('notebooks/train_with_visualizations.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Created: notebooks/train_with_visualizations.ipynb with {len(nb['cells'])} cells")
