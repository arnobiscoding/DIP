import json
from pathlib import Path

def analyze_notebook(nb_path):
    print("=" * 80)
    print(f"ANALYZING: {nb_path.name}")
    print("=" * 80)
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb.get("cells", [])
    print(f"Total cells: {len(cells)}")
    
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    print(f"Code cells: {len(code_cells)}")
    
    print("\n--- CODE CELLS OVERVIEW & KEY STATEMENTS ---")
    for i, c in enumerate(code_cells):
        src = "".join(c.get("source", []))
        outputs = c.get("outputs", [])
        
        # Extract title or first lines
        lines = [l.strip() for l in src.split("\n") if l.strip()]
        header = lines[0] if lines else "EMPTY CELL"
        for l in lines[:5]:
            if "CELL" in l or "SECTION" in l or "class " in l or "def " in l:
                header = l
                break
                
        print(f"\n[Cell {i}] {header}")
        
        # Check for key methodology components
        keywords = [
            "CONFIG", "vit", "timm", "transform", "Contrastive", "NTXent", "InfoNCE",
            "Projection", "Stratified", "train_indices", "test_indices", "val_indices",
            "leakage", "evaluate", "roc_auc", "f1_score", "recall", "precision",
            "confusion_matrix", "FRACTIONS", "Subset", "DataLoader", "loss", "optimizer"
        ]
        found_kw = [kw for kw in keywords if kw.lower() in src.lower()]
        if found_kw:
            print(f"   Keywords: {', '.join(found_kw)}")
            
        # Check outputs for stdout or errors
        out_texts = []
        for out in outputs:
            if out.get("output_type") == "stream":
                out_texts.append("".join(out.get("text", [])))
            elif out.get("output_type") == "error":
                out_texts.append(f"ERROR: {out.get('ename')}: {out.get('evalue')}")
        if out_texts:
            full_out = "".join(out_texts).strip()
            # print up to first 300 chars of output
            print(f"   Output preview: {full_out[:250].replace(chr(10), ' | ')}...")

if __name__ == "__main__":
    for nb in [Path("pipeline-2.ipynb"), Path("pcam-data-efficiency-benchmark.ipynb")]:
        if nb.exists():
            analyze_notebook(nb)
        else:
            print(f"Notebook not found: {nb}")
