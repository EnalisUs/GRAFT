import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CSVDataset
from metrics import F1Metric
from model import SCOLD_LFM_ImageMoE_LoRA

# Hallucination tokens to filter out from raw LLM generation
GARBAGE_TOKENS = {"fjärils", "butterfly", "butterflies"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SCOLD-LFM ImageMoE model on test dataset.")
    parser.add_argument("--checkpoint", type=str, default=r"checkpoints/model_epoch_19.pth", help="Path to model checkpoint.")
    parser.add_argument("--lora_path", type=str, default=r"checkpoints/lora_epoch_19", help="Path to LoRA weights directory.")
    parser.add_argument("--scold_ckpt", type=str, default="scold.pth", help="Path to SCOLD pretrained checkpoint.")
    parser.add_argument("--test_csv", type=str, default=r"E:\Tomato_test_new.csv", help="Path to evaluation test CSV.")
    parser.add_argument("--image_root", type=str, default=None, help="Optional image root directory.")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of MoE experts.")
    parser.add_argument("--top_k", type=int, default=2, help="Number of active experts per token.")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Whether LoRA was applied.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank parameter.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor.")
    parser.add_argument("--output_excel", type=str, default="image_moe_evaluation.xlsx", help="Filename for output Excel spreadsheet.")
    return parser.parse_args()


def clean_predict(pred: str) -> str:
    """
    Cleans generated predictions: eliminates hallucinated tokens and deduplicates repeated labels.
    """
    parts = [p.strip() for p in pred.split(",") if p.strip()]
    cleaned, seen = [], set()
    for part in parts:
        words = part.split()
        dedup = [
            w for i, w in enumerate(words)
            if w.lower() not in GARBAGE_TOKENS
            and (not words[:i] or w.lower() != words[i - 1].lower())
        ]
        label = " ".join(dedup).strip()
        label_key = label.lower()
        if label and label_key not in seen:
            seen.add(label_key)
            cleaned.append(label)
    return ", ".join(cleaned)


class CSVDatasetExtended(CSVDataset):
    """
    Extended dataset retaining image path and ground truth disease metadata for evaluation reporting.
    """
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        row = self.data.iloc[idx]
        item["img_path"] = str(row[0])
        item["disease_type"] = str(row[2]).strip()   # Question type (qtype)
        item["disease_name"] = str(row[9]).strip()   # Ground truth label
        return item


def collate_fn_extended(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    questions = [b["question"] for b in batch]
    answers = [b["answer"] for b in batch]
    img_paths = [b["img_path"] for b in batch]
    disease_types = [b["disease_type"] for b in batch]
    disease_names = [b["disease_name"] for b in batch]
    return images, questions, answers, img_paths, disease_types, disease_names


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🖥️ Device: {device}")
    print(f"📂 Test CSV: {args.test_csv}\n")

    dataset = CSVDatasetExtended(args.test_csv, image_root=args.image_root)
    test_loader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn_extended,
        batch_size=1,
    )
    print(f"✅ Loaded {len(dataset)} test samples\n")

    # ── Model Initialization ──────────────────────────────────────────────────
    print("🔧 Initializing model architecture...")
    model = SCOLD_LFM_ImageMoE_LoRA(
        scold_ckpt=args.scold_ckpt,
        num_experts=args.num_experts,
        top_k=args.top_k,
        aux_loss_coef=0.01,
        diversity_loss_coef=0.001,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    # ── Load Checkpoints ──────────────────────────────────────────────────────
    print(f"📥 Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    model.visual_text_fusion.load_state_dict(ckpt["fusion"])
    model.image_moe.load_state_dict(ckpt["image_moe"])

    if args.use_lora and os.path.exists(args.lora_path):
        model.text_model.load_adapter(args.lora_path, adapter_name="default")
        model.text_model.set_adapter("default")
        print("✅ Loaded LoRA adapter\n")

    model.eval()

    # ── Metrics Evaluator ─────────────────────────────────────────────────────
    f1_metric = F1Metric()

    # ── Inference Loop ────────────────────────────────────────────────────────
    all_results = []
    print("🧪 Running inference on test set...\n")

    with torch.no_grad():
        for images, questions, answers, img_paths, disease_types, disease_names in tqdm(
            test_loader, desc="Evaluating"
        ):
            images = images.to(device)

            for i in range(len(questions)):
                disease_type = disease_types[i]
                gt_class = disease_names[i].strip().upper()

                if disease_type == "SI":
                    max_tokens = 20
                else:
                    toks = model.tokenizer(answers[i].strip(), return_tensors="pt")
                    max_tokens = len(toks["input_ids"][0]) + 2

                pred = model.generate(images[i], questions[i], max_tokens)
                pred = clean_predict(pred)
                pred_upper = pred.strip().upper()

                ans_parts = [a.strip() for a in answers[i].split(",") if a.strip()]
                is_correct = all(a.upper() in pred_upper for a in ans_parts)

                all_results.append({
                    "img_path": img_paths[i],
                    "disease_type": disease_type,
                    "ground_truth_class": gt_class,
                    "pred_class": pred_upper,
                    "correct": is_correct,
                })

    print("\n✅ Inference completed successfully!\n")
    df_all = pd.DataFrame(all_results)
    unique_disease_types = sorted(df_all["disease_type"].unique())

    # ── Sheet 1: Metrics per Question Type ────────────────────────────────────
    sheet1_data = []
    for dtype in unique_disease_types:
        df_type = df_all[df_all["disease_type"] == dtype]
        total = len(df_type)
        correct = df_type["correct"].sum()
        accuracy = (correct / total) * 100 if total > 0 else 0

        try:
            f1_res = f1_metric.compute_for_type(df_type, dtype)
        except Exception:
            f1_res = {"F1-Score (%)": 0.0}

        row = {
            "Disease_Type": dtype,
            "Total_Samples": total,
            "Correct": correct,
            "Incorrect": total - correct,
            "Accuracy (%)": f"{accuracy:.2f}",
            "F1-Score (%)": f"{f1_res['F1-Score (%)']:.2f}",
        }
        sheet1_data.append(row)
    df_sheet1 = pd.DataFrame(sheet1_data)

    # ── Sheet 2: Breakdown per Disease Class ───────────────────────────────────
    sheet2_data = []
    for dtype in unique_disease_types:
        df_type = df_all[df_all["disease_type"] == dtype]
        for gt_class in sorted(df_type["ground_truth_class"].unique()):
            df_name = df_type[df_type["ground_truth_class"] == gt_class]
            total = len(df_name)
            correct = df_name["correct"].sum()
            accuracy = (correct / total) * 100 if total > 0 else 0
            sheet2_data.append({
                "Disease_Type": dtype,
                "Ground_Truth_Class": gt_class,
                "Total_Samples": total,
                "Correct": correct,
                "Incorrect": total - correct,
                "Accuracy (%)": f"{accuracy:.2f}",
            })
    df_sheet2 = pd.DataFrame(sheet2_data)

    # ── Sheet 3: Detailed Predictions ─────────────────────────────────────────
    sheet3_data = []
    for row in all_results:
        sheet3_data.append({
            "Disease_Type": row["disease_type"],
            "Image_Path": row["img_path"],
            "Ground_Truth_Answer": row["ground_truth_class"],
            "Predicted_Answer": row["pred_class"],
            "Correct": row["correct"],
        })
    df_sheet3 = pd.DataFrame(sheet3_data)

    # ── Save to Excel ─────────────────────────────────────────────────────────
    print(f"💾 Saving evaluation results to {args.output_excel}...")
    with pd.ExcelWriter(args.output_excel, engine="openpyxl") as writer:
        df_sheet1.to_excel(writer, sheet_name="Per-Disease-Type Metrics", index=False)
        df_sheet2.to_excel(writer, sheet_name="Disease Breakdown by Type", index=False)
        df_sheet3.to_excel(writer, sheet_name="Detailed Predictions", index=False)
    print("✅ Excel report generated!\n")

    # ── Final Summary ─────────────────────────────────────────────────────────
    overall_acc = (df_all["correct"].sum() / len(df_all) * 100) if len(df_all) > 0 else 0

    print("=" * 70)
    print("🏆 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total samples    : {len(df_all)}")
    print(f"  Overall accuracy : {overall_acc:.2f}%")
    print(f"  Disease types    : {', '.join(unique_disease_types)}")
    print(f"\n  Results saved to : {args.output_excel}")
    print("=" * 70)


if __name__ == "__main__":
    main()
