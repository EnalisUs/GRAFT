import argparse
from collections import defaultdict
import os
import torch
from torch.utils.data import DataLoader

from dataset import CSVDataset, collate_fn
from model import SCOLD_LFM_ImageMoE_LoRA


def parse_args():
    parser = argparse.ArgumentParser(description="Train SCOLD-LFM ImageMoE model for Plant Disease VQA.")
    parser.add_argument("--csv_path", type=str, default=r"E:\Tomato_train_new.csv", help="Path to training CSV file.")
    parser.add_argument("--image_root", type=str, default=None, help="Root directory containing images.")
    parser.add_argument("--scold_ckpt", type=str, default="scold.pth", help="Path to SCOLD checkpoint.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW optimizer.")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of MoE experts.")
    parser.add_argument("--top_k", type=int, default=2, help="Number of activated experts per sample.")
    parser.add_argument("--aux_loss_coef", type=float, default=0.01, help="Load balancing loss weight.")
    parser.add_argument("--diversity_loss_coef", type=float, default=0.001, help="Expert diversity loss weight.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor alpha.")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Whether to apply LoRA fine-tuning.")
    return parser.parse_args()


def log_expert_usage_by_qtype(usage_by_qtype: dict, num_experts: int):
    """
    Prints a formatted summary table of expert utilization per question type.
    """
    header = "  {:<30}".format("Question Type") + "".join(
        f"  E{e:>2}" for e in range(num_experts)
    )
    print(header)
    print("  " + "-" * (30 + 6 * num_experts))
    for qtype, usage in sorted(usage_by_qtype.items()):
        total = usage.sum().clamp(min=1e-8)
        row = "  {:<30}".format(qtype[:30]) + "".join(
            f"  {usage[e] / total:.2f}" for e in range(num_experts)
        )
        print(row)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    print(f"📂 Training CSV: {args.csv_path}")
    print(f"📦 SCOLD checkpoint: {args.scold_ckpt}\n")

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    dataset = CSVDataset(args.csv_path, image_root=args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=(device == "cuda"),
    )

    # ── Model Initialization ──────────────────────────────────────────────────
    model = SCOLD_LFM_ImageMoE_LoRA(
        scold_ckpt=args.scold_ckpt,
        num_experts=args.num_experts,
        top_k=args.top_k,
        aux_loss_coef=args.aux_loss_coef,
        diversity_loss_coef=args.diversity_loss_coef,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()

        epoch_task = 0.0
        epoch_aux = 0.0
        epoch_diversity = 0.0

        # Track expert allocation per question type
        usage_by_qtype = defaultdict(lambda: torch.zeros(args.num_experts))

        for step, (images, questions, answers, qtypes) in enumerate(loader, start=1):
            images = images.to(device)

            total_loss, task_loss, aux_loss, diversity_loss = model(
                images, questions, answers
            )

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️ NaN/Inf detected at step {step}, skipping batch.")
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accumulate loss terms
            epoch_task += task_loss.item()
            epoch_aux += aux_loss.item()
            epoch_diversity += diversity_loss.item()

            # Track expert mask per sample
            expert_mask = model.image_moe.last_expert_mask  # [B, E]
            if expert_mask is not None:
                mask_cpu = expert_mask.detach().cpu()
                for i, qt in enumerate(qtypes):
                    usage_by_qtype[qt] += mask_cpu[i]

            if step % 32 == 0:
                print(
                    f"Epoch {epoch} | Step {step:>4}/{len(loader)} | "
                    f"Task: {task_loss.item():.4f} | "
                    f"Aux: {aux_loss.item():.4f} | "
                    f"Div: {diversity_loss.item():.4f}"
                )

        # ── Epoch Summary ──────────────────────────────────────────────────────
        n = max(len(loader), 1)
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Task loss      : {epoch_task / n:.4f}")
        print(f"   Aux loss       : {epoch_aux / n:.4f}")
        print(f"   Diversity loss : {epoch_diversity / n:.4f}")

        # ── Expert Specialization Table ───────────────────────────────────────
        print(f"\n🔍 Expert Usage by Question Type (Epoch {epoch}):")
        log_expert_usage_by_qtype(usage_by_qtype, args.num_experts)
        print()

        # ── Checkpoint Saving ─────────────────────────────────────────────────
        lora_path = os.path.join(args.save_dir, f"lora_epoch_{epoch}")
        model.text_model.save_pretrained(lora_path)

        torch.save({
            "epoch": epoch,
            "visual_proj": model.visual_proj.state_dict(),
            "image_moe": model.image_moe.state_dict(),
            "fusion": model.visual_text_fusion.state_dict(),
        }, os.path.join(args.save_dir, f"model_epoch_{epoch}.pth"))

        print(f"💾 Checkpoints successfully saved for Epoch {epoch}: {lora_path}\n")


if __name__ == "__main__":
    main()
