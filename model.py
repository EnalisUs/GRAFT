import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe import ImageMoE
from vision_encoder import SCOLDVisionEncoder


class GRAFT(nn.Module):
    """
    Multimodal Visual Question Answering (VQA) Model with Image MoE and Residual Fusion.

    Architecture Overview:
        1. Vision          : SCOLD Encoder (frozen) -> visual_proj -> [B, 2048]
        2. Text            : LFM2.5-1.2B (LoRA) -> mean pooled embeddings -> [B, 2048]
        3. ImageMoE        : gate(text) -> select top-k experts -> expert(visual) -> visual_feat' [B, 2048]
                             Different questions route to different visual experts.
        4. Residual Fusion : combined = cat([visual', text]) -> MLP -> fused_feat [B, 2048]
                             inject fused representation into the first token embedding.
        5. LFM2.5 LLM      : Autoregressive language generation conditioned on multimodal prefix.
    """

    def __init__(
        self,
        scold_ckpt: str,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        diversity_loss_coef: float = 0.001,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()

        # ── 1. Vision Encoder (SCOLD, frozen) ─────────────────────────────────
        self.vision = SCOLDVisionEncoder(scold_ckpt)
        self.visual_proj = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
        )

        # ── 2. Text Model (LFM2.5 + LoRA) ─────────────────────────────────────
        model_id = "LiquidAI/LFM2.5-1.2B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            self.text_model = get_peft_model(base_model, lora_config)
            self.text_model.print_trainable_parameters()
            print(f"✅ LoRA applied: r={lora_r}, alpha={lora_alpha}")
        else:
            self.text_model = base_model

        # ── 3. ImageMoE ───────────────────────────────────────────────────────
        # Gate receives text_feat (question); experts process visual_feat (image)
        self.image_moe = ImageMoE(
            visual_dim=2048,
            text_dim=2048,
            num_experts=num_experts,
            top_k=top_k,
            hidden=4096,
            dropout=0.1,
        )

        # ── 4. Residual Fusion: cat([visual', text]) -> non-linear interaction ─
        self.visual_text_fusion = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
        )

        self.aux_loss_coef = aux_loss_coef
        self.diversity_loss_coef = diversity_loss_coef

        print(f"✅ GRAFT (Residual Fusion): {num_experts} experts, top_k={top_k}")

    # ──────────────────────────────────────────────────────────────────────────
    def get_text_embedding(self, questions: list, device: torch.device) -> torch.Tensor:
        """
        Extracts raw text embeddings from LFM2.5 via mean pooling.
        Used as routing condition for ImageMoE and input for multimodal fusion.

        Returns:
            text_feat: [B, 2048]
        """
        tokens = self.tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt",
        ).to(device)

        # Use input embedding layer of LFM2.5 (frozen for this feature extraction)
        with torch.no_grad():
            text_embeds = self.text_model.get_input_embeddings()(tokens.input_ids)
            # [B, seq_len, 2048]

        # Mean pooling with attention mask
        mask = tokens.attention_mask.unsqueeze(-1)              # [B, seq_len, 1]
        text_feat = (text_embeds * mask).sum(dim=1)             # [B, 2048]
        text_feat = text_feat / mask.sum(dim=1).clamp(min=1)    # Normalize

        return text_feat  # [B, 2048]

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor, questions: list, answers: list):
        """
        Training forward pass.

        Args:
            images   : [B, 3, 224, 224]
            questions: list[str] - input questions
            answers  : list[str] - ground truth answers

        Returns:
            total_loss, task_loss, aux_loss, diversity_loss
        """
        device = images.device

        # ── Step 1: Visual features ───────────────────────────────────────────
        visual_feat = self.vision(images)            # [B, 512] (frozen)
        visual_feat = self.visual_proj(visual_feat)  # [B, 2048]

        # ── Step 2: Text embeddings ───────────────────────────────────────────
        text_feat = self.get_text_embedding(questions, device)   # [B, 2048]

        # ── Step 3: ImageMoE ──────────────────────────────────────────────────
        # Question semantics guide the selection of visual experts
        visual_feat = self.image_moe(visual_feat, text_feat)     # [B, 2048]

        # ── Step 4: Residual Fusion ───────────────────────────────────────────
        combined = torch.cat([visual_feat, text_feat], dim=-1)   # [B, 4096]
        fused_feat = self.visual_text_fusion(combined)           # [B, 2048]

        # ── Step 5: Build prompt sequence ─────────────────────────────────────
        full_prompts = [f"{q} Answer: {a}" for q, a in zip(questions, answers)]
        tokens = self.tokenizer(
            full_prompts,
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt",
        ).to(device)

        # Inject multimodal fused features into the first token embedding
        input_embeds = self.text_model.get_input_embeddings()(tokens.input_ids)
        input_embeds[:, 0, :] = input_embeds[:, 0, :] + fused_feat

        # ── Step 6: LFM2.5 forward pass ───────────────────────────────────────
        outputs = self.text_model(
            inputs_embeds=input_embeds,
            attention_mask=tokens.attention_mask,
            labels=tokens.input_ids,
        )

        task_loss = outputs.loss
        aux_loss = self.image_moe.compute_load_balance_loss()
        diversity_loss = self.image_moe.compute_expert_diversity_loss()

        total_loss = (
            task_loss
            + self.aux_loss_coef * aux_loss
            + self.diversity_loss_coef * diversity_loss
        )

        return total_loss, task_loss, aux_loss, diversity_loss

    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, image: torch.Tensor, question: str, max_tokens: int = 20) -> str:
        """
        Inference: Generate answer given an input image and question.

        Args:
            image     : [3, 224, 224] - single image tensor
            question  : str - question prompt
            max_tokens: int - maximum new tokens to generate

        Returns:
            answer_text: str - decoded generated answer
        """
        device = image.device

        # ── Visual ────────────────────────────────────────────────────────────
        visual_feat = self.vision(image.unsqueeze(0))    # [1, 512]
        visual_feat = self.visual_proj(visual_feat)       # [1, 2048]

        # ── Text ──────────────────────────────────────────────────────────────
        text_feat = self.get_text_embedding([question], device)   # [1, 2048]

        # ── ImageMoE ──────────────────────────────────────────────────────────
        visual_feat = self.image_moe(visual_feat, text_feat)      # [1, 2048]

        # ── Residual Fusion ───────────────────────────────────────────────────
        combined = torch.cat([visual_feat, text_feat], dim=-1)
        fused_feat = self.visual_text_fusion(combined)             # [1, 2048]

        # ── Build prompt ──────────────────────────────────────────────────────
        prompt = f"{question} Answer:"
        tokens = self.tokenizer(prompt, return_tensors="pt").to(device)

        # Inject multimodal fused features into first token
        input_embeds = self.text_model.get_input_embeddings()(tokens.input_ids)
        input_embeds[:, 0, :] = input_embeds[:, 0, :] + fused_feat

        # Stop tokens: EOS + newline
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        stop_ids = list(set(
            [s for s in [self.tokenizer.eos_token_id] + newline_ids if s is not None]
        ))

        gen_ids = self.text_model.generate(
            inputs_embeds=input_embeds,
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=stop_ids,
        )

        # Extract generated continuation following prompt tokens
        new_ids = gen_ids[0, tokens.input_ids.shape[1]:]
        answer_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return answer_text
