from typing import Dict, List, Union, Tuple, Literal
import torch.distributed
from trl.trainer import DPOTrainer
import torch.nn.functional as F


class SVCOTrainerForLlava(DPOTrainer):
    def __init__(self, beta_img_win_vs_no_img_preference, beta_no_img_vs_img_lose_preference, beta_img_lose_vs_no_img_preference, beta_no_img_vs_img_win_preference, token_reg_alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_img_win_vs_no_img_preference, self.beta_no_img_vs_img_lose_preference, \
        self.beta_img_lose_vs_no_img_preference, self.beta_no_img_vs_img_win_preference, \
            = (
            beta_img_win_vs_no_img_preference,
            beta_no_img_vs_img_lose_preference,
            beta_img_lose_vs_no_img_preference,
            beta_no_img_vs_img_win_preference,
        )
        self.PAD_TOKEN_ID, self.IGNORE_TOKEN_ID, self.label_pad_token_id = (
            self.tokenizer.pad_token_id, -100, -100
        )

        self.token_reg_alpha = token_reg_alpha


    def get_token_logps(self, model, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs["logits"]  # [batch, seq_len, vocab]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_logps = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]
            return token_logps


    def compute_token_reward(self, ref_model, pos_img, pos_ans, pos_mask, neg_img, neg_ans, neg_mask):

        logp_ref_yw_iw = self.get_token_logps(ref_model, pos_img, pos_ans, pos_mask)  # π_eval(yw|iw)
        logp_ref_yw_il = self.get_token_logps(ref_model, neg_img, pos_ans, pos_mask)  # π_eval(yw|il)
        rew_yw = torch.sigmoid(logp_ref_yw_iw - logp_ref_yw_il) - 0.5
        rew_yw = rew_yw.clamp(-0.5, 0.5)

        logp_ref_yl_il = self.get_token_logps(ref_model, neg_img, neg_ans, neg_mask)  # π_eval(yl|il)
        logp_ref_yl_iw = self.get_token_logps(ref_model, pos_img, neg_ans, neg_mask)  # π_eval(yl|iw)
        rew_yl = torch.sigmoid(logp_ref_yl_il - logp_ref_yl_iw) - 0.5
        rew_yl = rew_yl.clamp(-0.5, 0.5)
        return rew_yw, rew_yl


    def compute_token_reg_loss(self, policy_model, ref_model, pos_img, pos_ans, pos_mask, neg_img, neg_ans, neg_mask,
                               alpha):

        rew_yw, rew_yl = self.compute_token_reward(ref_model, pos_img, pos_ans, pos_mask, neg_img, neg_ans, neg_mask)

        logp_policy_yw_iw = self.get_token_logps(policy_model, pos_img, pos_ans, pos_mask)
        logp_policy_yl_il = self.get_token_logps(policy_model, neg_img, neg_ans, neg_mask)
        # 只统计有效token
        mask_yw = (pos_mask > 0)
        mask_yl = (neg_mask > 0)
        reg_loss_yw = -((rew_yw * logp_policy_yw_iw * mask_yw).sum() / mask_yw.sum())
        reg_loss_yl = -((rew_yl * logp_policy_yl_il * mask_yl).sum() / mask_yl.sum())
        return alpha * (reg_loss_yw + reg_loss_yl)

    # 1. 四组输入
    def concatenated_inputs(self, batch: Dict) -> Dict[str, torch.Tensor]:
        (
            chosen_pixel_values, chosen_input_ids, chosen_labels, chosen_attention_mask,
            rejected_pixel_values, rejected_input_ids, rejected_labels, rejected_attention_mask,
            pos_img_neg_ans_pixel_values, pos_img_neg_ans_input_ids, pos_img_neg_ans_labels, pos_img_neg_ans_attention_mask,
            neg_img_pos_ans_pixel_values, neg_img_pos_ans_input_ids, neg_img_pos_ans_labels, neg_img_pos_ans_attention_mask,
        ) = batch
        return {
            "chosen_pixel_values": chosen_pixel_values,
            "chosen_input_ids": chosen_input_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_pixel_values": rejected_pixel_values,
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": rejected_attention_mask,
            "pos_img_neg_ans_pixel_values": pos_img_neg_ans_pixel_values,
            "pos_img_neg_ans_input_ids": pos_img_neg_ans_input_ids,
            "pos_img_neg_ans_labels": pos_img_neg_ans_labels,
            "pos_img_neg_ans_attention_mask": pos_img_neg_ans_attention_mask,
            "neg_img_pos_ans_pixel_values": neg_img_pos_ans_pixel_values,
            "neg_img_pos_ans_input_ids": neg_img_pos_ans_input_ids,
            "neg_img_pos_ans_labels": neg_img_pos_ans_labels,
            "neg_img_pos_ans_attention_mask": neg_img_pos_ans_attention_mask,
        }


    def concatenated_forward(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, ...]:
        inputs = self.concatenated_inputs(batch)
        dtype = next(model.parameters()).dtype

        def move_dtype(d):
            return {k: v.to(dtype) if "float" in str(v.dtype) else v for k, v in d.items()}


        out_chosen = model(
            pixel_values=inputs["chosen_pixel_values"],
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            labels=inputs["chosen_labels"],  # 一定要加labels
            return_dict=True
        )

        out_rejected = model(
            pixel_values=inputs["rejected_pixel_values"],
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            labels=inputs["rejected_labels"],
            return_dict=True
        )
        out_pos_img_neg_ans = model(
            pixel_values=inputs["pos_img_neg_ans_pixel_values"],
            input_ids=inputs["pos_img_neg_ans_input_ids"],
            attention_mask=inputs["pos_img_neg_ans_attention_mask"],
            labels=inputs["pos_img_neg_ans_labels"],
            return_dict=True
        )
        out_neg_img_pos_ans = model(
            pixel_values=inputs["neg_img_pos_ans_pixel_values"],
            input_ids=inputs["neg_img_pos_ans_input_ids"],
            attention_mask=inputs["neg_img_pos_ans_attention_mask"],
            labels=inputs["neg_img_pos_ans_labels"],
            return_dict=True
        )

        logits_chosen = out_chosen["logits"]
        logits_rejected = out_rejected["logits"]
        logits_pos_img_neg_ans = out_pos_img_neg_ans["logits"]
        logits_neg_img_pos_ans = out_neg_img_pos_ans["logits"]

        labels_chosen = out_chosen["labels_for_loss_calculation"]
        labels_rejected = out_rejected["labels_for_loss_calculation"]
        labels_pos_img_neg_ans = out_pos_img_neg_ans["labels_for_loss_calculation"]
        labels_neg_img_pos_ans = out_neg_img_pos_ans["labels_for_loss_calculation"]



        logps_chosen = self._get_batch_logps(logits_chosen, labels_chosen, average_log_prob=False)
        logps_rejected = self._get_batch_logps(logits_rejected, labels_rejected, average_log_prob=False)
        logps_pos_img_neg_ans = self._get_batch_logps(logits_pos_img_neg_ans, labels_pos_img_neg_ans, average_log_prob=False)
        logps_neg_img_pos_ans = self._get_batch_logps(logits_neg_img_pos_ans, labels_neg_img_pos_ans, average_log_prob=False)

        return (
            logps_chosen,         # (正图+正答)
            logps_rejected,       # (负图+负答)
            logps_pos_img_neg_ans,# (正图+负答)
            logps_neg_img_pos_ans,# (负图+正答)
            logits_chosen,
            logits_rejected,
            logits_pos_img_neg_ans,
            logits_neg_img_pos_ans
        )

    # 3. 对偶loss
    def dpo_loss(
        self,
        policy_logps_chosen: torch.FloatTensor,
        policy_logps_rejected: torch.FloatTensor,
        policy_logps_pos_img_neg_ans: torch.FloatTensor,
        policy_logps_neg_img_pos_ans: torch.FloatTensor,
        reference_logps_chosen: torch.FloatTensor,
        reference_logps_rejected: torch.FloatTensor,
        reference_logps_pos_img_neg_ans: torch.FloatTensor,
        reference_logps_neg_img_pos_ans: torch.FloatTensor,
        beta: float = 0.1,
        reference_free: bool = False,
        train_eval: str = "train",
    ):

        diff1 = (policy_logps_chosen - policy_logps_pos_img_neg_ans) - (
            0 if reference_free else (reference_logps_chosen - reference_logps_pos_img_neg_ans)
        )
        loss1 = -F.logsigmoid(beta * diff1)

        diff2 = (policy_logps_rejected - policy_logps_neg_img_pos_ans) - (
            0 if reference_free else (reference_logps_rejected - reference_logps_neg_img_pos_ans)
        )
        loss2 = -F.logsigmoid(beta * diff2)
        total_loss = loss1.mean() + loss2.mean()
        return {
            "loss1": loss1,
            "loss2": loss2,
            "loss": total_loss
        }


    def get_batch_metrics(
        self,
        model: torch.nn.Module,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.Tensor]]:
        metrics = {}
        (
            policy_logps_chosen,
            policy_logps_rejected,
            policy_logps_pos_img_neg_ans,
            policy_logps_neg_img_pos_ans,
            policy_logits_chosen,
            policy_logits_rejected,
            policy_logits_pos_img_neg_ans,
            policy_logits_neg_img_pos_ans,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_logps_chosen,
                        reference_logps_rejected,
                        reference_logps_pos_img_neg_ans,
                        reference_logps_neg_img_pos_ans,
                        *_,
                    ) = self.concatenated_forward(model, batch)
            else:
                (
                    reference_logps_chosen,
                    reference_logps_rejected,
                    reference_logps_pos_img_neg_ans,
                    reference_logps_neg_img_pos_ans,
                    *_,
                ) = self.concatenated_forward(self.ref_model, batch)

        loss_info = self.dpo_loss(
            policy_logps_chosen,
            policy_logps_rejected,
            policy_logps_pos_img_neg_ans,
            policy_logps_neg_img_pos_ans,
            reference_logps_chosen,
            reference_logps_rejected,
            reference_logps_pos_img_neg_ans,
            reference_logps_neg_img_pos_ans,
            beta=0.1,
            reference_free=False,
            train_eval=train_eval,
        )
        inputs = self.concatenated_inputs(batch)
        token_reg_loss = self.compute_token_reg_loss(
            policy_model=model,
            ref_model=self.ref_model,
            pos_img=inputs["chosen_pixel_values"],
            pos_ans=inputs["chosen_input_ids"],
            pos_mask=inputs["chosen_attention_mask"],
            neg_img=inputs["rejected_pixel_values"],
            neg_ans=inputs["rejected_input_ids"],
            neg_mask=inputs["rejected_attention_mask"],
            alpha=self.token_reg_alpha
        )
        total_loss = loss_info["loss"] + token_reg_loss

        metrics["loss1"] = loss_info["loss1"].mean().cpu()
        metrics["loss2"] = loss_info["loss2"].mean().cpu()
        metrics["dpo_loss"] = loss_info["loss"].cpu()
        metrics["token_reg_loss"] = token_reg_loss.cpu()
        metrics["total_loss"] = total_loss.cpu()
        return total_loss, metrics
