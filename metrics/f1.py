from .base import BaseMetric


class F1Metric(BaseMetric):
    """F1-Score = 2 * Precision * Recall / (Precision + Recall)

    Calculated per-sample using set matching, then averaged:
    - Non-SI (single-label): GT = {label}, Pred = {label}
      -> match: F1=1.0, mismatch: F1=0.0
    - SI (multi-label): each symptom is an element in the set
      -> GT = {curling, yellowing}, Pred = {curling}
      -> TP=1, P=1/1=1.0, R=1/2=0.5, F1=2*1*0.5/1.5=0.667
    """

    def compute_sentence(self, ground_truth: str, pred: str, is_multilabel: bool = False):
        """Compute F1 score for a single (GT, Pred) pair.

        Args:
            ground_truth: Ground truth string.
            pred: Prediction string.
            is_multilabel: True if multi-label evaluation (comma-separated items for SI type).

        Returns:
            dict: {'Precision (%)': float, 'Recall (%)': float, 'F1-Score (%)': float}
        """
        if is_multilabel:
            gt_set = set(s.strip() for s in ground_truth.split(',') if s.strip())
            pred_set = set(s.strip() for s in pred.split(',') if s.strip())
        else:
            gt_set = {ground_truth.strip()}
            pred_set = {pred.strip()}

        tp = len(gt_set & pred_set)

        precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = tp / len(gt_set) if len(gt_set) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            'Precision (%)': precision * 100,
            'Recall (%)': recall * 100,
            'F1-Score (%)': f1 * 100,
        }

    def compute_for_type(self, df_type, dtype: str = None):
        """Compute average Precision, Recall, and F1 score for a specific disease type.

        Computes per-sample F1 scores and then calculates the macro-average over all samples.

        Args:
            df_type: DataFrame containing 'ground_truth_class' and 'pred_class' columns.
            dtype: Question/disease type ('SI' indicates multi-label, others are single-label).

        Returns:
            dict: {'Precision (%)': float, 'Recall (%)': float, 'F1-Score (%)': float}
        """
        is_multilabel = (dtype == 'SI')

        all_p, all_r, all_f1 = [], [], []

        for _, row in df_type.iterrows():
            result = self.compute_sentence(
                row['ground_truth_class'],
                row['pred_class'],
                is_multilabel=is_multilabel
            )
            all_p.append(result['Precision (%)'])
            all_r.append(result['Recall (%)'])
            all_f1.append(result['F1-Score (%)'])

        n = len(all_p) if all_p else 1

        return {
            'Precision (%)': sum(all_p) / n,
            'Recall (%)': sum(all_r) / n,
            'F1-Score (%)': sum(all_f1) / n,
        }
