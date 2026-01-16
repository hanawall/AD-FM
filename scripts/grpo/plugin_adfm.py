import re
from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from swift.plugin import ORM, orms

# ----------------------------
# Helpers
# ----------------------------

def _extract_think(text: str) -> str:
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else ""

def parse_bbox_list(text: str) -> List[List[float]]:
    pat = r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]'
    bboxes: List[List[float]] = []
    for x1, y1, x2, y2 in re.findall(pat, text):
        bboxes.append([float(x1), float(y1), float(x2), float(y2)])
    return bboxes

def calculate_generalized_iou(b1: List[float], b2: List[float]) -> float:
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2

    xi1, yi1 = max(x11, x21), max(y11, y21)
    xi2, yi2 = min(x12, x22), min(y12, y22)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)

    a1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    a2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
    union = a1 + a2 - inter
    iou = 0.0 if union <= 0 else inter / union

    xc1, yc1 = min(x11, x21), min(y11, y21)
    xc2, yc2 = max(x12, x22), max(y12, y22)
    enc = max(0.0, xc2 - xc1) * max(0.0, yc2 - yc1)
    if enc <= 0:
        return iou
    return iou - (enc - union) / enc

def hungarian_bbox_matching(pred_bboxes: List[List[float]],
                            gt_bboxes: List[List[float]]) -> Tuple[List[int], List[float]]:
    if not pred_bboxes or not gt_bboxes:
        return [], []
    cost = np.zeros((len(pred_bboxes), len(gt_bboxes)), dtype=np.float32)
    for i, pb in enumerate(pred_bboxes):
        for j, gb in enumerate(gt_bboxes):
            cost[i, j] = -calculate_generalized_iou(pb, gb)
    pi, gi = linear_sum_assignment(cost)
    matched = [-1] * len(pred_bboxes)
    giou_scores = [0.0] * len(pred_bboxes)
    for p_idx, g_idx in zip(pi, gi):
        matched[p_idx] = g_idx
        giou_scores[p_idx] = -cost[p_idx, g_idx]
    return matched, giou_scores

# ----------------------------
# Rewards
# ----------------------------

class AnomalyCoTFormat(ORM):
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        gt_answers = kwargs.pop("answer", ['b'] * len(completions))          # 'a' or 'b'
        bbox_answers = kwargs.pop("bbox_answer", [[] for _ in completions])  # list of GT boxes per sample
        rewards = []

        bbox_pat = re.compile(
            r'\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]',
            re.IGNORECASE
        )

        for comp, gt, gt_bboxes in zip(completions, gt_answers, bbox_answers):
            reward = 0.0
            gt = str(gt).strip().lower()
            if gt_bboxes is None:
                gt_bboxes = []
            has_gt_loc = (gt == 'a') and isinstance(gt_bboxes, list) and len(gt_bboxes) > 0

            think_m = re.search(r'<think>(.*?)</think>', comp, re.DOTALL | re.IGNORECASE)
            think = think_m.group(1).strip() if think_m else ""
            if think:
                reward += 0.3
                if len(think) > 10:
                    reward += 0.1

            rethink_m = re.search(r'<rethink>(.*?)</rethink>', comp, re.DOTALL | re.IGNORECASE)
            if rethink_m:
                rtxt = rethink_m.group(1).strip()
                if rtxt:
                    reward += 0.2
                    if len(rtxt) > 10:
                        reward += 0.1

            ans_m = re.search(r'<answer>(.*?)</answer>', comp, re.DOTALL | re.IGNORECASE)
            ans = ans_m.group(1).strip().lower() if ans_m else ""
            if ans in ('a', 'b'):
                reward += 0.2
            elif 'a' in ans or 'b' in ans:
                reward += 0.1

            if think:
                has_bbox_in_think = bool(bbox_pat.search(think))
                has_bracket_hint = ('[' in think and ']' in think)

                if has_bbox_in_think:
                    reward += 0.2
                    bboxes = parse_bbox_list(think)
                    if bboxes and all(all(coord < 505 for coord in bbox) for bbox in bboxes): # For original image size=512*512
                        reward += 0.1
                elif has_bracket_hint:
                    reward += 0.1

            rewards.append(reward)

        return rewards

class AnomalyClsReward(ORM):
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        gt_answers: List[str] = kwargs.pop("answer", [])
        out: List[float] = []
        for comp, gt in zip(completions, gt_answers):
            gt_norm = 'a' if str(gt).strip().lower() == 'a' else 'b'
            m = re.search(r'<answer>(.*?)</answer>', comp, re.DOTALL | re.IGNORECASE)
            pred = (m.group(1) if m else comp).strip().lower()
            pred_norm = 'a' if 'a' == pred else ('b' if 'b' == pred else '')
            if pred_norm == gt_norm:
                out.append(1.0)
            elif gt_norm in pred and len(pred) <= 10:
                out.append(0.8)
            else:
                out.append(0.0)
        return out

class AnomalyLocReward(ORM):
    def __init__(self, alpha: float = 0.3):
        self.alpha = float(alpha)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        gt_answers: List[str] = kwargs.pop("answer", [])
        bbox_answers: List[List[List[float]]] = kwargs.pop("bbox_answer", [])
        scores: List[float] = []

        for comp, gt, gt_bboxes in zip(completions, gt_answers, bbox_answers):
            is_anom = (str(gt).strip().lower() == 'a')
            think = _extract_think(comp)
            pred_bboxes = parse_bbox_list(think)
            m = len(pred_bboxes)
            
            if gt_bboxes is None:
                gt_bboxes = []
            n = len(gt_bboxes)

            if not is_anom or n == 0:
                if m == 0: 
                    r_focus = 0.0
                elif m == 1: 
                    r_focus = 0.5
                else:       
                    r_focus = -0.1
                scores.append(r_focus)
                continue

            if m == 0 or n == 0:
                scores.append(0.0)
                continue

            if m == n:          
                r_count = 1.0
            elif abs(m - n)==1: 
                r_count = 0.5
            else:               
                r_count = -0.1

            _, giou_list = hungarian_bbox_matching(pred_bboxes, gt_bboxes)
            mean_giou = float(np.mean(giou_list)) if giou_list else 0.0
            r_loc = mean_giou + self.alpha * r_count

            r_min, r_max = -0.5, 1.0
            r_loc_norm = (r_loc - r_min) / (r_max - r_min)
            r_loc_norm = max(0.0, min(1.0, r_loc_norm))
            scores.append(r_loc_norm)
        return scores

orms["anomaly_cls_reward"] = AnomalyClsReward
orms["anomaly_loc_reward"] = AnomalyLocReward
orms["anomaly_cot_format"] = AnomalyCoTFormat