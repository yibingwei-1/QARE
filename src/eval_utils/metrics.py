import numpy as np
import logging

import numpy as np


logger = logging.getLogger(__name__)


class RankingMetrics:
    def __init__(self, metric_list, k_list=(1, 5)):
        """
        Initialize retrieval metrics.

        Args:
            metric_list (tuple or list): Metrics to compute ("precision", "recall", "ndcg", "map", "mrr").
            k_list (tuple or list): List of K values (e.g., (1, 5, 10)) for evaluation.
        """
        if isinstance(metric_list, str):
            metric_list = [metric_list]
        self.metric_list = metric_list
        self.k_list = sorted(k_list)  # Ensure K is in ascending order
    
    def gt_avg(self, prediction, prediction_scores, true_labels, k):
        """
        GT-Pair Confidence: average score over all true (image,attr) pairs
        """
        gt_scores = []
        for p, s in zip(prediction, prediction_scores):
            if p in true_labels:
                gt_scores.append(s)
        return sum(gt_scores[:k]) / k
    
    def first_err(self, prediction, prediction_scores, query_attr, query_image):
        """
        First Error: find the first pair whose attribute is not in the query’s ground-truth set but comes from the same image as the query. Record its similarity score.
        If the query is not in the prediction, return first attribute error score.
        """
        for p, s in zip(prediction, prediction_scores):
            p_image, p_attr = p.split(":")
            if p_image == query_image and p_attr != query_attr:
                return s
        for p, s in zip(prediction, prediction_scores):
            p_image, p_attr = p.split(":")
            if p_attr != query_attr:
                return s
        return 0.0
    
    def PN_gap(self, prediction, prediction_scores, true_labels, query_attr, query_image):
        """
        Positive (P): find the first correct pair that comes from a different image than the query (i.e., same attribute class, different picture).
        Negative (N): find the first incorrect attribute that still comes from the same image as the query.
                      If the query is not in the prediction, return first attribute error score.
        Compute P − N.
        """
        first_err_score = self.first_err(prediction, prediction_scores, query_attr, query_image)
        for p, s in zip(prediction, prediction_scores):
            p_image, p_attr = p.split(":")
            if p_image != query_image and p_attr == query_attr and p in true_labels:
                # assert s > first_err_score # baseline model s > first_err_score 
                return s - first_err_score
        return 0.0
    
    def PN_gap_no_qry_img(self, prediction, prediction_scores, true_labels, query_attr, query_image):
        """
        Positive (P): find the first correct pair
        Negative (N): find the first incorrect attribute that still comes from the same image as the query.
        Compute P − N.
        """
        first_err_score = self.first_err(prediction, prediction_scores, query_attr, query_image)
        for p, s in zip(prediction, prediction_scores):
            if p in true_labels:
                # assert s > first_err_score # baseline model s > first_err_score 
                return s - first_err_score
        return 0.0

    def precision_at_k(self, prediction, true_labels, k):
        """
        Compute Precision@K for multiple true labels.
        Precision@K = (Number of relevant items in top K) / K
        """
        if not true_labels: # No relevant items
            return 0.0
        if k == 0:
            return 0.0
        predicted_k = prediction[:k]
        relevant_hits = len(set(predicted_k).intersection(set(true_labels)))
        return relevant_hits / k

    def recall_at_k(self, prediction, true_labels, k):
        """
        Compute Recall@K for multiple true labels.
        Recall@K = (Number of relevant items in top K) / (Total number of relevant items)
        """
        if not true_labels: # No relevant items
            return 1.0 # Or 0.0 depending on convention if no relevant items exist.
            # Typically, if there are no true positives, recall is undefined or 1 if no predictions are made.
            # Let's assume if true_labels is empty, it means perfect recall if prediction is also empty, or 0 if not.
            # A common convention is that if the set of true positives is empty, recall is 1.0.
            # However, if true_labels is empty but predictions are made, it might be 0.
            # Let's use the common definition: if len(true_labels) == 0, recall is 1.0 if no items were expected and none were retrieved.
            # If items were expected (true_labels not empty) but none retrieved, recall is 0.
            # If true_labels is empty and nothing is predicted, it's 1.0.
            # If true_labels is empty and something is predicted, it's 0.0. (This is debatable)
            # For simplicity here, if there are no true_labels, then no relevant items can be found.
            return 1.0 # All 0 relevant items were recalled
        if k == 0 and not true_labels: # no predictions and no true labels
            return 1.0
        if k == 0 and true_labels: # no predictions but true labels exist
            return 0.0

        predicted_k = prediction[:k]
        relevant_hits = len(set(predicted_k).intersection(set(true_labels)))
        return relevant_hits / len(true_labels)

    def _get_relevant_hits_and_predicted_k(self, prediction, true_labels, k):
        """Helper function to get common elements for calculations."""
        if k == 0:
            return 0, []
        predicted_k = prediction[:k]
        if not true_labels: # No relevant items
            return 0, predicted_k
        relevant_hits = len(set(predicted_k).intersection(set(true_labels)))
        return relevant_hits, predicted_k

    def hit_at_k(self, prediction, true_labels, k):
        """
        Compute Hit@K (or Hit Rate@K).
        Returns 1.0 if at least one true label is found in the top K predictions, 0.0 otherwise.
        """
        if not true_labels: # No relevant items to hit
            return 0.0
        if k == 0:
            return 0.0
        relevant_hits, _ = self._get_relevant_hits_and_predicted_k(prediction, true_labels, k)
        return 1.0 if relevant_hits > 0 else 0.0

    def f1_at_k(self, prediction, true_labels, k):
        """
        Compute F1-score@K.
        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
        """
        p_k = self.precision_at_k(prediction, true_labels, k)
        r_k = self.recall_at_k(prediction, true_labels, k)

        if (p_k + r_k) == 0:
            return 0.0
        return 2 * (p_k * r_k) / (p_k + r_k)

    def average_precision_at_k(self, prediction, true_labels, k):
        """
        Compute Average Precision (AP)@K for multiple true labels.
        AP is the average of precision@i for i where the i-th item is relevant.
        """
        if not true_labels:
            return 0.0 # Or 1.0 depending on definition if no relevant items. Let's use 0.0 if no true positives.
        if k == 0:
            return 0.0

        predicted_k = prediction[:k]
        score = 0.0
        num_hits = 0
        for i, p in enumerate(predicted_k):
            if p in true_labels:
                num_hits += 1
                score += num_hits / (i + 1.0) # Precision at this new hit

        if not num_hits: # No hits among top k
            return 0.0

        return score / min(len(true_labels), k) # Normalize by min of total relevant or K for AP@K
        # Standard AP normalizes by total relevant items (len(true_labels))

    def mean_average_precision_at_k(self, test_cases_for_map_mrr, k_val):
        """Helper for MAP@k"""
        ap_scores = []
        for case in test_cases_for_map_mrr:
            prediction, true_labels_list = case["prediction"], case["label"]
            # Ensure true_labels_list is always a list for consistency
            if not isinstance(true_labels_list, list):
                true_labels_list = [true_labels_list]
            ap_scores.append(self.average_precision_at_k(prediction, true_labels_list, k_val))
        return np.mean(ap_scores) if ap_scores else 0.0


    def mean_reciprocal_rank_at_k(self, test_cases_for_map_mrr, k_val):
        """Helper for MRR@k"""
        rr_scores = []
        for case in test_cases_for_map_mrr:
            prediction, true_labels_list = case["prediction"], case["label"]
            if not isinstance(true_labels_list, list):
                true_labels_list = [true_labels_list]
            if not true_labels_list:
                rr_scores.append(0.0) # Or handle as per definition, no relevant items.
                continue

            predicted_k = prediction[:k_val]
            for rank, p_item in enumerate(predicted_k):
                if p_item in true_labels_list:
                    rr_scores.append(1.0 / (rank + 1))
                    break # Found first relevant item
            else: # No relevant item found in top K
                rr_scores.append(0.0)
        return np.mean(rr_scores) if rr_scores else 0.0


    def ndcg_at_k(self, prediction, true_labels, k, rel_scores, form="linear"):
        """
        Compute Normalized Discounted Cumulative Gain (NDCG@K) with optional graded relevance and exponential gain.

        Args:
            prediction (list): Ranked list of predicted document IDs.
            true_labels (str or list): Relevant document(s).
            rel_scores (list or None): If provided, must be same length as label. Interpreted as graded relevance.
                                 If None, use binary relevance.
        """

        def dcg(rel_list, form):
            if form == "linear":
                return sum(((rel) / np.log2(idx + 2)) for idx, rel in enumerate(rel_list))
            elif form == "exponential":
                return sum(((2 ** rel - 1) / np.log2(idx + 2)) for idx, rel in enumerate(rel_list))

        if rel_scores is None:
            # Binary relevance
            if isinstance(true_labels, list):
                label_set = set(true_labels)
            else:
                label_set = {true_labels}
            relevance = [1 if item in label_set else 0 for item in prediction[:k]]
            ideal_relevance = [1] * min(len(label_set), k)
        else:
            # Graded relevance
            if not isinstance(true_labels, list) or not isinstance(rel_scores, list) or len(true_labels) != len(rel_scores):
                raise ValueError("If rels is provided, label must be a list of the same length.")
            label_rels = dict(zip(true_labels, rel_scores))
            relevance = [label_rels.get(item, 0) for item in prediction[:k]]
            ideal_relevance = sorted(label_rels.values(), reverse=True)[:k]
        dcg_score = dcg(relevance, form)
        idcg_score = dcg(ideal_relevance, form) if ideal_relevance else 1  # Avoid division by zero

        return dcg_score / idcg_score if idcg_score > 0 else 0.0


    def evaluate(self, test_cases):
        """
        Evaluates predictions against true labels for a list of test cases.
        Handles multi-label by ensuring `label` is a list of true positives.

        Args:
            test_cases (list of dict): Each dict should have "prediction" (list)
                                       , "label" (list of true positive labels),
                                       , "prediction_scores" (list of prediction scores),
                                       and "rel_scores" (list of graded relevance scores).
        """

        metric_k_set = set(['precision', 'recall', 'hit', 'f1', 'ndcg_linear', 'ndcg_exponential', 'gt_avg', 'gt_avg_no_qry_img']) 
        metric_results_accumulators = {
            (f"ndcg_{v}" if metric == "ndcg" else metric): {k: [] for k in self.k_list}
            for metric in self.metric_list if metric in metric_k_set
            for v in (["linear", "exponential"] if metric == "ndcg" else [None])
        }
        # Add gt_avg_no_qry_img to the keys, with the same k-list structure
        metric_results_accumulators["gt_avg_no_qry_img"] = {k: [] for k in self.k_list}
        metric_results_accumulators["first_err"] = []
        metric_results_accumulators["PN_gap_first_gt"] = []
        metric_results_accumulators["PN_gap_first_not_qry_img"] = []

        processed_test_cases_for_map_mrr = []
        processed_test_cases_for_map_attr = {attr: [] for attr in ['object', 'background','style']}
        processed_test_cases_for_map_no_qry = []
        processed_test_cases_for_map_attr_no_qry = {attr: [] for attr in ['object', 'background','style']}
        
        for case_idx, case in enumerate(test_cases):
            prediction = case["prediction"]
            true_labels = case["label"]
            if not isinstance(true_labels, (list, set)): # Allow set for true_labels
                true_labels = [true_labels]
            true_labels = list(set(true_labels)) # Ensure unique true labels and convert to list for consistency

            # For MAP and MRR, we often want the original list of true_labels,
            # even if empty, for consistent calculation across cases.
            if "map" in self.metric_list or "mrr" in self.metric_list or "map_all" in self.metric_list:
                processed_test_cases_for_map_mrr.append(
                    {"prediction": prediction, "label": true_labels, "id": case_idx}
                )
            

            if "map" in self.metric_list or "map_all" in self.metric_list:
                query_attr = case["query_attr"]
                query_image = case["qry_img_path"].split("/")[-1]
                prediction_no_qry, prediction_scores_no_qry = self.remove_query(prediction, case["prediction_scores"], query_attr, query_image)
                processed_test_cases_for_map_no_qry.append(
                    {"prediction": prediction_no_qry, "label": true_labels, "id": case_idx}
                )
            
            if "map_attr" in self.metric_list:
                assert "query_attr" in case, "query_attr is required for map_attr"
                processed_test_cases_for_map_attr[case["query_attr"]].append(
                    {"prediction": prediction, "label": true_labels, "id": case_idx}
                )
                prediction_no_qry, prediction_scores_no_qry = self.remove_query(prediction, case["prediction_scores"], query_attr, query_image)
                processed_test_cases_for_map_attr_no_qry[case["query_attr"]].append(
                    {"prediction": prediction_no_qry, "label": true_labels, "id": case_idx}
                )

            for k in self.k_list:
                if "precision" in self.metric_list:
                    metric_results_accumulators["precision"][k].append(self.precision_at_k(prediction, true_labels, k))
                if "recall" in self.metric_list:
                    metric_results_accumulators["recall"][k].append(self.recall_at_k(prediction, true_labels, k))
                if "hit" in self.metric_list:
                    metric_results_accumulators["hit"][k].append(self.hit_at_k(prediction, true_labels, k))
                if "f1" in self.metric_list:
                    metric_results_accumulators["f1"][k].append(self.f1_at_k(prediction, true_labels, k))
                if "ndcg" in self.metric_list:
                    rel_scores = case["rel_scores"]
                    metric_results_accumulators["ndcg_linear"][k].append(
                        self.ndcg_at_k(prediction, true_labels, k, rel_scores, form="linear"))
                    metric_results_accumulators["ndcg_exponential"][k].append(
                        self.ndcg_at_k(prediction, true_labels, k, rel_scores, form="exponential"))
                if "gt_avg" in self.metric_list:
                    prediction_scores = case["prediction_scores"]
                    metric_results_accumulators["gt_avg"][k].append(self.gt_avg(prediction, prediction_scores, true_labels, k))

                    query_attr = case["query_attr"]
                    query_image = case["qry_img_path"].split("/")[-1]
                    prediction_no_qry_img, prediction_scores_no_qry_img = self.remove_query(prediction, prediction_scores, query_attr, query_image)
                    metric_results_accumulators["gt_avg_no_qry_img"][k].append(self.gt_avg(prediction_no_qry_img, prediction_scores_no_qry_img, true_labels, k))

            if "first_err" in self.metric_list:
                prediction_scores = case["prediction_scores"]
                query_attr = case["query_attr"]
                query_image = case["qry_img_path"].split("/")[-1]
                metric_results_accumulators["first_err"].append(self.first_err(prediction, prediction_scores, query_attr, query_image))
            
            if "PN_gap" in self.metric_list:
                prediction_scores = case["prediction_scores"]
                query_attr = case["query_attr"]
                query_image = case["qry_img_path"].split("/")[-1]
                metric_results_accumulators["PN_gap_first_not_qry_img"].append(self.PN_gap(prediction, prediction_scores, true_labels, query_attr, query_image))
                metric_results_accumulators["PN_gap_first_gt"].append(self.PN_gap_no_qry_img(prediction, prediction_scores, true_labels, query_attr, query_image))

        # Calculate final mean for most metrics
        score_dict = {}
        for metric, k_values_dict in metric_results_accumulators.items():
            if metric in metric_k_set: # other metrics are calculated differently
                for k, values in k_values_dict.items():
                    score_dict[f"{metric}@{k}"] = float(np.mean(values)) if values else 0.0
                    if "gt_avg" in metric:
                        score_dict[f"{metric}@{k}_mean"] = float(np.mean(values)) if values else 0.0
                        score_dict[f"{metric}@{k}_std"] = float(np.std(values)) if values else 0.0
            elif "PN_gap" in metric or 'first_err' in metric:
                score_dict[f"{metric}_mean"] = float(np.mean(metric_results_accumulators[metric])) if metric_results_accumulators[metric] else 0.0
                score_dict[f"{metric}_std"] = float(np.std(metric_results_accumulators[metric])) if metric_results_accumulators[metric] else 0.0

        # Calculate MAP and MRR across all test cases for each k
        if "map" in self.metric_list:
            for k_val in self.k_list:
                ap_scores = [
                    self.average_precision_at_k(case["prediction"], case["label"], k_val)
                    for case in processed_test_cases_for_map_mrr
                ]
                score_dict[f"map@{k_val}"] = float(np.mean(ap_scores)) if ap_scores else 0.0

                ap_scores_no_qry = [
                    self.average_precision_at_k(case["prediction"], case["label"], k_val)
                    for case in processed_test_cases_for_map_no_qry
                ]
                score_dict[f"map_no_qry@{k_val}"] = float(np.mean(ap_scores_no_qry)) if ap_scores_no_qry else 0.0

        if "mrr" in self.metric_list:
            for k_val in self.k_list:
                rr_scores = []
                for case in processed_test_cases_for_map_mrr:
                    reciprocal_rank = 0.0
                    if case["label"]: # Only if there are true labels
                        for rank, p_item in enumerate(case["prediction"][:k_val]):
                            if p_item in case["label"]:
                                reciprocal_rank = 1.0 / (rank + 1)
                                break
                    rr_scores.append(reciprocal_rank)
                score_dict[f"mrr@{k_val}"] = float(np.mean(rr_scores)) if rr_scores else 0.0
        
        if "map_all" in self.metric_list:
            ap_scores = [
                self.average_precision_at_k(case["prediction"], case["label"], len(case["label"]))
                for case in processed_test_cases_for_map_mrr
            ]
            score_dict[f"map_all"] = float(np.mean(ap_scores)) if ap_scores else 0.0
            
            ap_scores_no_qry = [
                self.average_precision_at_k(case["prediction"], case["label"], len(case["label"]))
                for case in processed_test_cases_for_map_no_qry
            ]
            score_dict[f"map_all_no_qry"] = float(np.mean(ap_scores_no_qry)) if ap_scores_no_qry else 0.0

        if "map_attr" in self.metric_list:
            for attr, cases in processed_test_cases_for_map_attr.items():
                ap_scores = [
                    self.average_precision_at_k(case["prediction"], case["label"], len(case["label"]))
                    for case in cases
                ]
                score_dict[f"map_attr@{attr}"] = float(np.mean(ap_scores)) if ap_scores else 0.0

            for attr, cases in processed_test_cases_for_map_attr_no_qry.items():
                ap_scores = [
                    self.average_precision_at_k(case["prediction"], case["label"], len(case["label"]))
                    for case in cases
                ]
                score_dict[f"map_attr@{attr}_no_qry"] = float(np.mean(ap_scores)) if ap_scores else 0.0

        return score_dict

    def remove_query(self, prediction, prediction_scores, query_attr, query_image):
        """
        Remove the query from the prediction and prediction_scores
        """
        # get index of query
        if f"{query_image}:{query_attr}" not in prediction:
            return prediction, prediction_scores
        query_index = prediction.index(f"{query_image}:{query_attr}")
        # remove the query from the prediction and prediction_scores
        prediction_no_qry_img = prediction[:query_index] + prediction[query_index+1:]
        prediction_scores_no_qry_img = prediction_scores[:query_index] + prediction_scores[query_index+1:]
        return prediction_no_qry_img, prediction_scores_no_qry_img