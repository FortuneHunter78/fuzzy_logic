import math
import random
from typing import List, Dict, Tuple
from fuzzy_core import ExponentialFuzzyVariable


class FuzzyClusterAnalyzer:

    @staticmethod
    def hausdorff_distance(int1: Tuple[float, float], int2: Tuple[float, float]) -> float:
        return max(abs(int1[0] - int2[0]), abs(int1[1] - int2[1]))

    @classmethod
    def filter_clusters_by_min_size(
        cls,
        raw_clusters: Dict[int, List[ExponentialFuzzyVariable]],
        min_size: int = 1
    ) -> Tuple[Dict[int, List[ExponentialFuzzyVariable]], List[ExponentialFuzzyVariable]]:

        if min_size <= 1:
            valid = {cid: items for cid, items in raw_clusters.items() if items}
            return valid, []

        valid_clusters: Dict[int, List[ExponentialFuzzyVariable]] = {}
        noise_items: List[ExponentialFuzzyVariable] = []
        new_cid = 1

        for _, items in raw_clusters.items():
            if len(items) >= min_size:
                valid_clusters[new_cid] = items
                new_cid += 1
            else:
                noise_items.extend(items)

        return valid_clusters, noise_items

    @classmethod
    def cluster_by_k_means(
        cls,
        dataset: List[ExponentialFuzzyVariable],
        k: int,
        alpha: float = 0.7,
        min_cluster_size: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int = 2026
    ) -> Tuple[Dict[int, List[ExponentialFuzzyVariable]], List[ExponentialFuzzyVariable]]:

        if k <= 0 or k > len(dataset):
            raise ValueError("Число кластеров K должно быть от 1 до размера выборки.")

        if seed is not None:
            random.seed(seed)

        intervals = [fv.get_alpha_cut(alpha) for fv in dataset]

        sorted_indices = sorted(range(len(dataset)), key=lambda idx: intervals[idx][1])
        step = max(1, len(dataset) // k)
        centroids: List[Tuple[float, float]] = [
            intervals[sorted_indices[min(i * step, len(dataset) - 1)]] for i in range(k)
        ]

        clusters: Dict[int, List[int]] = {i: [] for i in range(k)}

        for _ in range(max_iter):
            new_clusters: Dict[int, List[int]] = {i: [] for i in range(k)}
            for idx, inter in enumerate(intervals):
                distances = [cls.hausdorff_distance(inter, c) for c in centroids]
                closest_cluster = distances.index(min(distances))
                new_clusters[closest_cluster].append(idx)

            new_centroids: List[Tuple[float, float]] = []
            max_shift = 0.0
            for cid in range(k):
                pts = [intervals[idx] for idx in new_clusters[cid]]
                if pts:
                    mean_a = sum(p[0] for p in pts) / len(pts)
                    mean_b = sum(p[1] for p in pts) / len(pts)
                    new_c = (mean_a, mean_b)
                else:
                    new_c = random.choice(intervals)
                max_shift = max(max_shift, cls.hausdorff_distance(centroids[cid], new_c))
                new_centroids.append(new_c)

            clusters = new_clusters
            centroids = new_centroids
            if max_shift < tol:
                break

        sorted_cids = sorted(
            clusters.keys(),
            key=lambda cid: (centroids[cid][0] + centroids[cid][1]) / 2.0 if clusters[cid] else float('inf')
        )
        raw_result: Dict[int, List[ExponentialFuzzyVariable]] = {}
        for new_id, old_id in enumerate(sorted_cids, start=1):
            raw_result[new_id] = [dataset[idx] for idx in clusters[old_id]]

        return cls.filter_clusters_by_min_size(raw_result, min_cluster_size)

    @classmethod
    def cluster_by_criterion(
        cls,
        dataset: List[ExponentialFuzzyVariable],
        threshold: float,
        alpha: float = 0.7,
        min_cluster_size: int = 4
    ) -> Tuple[Dict[int, List[ExponentialFuzzyVariable]], List[ExponentialFuzzyVariable]]:

        if threshold <= 0:
            raise ValueError("Порог критерия должен быть больше 0.")

        sorted_data = sorted(dataset, key=lambda fv: fv.get_alpha_cut(alpha)[1])
        raw_clusters: Dict[int, List[ExponentialFuzzyVariable]] = {}
        if not sorted_data:
            return raw_clusters, []

        current_cid = 1
        raw_clusters[current_cid] = [sorted_data[0]]
        base_int = sorted_data[0].get_alpha_cut(alpha)

        for fv in sorted_data[1:]:
            cur_int = fv.get_alpha_cut(alpha)
            if cls.hausdorff_distance(cur_int, base_int) <= threshold:
                raw_clusters[current_cid].append(fv)
            else:
                current_cid += 1
                raw_clusters[current_cid] = [fv]
                base_int = cur_int

        return cls.filter_clusters_by_min_size(raw_clusters, min_cluster_size)

    @classmethod
    def generate_detailed_cluster_logs(
        cls,
        clusters: Dict[int, List[ExponentialFuzzyVariable]],
        noise_items: List[ExponentialFuzzyVariable],
        alpha: float,
        params_str: str = "k=5, max_iter=100, tol=1e-06, seed=2026"
    ) -> List[str]:
        clustered_count = sum(len(items) for items in clusters.values())
        noise_count = len(noise_items)
        total_elements = clustered_count + noise_count
        pct = (clustered_count / total_elements * 100.0) if total_elements > 0 else 0.0

        non_empty_clusters = {cid: items for cid, items in clusters.items() if len(items) > 0}
        k_actual = len(non_empty_clusters)
        sizes = [len(items) for items in non_empty_clusters.values()]

        logs = [
            f"[CLUSTERING] Запуск: параметры=({params_str})",
            f"[CLUSTERING] Разбиение: всего={total_elements}; кластеризовано={clustered_count} ({pct:.1f}%); "
            f"кластеров={k_actual}; отсеяно_шума={noise_count}; распределение={sizes}"
        ]

        if k_actual == 0:
            logs.append("[CLUSTERING] Все элементы были отсеяны критерием минимального размера кластера.")
            return logs

        cluster_stats = {}
        total_inertia = 0.0
        var_centers_list = []
        max_diameter = 0.0

        cluster_intervals: Dict[int, List[Tuple[float, float]]] = {}
        for cid, items in non_empty_clusters.items():
            cluster_intervals[cid] = [fv.get_alpha_cut(alpha) for fv in items]

        for cid, intervals in cluster_intervals.items():
            n = len(intervals)
            mean_a = sum(p[0] for p in intervals) / n
            mean_b = sum(p[1] for p in intervals) / n
            centroid = (mean_a, mean_b)
            c_val = (mean_a + mean_b) / 2.0
            w_val = mean_b - mean_a
            lam_c = -math.log(alpha) / mean_b if mean_b > 0 else 0.0

            inertia = sum(cls.hausdorff_distance(p, centroid) ** 2 for p in intervals)
            total_inertia += inertia

            centers = [(p[0] + p[1]) / 2.0 for p in intervals]
            var_c = sum((c - c_val) ** 2 for c in centers) / n
            var_centers_list.append(var_c)

            diameter = 0.0
            if n > 1:
                min_a_cl = min(p[0] for p in intervals)
                max_a_cl = max(p[0] for p in intervals)
                min_b_cl = min(p[1] for p in intervals)
                max_b_cl = max(p[1] for p in intervals)
                diameter = max(abs(max_a_cl - min_a_cl), abs(max_b_cl - min_b_cl))
            max_diameter = max(max_diameter, diameter)

            cluster_stats[cid] = {
                "size": n,
                "centroid": centroid,
                "lam_c": lam_c,
                "c": c_val,
                "w": w_val,
                "inertia": inertia,
                "var_c": var_c,
                "diameter": diameter
            }

        mean_var_centers = sum(var_centers_list) / len(var_centers_list) if var_centers_list else 0.0

        silhouette_vals = []
        if k_actual > 1:
            for cid, intervals in cluster_intervals.items():
                for i, p_i in enumerate(intervals):
                    if len(intervals) > 1:
                        a_i = sum(cls.hausdorff_distance(p_i, p_j) for j, p_j in enumerate(intervals) if i != j) / (len(intervals) - 1)
                    else:
                        a_i = 0.0

                    b_i = float('inf')
                    for other_cid, other_intervals in cluster_intervals.items():
                        if other_cid == cid or not other_intervals:
                            continue
                        mean_dist_other = sum(cls.hausdorff_distance(p_i, p_other) for p_other in other_intervals) / len(other_intervals)
                        if mean_dist_other < b_i:
                            b_i = mean_dist_other

                    denom = max(a_i, b_i)
                    s_i = (b_i - a_i) / denom if denom > 0 else 0.0
                    silhouette_vals.append(s_i)

        mean_silhouette = sum(silhouette_vals) / len(silhouette_vals) if silhouette_vals else 0.0

        logs.append(
            f"[CLUSTERING] Метрики качества: сумм.инерция={total_inertia:.3f}; "
            f"ср.дисперсия_центров={mean_var_centers:.5f}; макс.диаметр={max_diameter:.5f}; силуэт={mean_silhouette:.4f}"
        )

        for cid, st in cluster_stats.items():
            ca, cb = st["centroid"]
            logs.append(
                f"[CLUSTER #{cid}] n={st['size']}; центроид=[{ca:.3f}, {cb:.3f}]; центр={st['c']:.3f}; "
                f"ширина={st['w']:.3f}; лямбда_ц={st['lam_c']:.3f}; инерция={st['inertia']:.3f}; "
                f"дисперсия={st['var_c']:.5f}; диаметр={st['diameter']:.5f}"
            )

        return logs