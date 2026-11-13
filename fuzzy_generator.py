import random
import datetime
from typing import List
from fuzzy_core import ExponentialFuzzyVariable


class FuzzyLogger:

    def __init__(self, filename: str = "fuzzy_journal.log"):
        self.filename = filename

    def log(self, message: str, level: str = "INFO") -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{now}] [{level}] {message}"
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
        return entry

    def error(self, context: str, err: Exception) -> str:
        err_type = type(err).__name__
        return self.log(f"[{context}] Исключение {err_type}: {str(err)}", level="ERROR")

    def warning(self, context: str, message: str) -> str:
        return self.log(f"[{context}] {message}", level="WARNING")


class FuzzySetGenerator:

    @staticmethod
    def generate_set(
        count: int = 150,
        lam_min: float = 0.2,
        lam_max: float = 3.0,
        normalized: bool = True,
        seed: int = 2026
    ) -> List[ExponentialFuzzyVariable]:
        if not (10 <= count <= 1000):
            raise ValueError("Рекомендуемое количество величин по заданию: от 100 до 300.")
        if lam_min <= 0 or lam_min >= lam_max:
            raise ValueError("Некорректный диапазон лямбда (0 < lam_min < lam_max).")

        if seed is not None:
            random.seed(seed)

        dataset = []
        for i in range(1, count + 1):
            lam = random.uniform(lam_min, lam_max)
            xi_p = random.uniform(0.05, 0.99)
            fv = ExponentialFuzzyVariable(lam=lam, xi_p=xi_p, normalized=normalized, name=f"FV_{i}")
            dataset.append(fv)

        return dataset

    @staticmethod
    def get_sample_stats_logs(dataset: List[ExponentialFuzzyVariable], alpha: float, seed: int = 2026) -> List[str]:
        if not dataset:
            return []

        intervals = [fv.get_alpha_cut(alpha) for fv in dataset]
        min_a = min(a for a, _ in intervals)
        max_b = max(b for _, b in intervals)
        centers = [(a + b) / 2.0 for a, b in intervals]
        widths = [b - a for a, b in intervals]

        mean_center = sum(centers) / len(centers)
        mean_width = sum(widths) / len(widths)

        return [
            f"[GENERATOR] Сформирована выборка: объем={len(dataset)}; seed={seed}; alpha={alpha}",
            f"[GENERATOR] Статистика выборки: границы=[{min_a:.3f}, {max_b:.3f}]; ср.центр={mean_center:.5f}; ср.ширина={mean_width:.5f}"
        ]