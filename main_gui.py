import colorsys
import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from fuzzy_core import ExponentialFuzzyVariable
from fuzzy_generator import FuzzySetGenerator, FuzzyLogger
from fuzzy_clustering import FuzzyClusterAnalyzer


class FuzzyAppGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Комплекс программ: Экспоненциальные нечёткие величины и кластерный анализ")
        self.root.geometry("1420x870")

        self.logger = FuzzyLogger("fuzzy_journal.log")
        self.dataset = []
        self.clusters = {}
        self.noise_items = []
        self.cluster_colors_rgb = []
        self.last_cluster_mode = None
        self._slider_job = None

        self._build_ui()
        self.log_msg("[SYSTEM] Программа запущена. Готова к генерации нечётких величин.")

    @staticmethod
    def generate_contrasting_colors(n: int):
        rgb_list = []
        hex_list = []
        golden_ratio_conjugate = 0.618033988749895

        for i in range(n):
            hue = (i * golden_ratio_conjugate) % 1.0
            sat = 0.85 if i % 2 == 0 else 0.68
            val = 0.88 if i % 3 != 0 else 0.65

            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            rgb_list.append((r, g, b))
            hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            hex_list.append(hex_color)

        return rgb_list, hex_list

    def _build_ui(self):
        top_container = ttk.Frame(self.root)
        top_container.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        gen_frame = ttk.LabelFrame(top_container, text="1. Генерация множества нечётких величин", padding=8)
        gen_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Label(gen_frame, text="Кол-во (100-300):").grid(row=0, column=0, padx=3)
        self.ent_count = ttk.Entry(gen_frame, width=6)
        self.ent_count.insert(0, "150")
        self.ent_count.grid(row=0, column=1, padx=3)

        ttk.Label(gen_frame, text="λ min:").grid(row=0, column=2, padx=3)
        self.ent_lmin = ttk.Entry(gen_frame, width=5)
        self.ent_lmin.insert(0, "0.2")
        self.ent_lmin.grid(row=0, column=3, padx=3)

        ttk.Label(gen_frame, text="λ max:").grid(row=0, column=4, padx=3)
        self.ent_lmax = ttk.Entry(gen_frame, width=5)
        self.ent_lmax.insert(0, "2.5")
        self.ent_lmax.grid(row=0, column=5, padx=3)

        ttk.Label(gen_frame, text="Seed:").grid(row=0, column=6, padx=3)
        self.ent_seed = ttk.Entry(gen_frame, width=6)
        self.ent_seed.insert(0, "2026")
        self.ent_seed.grid(row=0, column=7, padx=3)

        self.var_norm = tk.BooleanVar(value=True)
        ttk.Checkbutton(gen_frame, text="Нормировать к 1", variable=self.var_norm).grid(row=0, column=8, padx=5)

        ttk.Button(gen_frame, text="Сгенерировать выборку", command=self.on_generate).grid(row=0, column=9, padx=6)

        alpha_frame = ttk.LabelFrame(top_container, text="Уровень отсечения α", padding=8)
        alpha_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        ttk.Label(alpha_frame, text="α =").grid(row=0, column=0, padx=3)
        self.ent_alpha = ttk.Entry(alpha_frame, width=6)
        self.ent_alpha.insert(0, "0.70")
        self.ent_alpha.grid(row=0, column=1, padx=3)
        self.ent_alpha.bind("<Return>", lambda event: self.on_alpha_entry_apply())

        self.scale_alpha = tk.Scale(
            alpha_frame, from_=0.05, to=0.99, resolution=0.01,
            orient=tk.HORIZONTAL, showvalue=False, length=210,
            command=self.on_alpha_slider_move
        )
        self.scale_alpha.set(0.70)
        self.scale_alpha.grid(row=0, column=2, padx=6)

        ttk.Button(alpha_frame, text="Применить α", command=self.on_alpha_entry_apply).grid(row=0, column=3, padx=4)

        mid_frame = ttk.Frame(self.root)
        mid_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

        clust_frame = ttk.LabelFrame(mid_frame, text="2. Кластерный анализ по отрезкам [A; B]", padding=8)
        clust_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Label(clust_frame, text="Число K:").grid(row=0, column=0, padx=3)
        self.ent_k = ttk.Entry(clust_frame, width=4)
        self.ent_k.insert(0, "5")
        self.ent_k.grid(row=0, column=1, padx=3)
        ttk.Button(clust_frame, text="K-Means", command=self.on_cluster_kmeans).grid(row=0, column=2, padx=5)

        ttk.Label(clust_frame, text="Порог ΔB:").grid(row=0, column=3, padx=3)
        self.ent_thresh = ttk.Entry(clust_frame, width=5)
        self.ent_thresh.insert(0, "0.18")
        self.ent_thresh.grid(row=0, column=4, padx=3)
        ttk.Button(clust_frame, text="По критерию ΔB", command=self.on_cluster_criterion).grid(row=0, column=5, padx=5)

        ttk.Separator(clust_frame, orient=tk.VERTICAL).grid(row=0, column=6, sticky="ns", padx=6)
        ttk.Label(clust_frame, text="Мин. элементов в кластере (отсев):").grid(row=0, column=7, padx=3)
        self.ent_min_size = ttk.Entry(clust_frame, width=4)
        self.ent_min_size.insert(0, "4")
        self.ent_min_size.grid(row=0, column=8, padx=3)

        op_frame = ttk.LabelFrame(mid_frame, text="3. Манипулирование величинами", padding=8)
        op_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        ttk.Label(op_frame, text="λ1:").grid(row=0, column=0, padx=3)
        self.ent_op_l1 = ttk.Entry(op_frame, width=5)
        self.ent_op_l1.insert(0, "0.5")
        self.ent_op_l1.grid(row=0, column=1, padx=3)

        ttk.Label(op_frame, text="λ2:").grid(row=0, column=2, padx=3)
        self.ent_op_l2 = ttk.Entry(op_frame, width=5)
        self.ent_op_l2.insert(0, "1.2")
        self.ent_op_l2.grid(row=0, column=3, padx=3)

        ttk.Button(op_frame, text="Выполнить (+, -, *, /, сравнение)", command=self.on_calculate_ops).grid(row=0, column=4, padx=6)

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        plot_frame = ttk.LabelFrame(main_pane, text="Визуализация графиков Э(x; λ), линии отсечения α и кластеров")
        main_pane.add(plot_frame, weight=3)

        self.fig, (self.ax_curves, self.ax_intervals) = plt.subplots(2, 1, figsize=(8, 6), dpi=100)
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=2)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        tab_vars = ttk.Frame(self.notebook)
        self.notebook.add(tab_vars, text=" Все величины (Таблица) ")

        cols = ("name", "lam", "interval", "x_star", "cluster")
        self.tree = ttk.Treeview(tab_vars, columns=cols, show="headings", height=10)
        self.tree.heading("name", text="Имя")
        self.tree.heading("lam", text="Лямбда (λ)")
        self.tree.heading("interval", text="Отрезок [A; B]")
        self.tree.heading("x_star", text="x* = -1/λ ln(ξ)")
        self.tree.heading("cluster", text="Статус / Кластер")

        for c, w in zip(cols, (65, 80, 125, 95, 95)):
            self.tree.column(c, width=w, anchor=tk.CENTER)

        scrollbar_vars = ttk.Scrollbar(tab_vars, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar_vars.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_vars.pack(side=tk.RIGHT, fill=tk.Y)

        tab_legend = ttk.Frame(self.notebook)
        self.notebook.add(tab_legend, text=" Легенда кластеров ")

        leg_cols = ("swatch", "cid", "count", "b_range", "lam_c")
        self.tree_legend = ttk.Treeview(tab_legend, columns=leg_cols, show="headings", height=10)
        self.tree_legend.heading("swatch", text="Цвет")
        self.tree_legend.heading("cid", text="Кластер / Группа")
        self.tree_legend.heading("count", text="Кол-во")
        self.tree_legend.heading("b_range", text="Диапазон B (от..до)")
        self.tree_legend.heading("lam_c", text="Эталон λ")

        for c, w in zip(leg_cols, (55, 95, 65, 135, 75)):
            self.tree_legend.column(c, width=w, anchor=tk.CENTER)

        scrollbar_leg = ttk.Scrollbar(tab_legend, orient=tk.VERTICAL, command=self.tree_legend.yview)
        self.tree_legend.configure(yscroll=scrollbar_leg.set)
        self.tree_legend.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_leg.pack(side=tk.RIGHT, fill=tk.Y)

        log_frame = ttk.LabelFrame(right_frame, text="Экранный журнал (Лог-файл: fuzzy_journal.log)")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.txt_log = tk.Text(log_frame, height=14, wrap=tk.NONE, font=("Consolas", 8))
        self.txt_log.tag_config("INFO", foreground="black")
        self.txt_log.tag_config("WARNING", foreground="#b35900")
        self.txt_log.tag_config("ERROR", foreground="#cc0000")

        log_scroll_y = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.txt_log.yview)
        log_scroll_x = ttk.Scrollbar(log_frame, orient=tk.HORIZONTAL, command=self.txt_log.xview)
        self.txt_log.configure(yscrollcommand=log_scroll_y.set, xscrollcommand=log_scroll_x.set)

        log_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        log_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    def log_msg(self, msg: str, level: str = "INFO"):
        entry = self.logger.log(msg, level=level)
        self.txt_log.insert(tk.END, entry + "\n", level)
        self.txt_log.see(tk.END)

    def log_error(self, context: str, err: Exception, raw_params: str = ""):
        param_info = f" (входные данные: {raw_params})" if raw_params else ""
        entry = self.logger.error(context, Exception(f"{str(err)}{param_info}"))
        self.txt_log.insert(tk.END, entry + "\n", "ERROR")
        self.txt_log.see(tk.END)

    def log_warning(self, context: str, msg: str):
        entry = self.logger.warning(context, msg)
        self.txt_log.insert(tk.END, entry + "\n", "WARNING")
        self.txt_log.see(tk.END)

    def _build_cluster_map(self) -> dict:
        cluster_map = {}
        for cid, items in self.clusters.items():
            for fv in items:
                cluster_map[fv.name] = f"Кластер {cid}"
        for fv in self.noise_items:
            cluster_map[fv.name] = "Отсеяно"
        return cluster_map

    def update_table(self, alpha: float, cluster_map: dict = None):
        for row in self.tree.get_children():
            self.tree.delete(row)

        for fv in self.dataset:
            A, B = fv.get_alpha_cut(alpha)
            cl_str = str(cluster_map.get(fv.name, "-")) if cluster_map else "-"
            self.tree.insert("", tk.END, values=(
                fv.name,
                f"{fv.lam:.4f}",
                f"[{A:.3f}; {B:.3f}]",
                f"{fv.x_star:.4f}",
                cl_str
            ))

    def update_legend_tab(self, alpha: float, hex_colors: list):
        for row in self.tree_legend.get_children():
            self.tree_legend.delete(row)

        if not self.clusters and not self.noise_items:
            return

        for idx_c, (cid, items) in enumerate(self.clusters.items()):
            hex_col = hex_colors[idx_c]
            tag_name = f"cluster_color_{cid}"
            self.tree_legend.tag_configure(tag_name, foreground=hex_col)

            if items:
                b_vals = [fv.get_alpha_cut(alpha)[1] for fv in items]
                min_b, max_b = min(b_vals), max(b_vals)
                mean_b = sum(b_vals) / len(b_vals)
                lam_c = -math.log(alpha) / mean_b if mean_b > 0 else 0.0
                b_range_str = f"[{min_b:.3f} .. {max_b:.3f}]"
            else:
                lam_c = 0.0
                b_range_str = "[-]"

            self.tree_legend.insert(
                "", tk.END,
                values=("████████", f"Кластер #{cid}", f"{len(items)} шт.", b_range_str, f"{lam_c:.3f}"),
                tags=(tag_name,)
            )

        if self.noise_items:
            self.tree_legend.tag_configure("noise_tag", foreground="#808080")
            b_vals = [fv.get_alpha_cut(alpha)[1] for fv in self.noise_items]
            b_range_str = f"[{min(b_vals):.3f} .. {max(b_vals):.3f}]"
            self.tree_legend.insert(
                "", tk.END,
                values=("▒▒▒▒▒▒▒▒", "Отсеяно", f"{len(self.noise_items)} шт.", b_range_str, "-"),
                tags=("noise_tag",)
            )

    def on_alpha_slider_move(self, val_str: str):
        alpha = float(val_str)
        self.ent_alpha.delete(0, tk.END)
        self.ent_alpha.insert(0, f"{alpha:.2f}")

        if not self.dataset:
            return

        if self._slider_job is not None:
            self.root.after_cancel(self._slider_job)
        self._slider_job = self.root.after(40, lambda: self._recalculate_for_new_alpha(alpha, log_change=False))

    def on_alpha_entry_apply(self):
        raw_alpha = self.ent_alpha.get()
        try:
            alpha = float(raw_alpha)
            if not (0 < alpha <= 1.0):
                raise ValueError("Уровень отсечения alpha должен находиться в интервале (0; 1].")
            self.scale_alpha.set(alpha)
            if self.dataset:
                self._recalculate_for_new_alpha(alpha, log_change=True)
        except Exception as e:
            self.log_error("ALPHA_CUT", e, f"alpha='{raw_alpha}'")
            messagebox.showerror("Ошибка уровня отсечения", str(e))

    def _recalculate_for_new_alpha(self, alpha: float, log_change: bool = True):
        if not self.dataset:
            return

        if self.last_cluster_mode == "kmeans":
            k = int(self.ent_k.get())
            min_sz = int(self.ent_min_size.get())
            seed = int(self.ent_seed.get())
            self.clusters, self.noise_items = FuzzyClusterAnalyzer.cluster_by_k_means(
                self.dataset, k, alpha, min_cluster_size=min_sz, seed=seed
            )
            rgb_colors, hex_colors = self.generate_contrasting_colors(len(self.clusters))
            self.cluster_colors_rgb = rgb_colors
            self.update_table(alpha, self._build_cluster_map())
            self.update_legend_tab(alpha, hex_colors)
        elif self.last_cluster_mode == "criterion":
            thresh = float(self.ent_thresh.get())
            min_sz = int(self.ent_min_size.get())
            self.clusters, self.noise_items = FuzzyClusterAnalyzer.cluster_by_criterion(
                self.dataset, thresh, alpha, min_cluster_size=min_sz
            )
            rgb_colors, hex_colors = self.generate_contrasting_colors(len(self.clusters))
            self.cluster_colors_rgb = rgb_colors
            self.update_table(alpha, self._build_cluster_map())
            self.update_legend_tab(alpha, hex_colors)
        else:
            self.update_table(alpha)

        self.draw_plots(alpha)

        if log_change:
            sample_logs = FuzzySetGenerator.get_sample_stats_logs(self.dataset, alpha, seed=int(self.ent_seed.get()))
            self.log_msg(f"[ALPHA_CUT] Уровень отсечения изменен в реальном времени: alpha={alpha}")
            self.log_msg(sample_logs[1])

    def on_generate(self):
        raw_params = (
            f"count='{self.ent_count.get()}', lmin='{self.ent_lmin.get()}', "
            f"lmax='{self.ent_lmax.get()}', alpha='{self.ent_alpha.get()}', seed='{self.ent_seed.get()}'"
        )
        try:
            count = int(self.ent_count.get())
            lmin = float(self.ent_lmin.get())
            lmax = float(self.ent_lmax.get())
            alpha = float(self.ent_alpha.get())
            seed = int(self.ent_seed.get())
            norm = self.var_norm.get()

            if not (0 < alpha <= 1.0):
                raise ValueError("Уровень отсечения alpha должен находиться в интервале (0; 1].")

            if not norm and lmin <= alpha:
                self.log_warning(
                    "GENERATOR",
                    f"Режим без нормировки: часть величин с lambda <= alpha ({alpha}) даст вырожденные отрезки [0, 0]."
                )

            self.dataset = FuzzySetGenerator.generate_set(count, lmin, lmax, norm, seed=seed)
            self.clusters = {}
            self.noise_items = []
            self.last_cluster_mode = None

            self.update_table(alpha)
            self.update_legend_tab(alpha, [])
            self.draw_plots(alpha)

            sample_logs = FuzzySetGenerator.get_sample_stats_logs(self.dataset, alpha, seed=seed)
            for line in sample_logs:
                self.log_msg(line)

        except Exception as e:
            self.log_error("GENERATOR", e, raw_params)
            messagebox.showerror("Ошибка генерации выборки", str(e))

    def on_cluster_kmeans(self):
        if not self.dataset:
            self.log_warning("CLUSTERING", "Отказ запуска K-Means: выборка нечетких величин пуста.")
            messagebox.showwarning("Внимание", "Сначала сгенерируйте набор нечётких величин!")
            return

        raw_params = (
            f"k='{self.ent_k.get()}', min_size='{self.ent_min_size.get()}', "
            f"alpha='{self.ent_alpha.get()}', seed='{self.ent_seed.get()}'"
        )
        try:
            k = int(self.ent_k.get())
            min_sz = int(self.ent_min_size.get())
            if min_sz < 1:
                raise ValueError("Минимальный размер кластера для отсева должен быть >= 1.")
            alpha = float(self.ent_alpha.get())
            seed = int(self.ent_seed.get())
            max_iter = 100
            tol = 1e-6

            self.clusters, self.noise_items = FuzzyClusterAnalyzer.cluster_by_k_means(
                self.dataset, k, alpha, min_cluster_size=min_sz, max_iter=max_iter, tol=tol, seed=seed
            )
            self.last_cluster_mode = "kmeans"
            params_str = f"k={k}, min_size={min_sz}, max_iter={max_iter}, tol={tol:.0e}, seed={seed}"
            self._apply_clustering_results(alpha, params_str=params_str)
        except Exception as e:
            self.log_error("CLUSTERING:KMEANS", e, raw_params)
            messagebox.showerror("Ошибка кластеризации K-Means", str(e))

    def on_cluster_criterion(self):
        if not self.dataset:
            self.log_warning("CLUSTERING", "Отказ запуска пороговой кластеризации: выборка пуста.")
            messagebox.showwarning("Внимание", "Сначала сгенерируйте набор нечётких величин!")
            return

        raw_params = (
            f"threshold='{self.ent_thresh.get()}', min_size='{self.ent_min_size.get()}', alpha='{self.ent_alpha.get()}'"
        )
        try:
            thresh = float(self.ent_thresh.get())
            min_sz = int(self.ent_min_size.get())
            if min_sz < 1:
                raise ValueError("Минимальный размер кластера для отсева должен быть >= 1.")
            alpha = float(self.ent_alpha.get())
            seed = int(self.ent_seed.get())

            self.clusters, self.noise_items = FuzzyClusterAnalyzer.cluster_by_criterion(
                self.dataset, thresh, alpha, min_cluster_size=min_sz
            )
            self.last_cluster_mode = "criterion"
            params_str = f"threshold={thresh}, min_size={min_sz}, alpha={alpha}, seed={seed}"
            self._apply_clustering_results(alpha, params_str=params_str)
        except Exception as e:
            self.log_error("CLUSTERING:CRITERION", e, raw_params)
            messagebox.showerror("Ошибка кластеризации по критерию", str(e))

    def _apply_clustering_results(self, alpha: float, params_str: str):
        rgb_colors, hex_colors = self.generate_contrasting_colors(len(self.clusters))
        self.cluster_colors_rgb = rgb_colors

        self.update_table(alpha, self._build_cluster_map())
        self.update_legend_tab(alpha, hex_colors)
        self.notebook.select(1)
        self.draw_plots(alpha)

        detailed_logs = FuzzyClusterAnalyzer.generate_detailed_cluster_logs(
            self.clusters, self.noise_items, alpha, params_str=params_str
        )
        for line in detailed_logs:
            self.log_msg(line)

    def on_calculate_ops(self):
        raw_params = f"l1='{self.ent_op_l1.get()}', l2='{self.ent_op_l2.get()}', alpha='{self.ent_alpha.get()}'"
        try:
            l1 = float(self.ent_op_l1.get())
            l2 = float(self.ent_op_l2.get())
            alpha = float(self.ent_alpha.get())
            norm = self.var_norm.get()

            fv1 = ExponentialFuzzyVariable(l1, normalized=norm, name="X1")
            fv2 = ExponentialFuzzyVariable(l2, normalized=norm, name="X2")

            fv_add, int_add = fv1.add(fv2, alpha)
            fv_sub, int_sub = fv1.subtract(fv2, alpha)
            fv_mul, int_mul = fv1.multiply(fv2, alpha)
            fv_div, int_div = fv1.divide(fv2, alpha)
            cmp_res = fv1.compare(fv2, alpha)

            self.log_msg(f"[ARITHMETIC] Исходные данные: alpha={alpha}; X1(lam={l1:.3f})={fv1.get_alpha_cut(alpha)}; X2(lam={l2:.3f})={fv2.get_alpha_cut(alpha)}")
            self.log_msg(f"[ARITHMETIC] Сложение (X1+X2): интервал={int_add}; восст_лямбда={fv_add.lam:.4f}")
            self.log_msg(f"[ARITHMETIC] Вычитание (X1-X2): интервал={int_sub}; восст_лямбда={fv_sub.lam:.4f}")
            self.log_msg(f"[ARITHMETIC] Умножение (X1*X2): интервал={int_mul}; восст_лямбда={fv_mul.lam:.4f}")
            self.log_msg(f"[ARITHMETIC] Деление (X1/X2): интервал={int_div}; восст_лямбда={fv_div.lam:.4f}")
            self.log_msg(f"[ARITHMETIC] Сравнение: {cmp_res}")

            self.draw_operation_plots(alpha, fv1, fv2, fv_add, fv_sub, fv_mul, fv_div)

            msg = (
                f"ИСХОДНЫЕ ВЕЛИЧИНЫ (при α = {alpha}):\n"
                f"  • X1: λ1 = {fv1.lam:.4f}  -->  Отрезок {fv1.get_alpha_cut(alpha)}\n"
                f"  • X2: λ2 = {fv2.lam:.4f}  -->  Отрезок {fv2.get_alpha_cut(alpha)}\n\n"
                f"РЕЗУЛЬТАТЫ (Отрезок [A; B]  -->  Восстановленная λ_new):\n"
                f"  1) Сложение (X1 + X2):   {int_add}  -->  λ_new = {fv_add.lam:.4f}\n"
                f"  2) Вычитание (X1 - X2):  {int_sub}  -->  λ_new = {fv_sub.lam:.4f}\n"
                f"  3) Умножение (X1 * X2):  {int_mul}  -->  λ_new = {fv_mul.lam:.4f}\n"
                f"  4) Деление (X1 / X2):    {int_div}  -->  λ_new = {fv_div.lam:.4f}\n\n"
                f"СРАВНЕНИЕ:\n"
                f"  • {cmp_res}"
            )
            messagebox.showinfo("Результаты манипулирования и восстановленные λ", msg)
        except Exception as e:
            self.log_error("ARITHMETIC", e, raw_params)
            messagebox.showerror("Ошибка вычислений", str(e))

    def draw_operation_plots(self, alpha: float, fv1, fv2, fv_add, fv_sub, fv_mul, fv_div):
        self.ax_curves.clear()
        self.ax_intervals.clear()

        ops_list = [
            (fv1, f"Исходная X1 (λ={fv1.lam:.3f})", "black", "-"),
            (fv2, f"Исходная X2 (λ={fv2.lam:.3f})", "gray", "-"),
            (fv_add, f"Сумма X1+X2 (λ_new={fv_add.lam:.3f})", "green", "--"),
            (fv_sub, f"Разность X1-X2 (λ_new={fv_sub.lam:.3f})", "orange", "--"),
            (fv_mul, f"Умножение X1*X2 (λ_new={fv_mul.lam:.3f})", "blue", "-."),
            (fv_div, f"Деление X1/X2 (λ_new={fv_div.lam:.3f})", "purple", ":")
        ]

        max_b = max([item[0].get_alpha_cut(alpha)[1] for item in ops_list] + [2.0])
        x_vals = np.linspace(0, max_b * 1.4, 300)

        for idx, (fv, label, color, style) in enumerate(ops_list, start=1):
            y_vals = [fv.membership(x) for x in x_vals]
            self.ax_curves.plot(x_vals, y_vals, color=color, linestyle=style, linewidth=2.0, label=label)

            A, B = fv.get_alpha_cut(alpha)
            self.ax_intervals.hlines(y=idx, xmin=A, xmax=B, color=color, linewidth=4.0, label=f"{label}: [{A:.2f}; {B:.2f}]")
            self.ax_intervals.plot(B, idx, "o", color=color)

        self.ax_curves.axhline(y=alpha, color="red", linestyle="--", linewidth=1.5, label=f"Уровень отсечения α = {alpha}")
        self.ax_curves.set_title("Восстановленные графики Э(x; λ_new) после арифметических операций")
        self.ax_curves.set_xlabel("x")
        self.ax_curves.set_ylabel("Э(x; λ)")
        self.ax_curves.grid(True, linestyle=":", alpha=0.6)
        self.ax_curves.legend(loc="upper right", fontsize=8)

        self.ax_intervals.set_yticks(range(1, 7))
        self.ax_intervals.set_yticklabels(["X1", "X2", "X1+X2", "X1-X2", "X1*X2", "X1/X2"])
        self.ax_intervals.set_title(f"Отрезки [A; B] при α = {alpha}, из которых восстановлены λ_new")
        self.ax_intervals.set_xlabel("Отрезок [A; B] по оси X")
        self.ax_intervals.grid(True, linestyle=":", alpha=0.6)
        self.ax_intervals.legend(loc="upper right", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def draw_plots(self, alpha: float):
        self.ax_curves.clear()
        self.ax_intervals.clear()

        if not self.dataset:
            self.canvas.draw()
            return

        min_lam = min(fv.lam for fv in self.dataset)
        fixed_x_max = max(3.0, 1.5 / min_lam)
        x_vals = np.linspace(0, fixed_x_max, 250)

        self.ax_curves.axhline(
            y=alpha, color="red", linestyle="--", linewidth=1.8, label=f"Уровень отсечения α = {alpha:.2f}"
        )

        if not self.clusters and not self.noise_items:
            for fv in self.dataset:
                y_vals = [fv.membership(x) for x in x_vals]
                self.ax_curves.plot(x_vals, y_vals, color="steelblue", alpha=0.35, linewidth=1.0)

            for idx, fv in enumerate(self.dataset):
                A, B = fv.get_alpha_cut(alpha)
                self.ax_intervals.hlines(y=idx + 1, xmin=A, xmax=B, color="steelblue", alpha=0.7, linewidth=1.5)
        else:
            y_pos = 1
            for idx_c, (cid, items) in enumerate(self.clusters.items()):
                color = self.cluster_colors_rgb[idx_c]
                for fv in items:
                    y_vals = [fv.membership(x) for x in x_vals]
                    self.ax_curves.plot(x_vals, y_vals, color=color, alpha=0.7, linewidth=1.3)

                    A, B = fv.get_alpha_cut(alpha)
                    self.ax_intervals.hlines(y=y_pos, xmin=A, xmax=B, color=color, alpha=0.9, linewidth=2.2)
                    y_pos += 1

            for i, fv in enumerate(self.noise_items):
                y_vals = [fv.membership(x) for x in x_vals]
                lbl = f"Отсеяно / шум ({len(self.noise_items)} шт.)" if i == 0 else None
                self.ax_curves.plot(x_vals, y_vals, color="#999999", linestyle=":", alpha=0.45, linewidth=1.0,
                                    label=lbl)

                A, B = fv.get_alpha_cut(alpha)
                self.ax_intervals.hlines(y=y_pos, xmin=A, xmax=B, color="#999999", linestyle=":", alpha=0.6,
                                         linewidth=1.5)
                y_pos += 1

        clustered_n = sum(len(v) for v in self.clusters.values())
        noise_n = len(self.noise_items)
        status_str = (
            f"Кластеризовано: {clustered_n}/{len(self.dataset)} шт., отсеяно: {noise_n} шт."
            if (self.clusters or self.noise_items) else "До кластеризации"
        )

        self.ax_curves.set_xlim(0, fixed_x_max)
        self.ax_intervals.set_xlim(0, fixed_x_max)

        self.ax_curves.set_title(f"Семейство из {len(self.dataset)} графиков Э(x; λ) | {status_str}")
        self.ax_curves.set_xlabel("x")
        self.ax_curves.set_ylabel("Э(x; λ)")
        self.ax_curves.grid(True, linestyle=":", alpha=0.6)
        self.ax_curves.legend(loc="upper right", fontsize=8.5)

        self.ax_intervals.set_title(
            f"Отрезки [A; B] при α = {alpha:.2f} (цветные — в кластерах, серый пунктир — отсеяны)")
        self.ax_intervals.set_xlabel("Отрезок [A; B] по оси X")
        self.ax_intervals.set_ylabel("Индекс величины")
        self.ax_intervals.grid(True, linestyle=":", alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = FuzzyAppGUI(root)
    root.mainloop()