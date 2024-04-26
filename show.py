import sys
import itertools
import random
from collections import defaultdict
import math
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QLabel, QComboBox, \
    QHBoxLayout
from PyQt5.QtCore import pyqtSignal, QThread


def generate_combinations(elements, k):
    return list(itertools.combinations(elements, k))


def subset_cover_graph(n_numbers, k, j, s):
    j_subsets = generate_combinations(n_numbers, j)
    k_combinations = generate_combinations(n_numbers, k)
    cover_graph = defaultdict(list)
    for k_comb in k_combinations:
        for j_sub in j_subsets:
            if any(set(subset).issubset(k_comb) for subset in generate_combinations(j_sub, s)):
                cover_graph[tuple(k_comb)].append(tuple(j_sub))
    return cover_graph


def all_j_subsets_covered(cover_graph, solution):
    all_j_subsets = set(itertools.chain(*cover_graph.values()))
    covered_j_subsets = set(itertools.chain(*[cover_graph[k] for k in solution]))
    return covered_j_subsets == all_j_subsets


def simulated_annealing(cover_graph, n_numbers, T=10000, T_min=0.001, alpha=0.99, time_limit=8):
    print("Switching to greedy algorithm due to time limit.")
    return greedy_set_cover(cover_graph)
    start_time = time.time()
    k_combinations = list(cover_graph.keys())
    random.shuffle(k_combinations)
    current_solution = k_combinations[:len(k_combinations) // 4]
    while not all_j_subsets_covered(cover_graph, current_solution):
        current_solution.append(random.choice([k for k in k_combinations if k not in current_solution]))
    current_energy = len(current_solution)
    while T > T_min:
        if time.time() - start_time > time_limit:
            print("Switching to greedy algorithm due to time limit.")
            return greedy_set_cover(cover_graph)
        for _ in range(50):
            new_solution = current_solution[:]
            if random.random() > 0.5 and len(new_solution) > 1:
                new_solution.remove(random.choice(new_solution))
            else:
                possible_additions = [k for k in k_combinations if k not in new_solution]
                if possible_additions:
                    new_solution.append(random.choice(possible_additions))
            if all_j_subsets_covered(cover_graph, new_solution):
                new_energy = len(new_solution)
                if new_energy < current_energy or math.exp((current_energy - new_energy) / T) > random.random():
                    current_solution = new_solution
                    current_energy = new_energy
        T *= alpha
    return current_solution


def greedy_set_cover(cover_graph):
    covered = set()
    selected_k_combs = []
    while any(j_sub not in covered for j_subsets in cover_graph.values() for j_sub in j_subsets):
        best_k_comb = max(cover_graph, key=lambda k: len(set(cover_graph[k]) - covered))
        selected_k_combs.append(best_k_comb)
        covered.update(cover_graph[best_k_comb])
    return selected_k_combs


class AlgorithmWorker(QThread):
    finished = pyqtSignal(list, list)

    def __init__(self, n_numbers, k, j, s):
        super().__init__()
        self.n_numbers = n_numbers
        self.k = k
        self.j = j
        self.s = s

    def run(self):
        cover_graph = subset_cover_graph(self.n_numbers, self.k, self.j, self.s)
        result = simulated_annealing(cover_graph, self.n_numbers)
        self.finished.emit(self.n_numbers, result)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Set Cover Solver')
        self.setGeometry(400, 100, 800, 600)
        layout = QVBoxLayout()

        params_layout = QHBoxLayout()
        self.m_input = self.create_combobox(range(45, 101))  # Range for m
        self.n_input = self.create_combobox(range(7, 26))  # Range for n
        self.k_input = self.create_combobox(range(4, 11))  # Range for k
        self.j_input = self.create_combobox([])
        self.s_input = self.create_combobox([])

        params_layout.addWidget(QLabel("m:"))
        params_layout.addWidget(self.m_input)
        params_layout.addWidget(QLabel("n:"))
        params_layout.addWidget(self.n_input)
        params_layout.addWidget(QLabel("k:"))
        params_layout.addWidget(self.k_input)
        params_layout.addWidget(QLabel("j:"))
        params_layout.addWidget(self.j_input)
        params_layout.addWidget(QLabel("s:"))
        params_layout.addWidget(self.s_input)
        layout.addLayout(params_layout)

        self.result_text_edit = QTextEdit()
        self.result_text_edit.setReadOnly(True)
        layout.addWidget(self.result_text_edit)

        btn = QPushButton('Start', self)
        btn.clicked.connect(self.start_thread)
        layout.addWidget(btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.k_input.currentIndexChanged.connect(self.update_j_options)
        self.j_input.currentIndexChanged.connect(self.update_s_options)

    def create_combobox(self, values):
        combobox = QComboBox()
        combobox.addItem("")  # Empty default option
        for value in values:
            combobox.addItem(str(value))
        combobox.setFixedWidth(150)
        return combobox

    def update_j_options(self):
        k = int(self.k_input.currentText()) if self.k_input.currentText() else 0
        self.j_input.clear()
        self.j_input.addItem("")
        for value in range(1, k + 1):
            self.j_input.addItem(str(value))

    def update_s_options(self):
        j = int(self.j_input.currentText()) if self.j_input.currentText() else 0
        self.s_input.clear()
        self.s_input.addItem("")
        for value in range(1, j + 1):
            self.s_input.addItem(str(value))

    def start_thread(self):
        m = int(self.m_input.currentText()) if self.m_input.currentText() else 0
        n = int(self.n_input.currentText()) if self.n_input.currentText() else 0
        k = int(self.k_input.currentText()) if self.k_input.currentText() else 0
        j = int(self.j_input.currentText()) if self.j_input.currentText() else 0
        s = int(self.s_input.currentText()) if self.s_input.currentText() else 0

        if m > 0 and n > 0 and k > 0 and j > 0 and s > 0:
            n_numbers = random.sample(range(1, m + 1), n)
            self.worker = AlgorithmWorker(n_numbers, k, j, s)
            self.worker.finished.connect(self.update_result)
            self.worker.start()

    def update_result(self, n_numbers, result):
        self.result_text_edit.setText(
            f"Randomly selected n={len(n_numbers)} numbers: {n_numbers}\n\nThe approximate minimal set cover of k samples combinations found:\n")
        for idx, comb in enumerate(result, 1):
            self.result_text_edit.append(f"Combination {idx}: {comb}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
