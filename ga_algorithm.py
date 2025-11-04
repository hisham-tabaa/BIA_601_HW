# ga_algorithm.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class GeneticFeatureSelection:
    def __init__(self, X, y, pop_size=20, n_gen=20, crossover_rate=0.8, mutation_rate=0.05):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.population = np.random.randint(0, 2, (pop_size, self.n_features))
        self.best_chrom = None
        self.best_score = 0

    def fitness(self, chrom):
        selected = np.where(chrom == 1)[0]
        if len(selected) == 0:
            return 0
        X_sel = self.X[:, selected]
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_sel, self.y)
        preds = model.predict(X_sel)
        return accuracy_score(self.y, preds)

    def selection(self, fitnesses):
        probs = fitnesses / (np.sum(fitnesses) + 1e-8)
        idx = np.random.choice(len(probs), size=2, p=probs)
        return self.population[idx[0]], self.population[idx[1]]

    def crossover(self, p1, p2):
        if np.random.rand() < self.cr:
            point = np.random.randint(1, self.n_features - 1)
            c1 = np.concatenate([p1[:point], p2[point:]])
            c2 = np.concatenate([p2[:point], p1[point:]])
            return c1, c2
        return p1.copy(), p2.copy()

    def mutate(self, chrom):
        for i in range(len(chrom)):
            if np.random.rand() < self.mr:
                chrom[i] = 1 - chrom[i]
        return chrom

    def evolve(self):
        for _ in range(self.n_gen):
            fitnesses = np.array([self.fitness(ind) for ind in self.population])
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_score:
                self.best_score = fitnesses[best_idx]
                self.best_chrom = self.population[best_idx].copy()

            new_pop = []
            while len(new_pop) < self.pop_size:
                p1, p2 = self.selection(fitnesses)
                c1, c2 = self.crossover(p1, p2)
                c1, c2 = self.mutate(c1), self.mutate(c2)
                new_pop.extend([c1, c2])
            self.population = np.array(new_pop[:self.pop_size])
        return self.best_chrom, self.best_score
