import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

class SFLAOptimizer:
    def __init__(self, X, y, n_frogs=20, n_memeplexes=5, max_iter=30):
        self.X = X
        self.y = y
        self.n_frogs = n_frogs
        self.n_memeplexes = n_memeplexes
        self.max_iter = max_iter
        self.best_params = None
        self.best_score = -np.inf

    def fitness(self, params):
        n_estimators = int(params[0])
        learning_rate = params[1]
        if n_estimators < 10 or learning_rate < 0.01:
            return 0.0
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        scores = cross_val_score(model, self.X, self.y, cv=5, scoring='accuracy')
        return scores.mean()

    def optimize(self):
        # Initialize frogs: [n_estimators, learning_rate]
        frogs = np.random.uniform(low=[10, 0.01], high=[200, 2.0], size=(self.n_frogs, 2))
        
        for iteration in range(self.max_iter):
            # Evaluate fitness
            fitness = np.array([self.fitness(frog) for frog in frogs])
            sorted_idx = np.argsort(fitness)[::-1]
            frogs = frogs[sorted_idx]
            fitness = fitness[sorted_idx]

            # Update global best
            if fitness[0] > self.best_score:
                self.best_score = fitness[0]
                self.best_params = frogs[0].copy()

            # Partition into memeplexes
            memeplex_size = self.n_frogs // self.n_memeplexes
            for m in range(self.n_memeplexes):
                start = m * memeplex_size
                memeplex = frogs[start:start + memeplex_size]
                mem_fitness = fitness[start:start + memeplex_size]

                # Local search: worst frog leaps toward best in memeplex
                best_idx = np.argmax(mem_fitness)
                worst_idx = np.argmin(mem_fitness)
                Xb = memeplex[best_idx]
                Xw = memeplex[worst_idx]

                # Leap update (paper's local search rule)
                step = np.random.uniform(-1, 1, 2) * (Xb - Xw)
                new_pos = Xw + step
                new_pos = np.clip(new_pos, [10, 0.01], [200, 2.0])

                if self.fitness(new_pos) > mem_fitness[worst_idx]:
                    memeplex[worst_idx] = new_pos
                else:
                    # Random leap
                    memeplex[worst_idx] = np.random.uniform(low=[10, 0.01], high=[200, 2.0], size=2)

                frogs[start:start + memeplex_size] = memeplex

            print(f"Iter {iteration+1}/{self.max_iter} - Best fitness: {self.best_score:.4f}")

        n_est, lr = self.best_params
        return {'n_estimators': int(n_est), 'learning_rate': float(lr)}
