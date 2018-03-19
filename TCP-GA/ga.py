import math
import random
import numpy as np

class GeneticRules(object):
    """docstring for GeneticRules."""
    def __init__(self, mutation_rate = 0.015):
        super(GeneticRules, self).__init__()
        self.mutation_rate = mutation_rate

    def init_params(self, X, Y, population_len):
        points = zip(X,Y)
        self.points = points
        population = []
        for i in range(population_len):
            population.append(points)
            random.shuffle(population[i])
        self.population = population

    def get_distance(self, chromosome):
        dist = 0
        sz = len(chromosome)
        for i in range(sz):
            p1 = chromosome[i]
            p2 = chromosome[(i + 1) % sz]
            dist += self.distance(p1, p2)
        return dist

    def distance(self, p1, p2):
        x1,y1 = p1
        x2,y2 = p2
        a = x1 - x2
        b = y1 - y2
        return math.sqrt(a * a + b * b)

    def get_fittest(self):
        population = self.population
        fittest_index = 0
        fittest_value = 0
        for it, chromosome in enumerate(population):
            dist = self.get_distance(chromosome)
            if fittest_value < dist:
                fittest_value = dist
                fittest_index = it
        return self.population[fittest_index], fittest_value

    def evolve_population(self):
        """
        Crossover + Selection
        Mutation
        """
        pass

    def selection(self):
        pass

    def crossover(self, parent1, parent2):
        sz = len(parent1)
        startPos = int(np.random.rand() * sz)
        endPos = int(np.random.rand() * sz)
        child = [0] * sz
        for i in range(sz):
            if startPos < endPos and i in range(startPos, endPos + 1):
                child[i] = parent1[i]
            elif startPos > endPos:
                if (i < startPos and i > endPos) == False:
                    child[i] = parent1[i]

        for i in range(sz):
            inChild = False
            for j in range(sz):
                if parent2[i] == child[j]:
                    inChild = True
                    break
            if inChild == False:
                for j in range(sz):
                    if child[j] == 0:
                        child[j] = parent2[i]

        return child

    def mutate(self, chromosome):
        sz = len(chromosome)
        for i in range(sz):
            if np.random.rand() < self.mutation_rate:
                j = int(sz * np.random.rand())
                p = chromosome[i]
                chromosome[i] = chromosome[j]
                chromosome[j] = p
        return chromosome

if __name__ == '__main__':
    sz = 10
    X = [int(np.random.randn() * 1000) / 100.0 for i in range(sz)]
    Y = [int(np.random.randn() * 1000) / 100.0 for i in range(sz)]

    gr = GeneticRules(0.5)
    gr.init_params(X, Y, 10)

    num_generations = 100
    for i in range(num_generations):
        gr.evolve_population()

    fittest, fittest_value = gr.get_fittest()
    print("Distance: ", fittest_value)
    print("Solution: ", fittest)
    print("          ", zip(X,Y))
