from tapes import Tape
from numpy import random
import numpy
import lollocodeinterpreter as lci
import re

class GeneticAlgorithm:
    interpreter = None
    population = []
    level = 2
    input_tape = None
    output_tape = None
    initial_population_size = 100
    rank = {}
    maximum_fitness = 255
    accumulated_normalized_fitness = {}
    cross_over_probability = 0.7
    mutation_probability = 0.2
    best_solution = None
    best_fitness = None
    best_solution_ever = None
    best_fitness_ever = None
    max_population = 400
    max_program_size = 200
    min_program_size = 20
    instructions = [chr(1),chr(2), chr(3),chr(4),chr(5),chr(6)]
    instructions_compleate = [chr(1),chr(2), chr(3),chr(4),chr(5),chr(6),chr(7), chr(8), chr(9)]

    diseases = [lci.translate_from_bf_code("<>"), lci.translate_from_bf_code("><"), lci.translate_from_bf_code("+-"), lci.translate_from_bf_code("-+"), lci.translate_from_bf_code("[]")]

    def __init__(self, input_tape, output_tape, initial_population_size):
        self.input_tape = input_tape
        self.output_tape = output_tape
        self.initial_population_size = initial_population_size
        self.maximum_fitness = len(str(self.input_tape))*255


    def apply_genetic_algorithm(self):
        self.initialize_population(self.level)
        self.population = list(self.rank.keys())
        # during this procedure all fitness are not normalized
        zero_score = self.get_individual_score(Tape(None, None))
        old_fitness = zero_score
        fitness = min(self.rank.values())
        generation = 0
        print("Fitness variation:" + str((old_fitness-fitness)/old_fitness))

        while fitness > 1:
        # while (old_fitness-fitness)/old_fitness < 0.9995:
        #     print('\033c')
            self.print_rank()
            print("#################################################################")
            print("#################################################################")
            print("#################################################################")
            print("Starting generation n:" + str(generation))
            print("Best example: " + lci.translate_to_bf_code(str(self.best_solution)) + " fitness: " + str(self.best_fitness))
            print("Best solution: " + lci.translate_to_bf_code(str(self.best_solution_ever)) + " fitness: " + str(self.best_fitness_ever))
            print("Fitness variation:" + str((old_fitness-fitness)/old_fitness) + " fitness: " + str(fitness))
            print("[Selection] actual population size:" + str(len(self.population)) + " " + str(len(self.rank)))
            self.selection()
            self.repopulate()
            print("[Selection] Finished, actual population size:" + str(len(self.population)) + " " + str(len(self.rank)))
            print("[Cross-over] actual population size:" + str(len(self.population)) + " " + str(len(self.rank)))
            self.cross_over()
            print("[Cross-over] Finished, actual population size:" + str(len(self.population)) + " " + str(len(self.rank)))
            print("[Mutation] actual population size:" + str(len(self.population)) + " " + str(len(self.rank)))
            self.mutation()
            print("[Mutation]  Finished, actual population:" + str(len(self.population)) + " " + str(len(self.rank)))
            old_fitness = fitness
            self.population.append(self.best_solution_ever)
            self.calculate_rank()
            fitness = min(self.rank.values())
            generation += 1
        print("Found solution:")
        print("Best example: " + lci.translate_to_bf_code(str(self.best_solution)) + " fitness: " + str(self.best_fitness))
        print("Best solution: " + lci.translate_to_bf_code(str(self.best_solution_ever)) + " fitness: " + str(self.best_fitness_ever))

    def print_rank(self):
        print("Rank:")
        for i, tape in enumerate(self.rank.keys()):
            print('{:<200s} {:<20s}'.format(lci.translate_to_bf_code(str(tape)), str(self.rank[tape])))
            if i == 10:
                break

    def repopulate(self):
        if len(self.population) < 0.2 * self.initial_population_size:
            self.generate_random_population(self.initial_population_size)
            self.calculate_rank()
            self.normalize_fitness()

    def mutation(self):
        new_individuals = []
        remove_individuals = []
        for individual in self.population:
            if random.random() < self.mutation_probability:
                choices = range(4)
                choice = random.choice(choices)
                if choice == 0:
                    new_individual = self.mutate(individual)
                    remove_individuals.append(individual)
                elif choice == 1:
                    new_individual = self.extend(individual)
                    remove_individuals.append(individual)
                elif choice == 2:
                    new_individual = self.shorten(individual)
                    remove_individuals.append(individual)
                else:
                    new_individual = self.heal(individual)
                    remove_individuals.append(individual)
                # print(lci.translate_to_bfl_code(str(new_individual)+ " <-- " + str(individual)))
                new_individuals.append(new_individual)
        self.population += new_individuals
        for individual in remove_individuals:
            self.population.remove(individual)


    def heal(self, individual):
        return Tape(None, None, input_text=self.heal_code(str(individual)))

    def extend(self, individual):
        text = str(individual)
        if not text:
            return individual
        position = random.randint(0,len(text))
        new_char = self.generate_random_bfl_code(1)
        while new_char == text[position] == new_char:
            new_char = self.generate_random_bfl_code(1)

        text = text[:position] + new_char + text[position:]
        return Tape(None, None, input_text=text)

    def shorten(self, individual):
        text = str(individual)
        if not text:
            return individual
        position = random.randint(0,len(text))
        text = text[:position] + text[position+1:]
        return Tape(None, None, input_text=text)

    def mutate(self, individual):
        text = str(individual)
        if not text:
            return individual
        position = random.randint(0,len(text))
        new_char = self.generate_random_bfl_code(1)
        while new_char == text[position] == new_char:
            new_char = self.generate_random_bfl_code(1)

        text = text[:position] + new_char + text[position+1:]
        return Tape(None, None, input_text=text)

    # def cross_over(self):
    #     children_pool = []
    #     self.normalize_fitness()
    #     i = 0
    #     while i < len(self.population):
    #         father = self.population[i]
    #         if random.random() <= self.cross_over_probability:
    #             probabilities = numpy.array(list(self.rank.values()))
    #             mother = random.choice(list(self.rank.keys()), p=probabilities)
    #             while mother == father:
    #                 mother = random.choice(list(self.rank.keys()), p=probabilities)
    #             # print("[Cross-over] Searching tow childrens for:" + lci.translate_to_bfl_code(str(father)) + " and " + lci.translate_to_bfl_code(str(mother)))
    #             children1, children2 = self.generate_from_cross_over(father, mother)
    #             children_pool.append(children1)
    #             children_pool.append(children2)
    #         i += 1
    #     self.population = children_pool

    def cross_over(self):
        children_pool = []
        self.normalize_fitness()
        i = 0
        while i < len(self.population):
            father = self.population[i]
            for mother in self.population:
                if random.rand() <= self.rank[mother]*self.cross_over_probability:
                    children1, children2 = self.generate_from_cross_over(father, mother)
                    children_pool.append(children1)
                    children_pool.append(children2)
            i += 1
        self.population = children_pool

    def generate_from_cross_over(self, father, mother):
        father_text = str(father)
        mother_text = str(mother)
        children1_text, children2_text = self.get_splitted_text(father_text, mother_text)
        return Tape(None,None, input_text=children1_text),Tape(None,None, input_text=children2_text)

    def get_splitted_text(self, father_text, mother_text):
        line = numpy.random.randint(min(len(father_text), len(mother_text))+1)
        children1_text = mother_text[:line] + father_text[line:]
        children2_text = father_text[:line] + mother_text[line:]
        return children1_text, children2_text

    def selection(self):
        self.normalize_fitness()
        accumulator = 0
        self.accumulated_normalized_fitness = {}
        for elem in sorted(self.rank.items(), key=lambda kv: kv[1]):
            accumulator += elem[1]
            self.accumulated_normalized_fitness[elem[0]] = accumulator
        R = 1 - random.random() * random.random()  # random.random()
        if len(self.population) > self.max_population:
           R *= random.random()
        print("[selection] % of candidates surviving:" + str(R))
        new_candidates = {}
        counter = 0
        for elem in sorted(self.rank.items(), key=lambda kv: kv[1]):
            if self.accumulated_normalized_fitness[elem[0]] < R:
                counter += 1
                new_candidates[elem[0]] = self.rank[elem[0]]

        self.rank = new_candidates
        self.population = list(self.rank.keys())
        # print("Selection stage compleated, we are left with " + str(len(self.candidates)) + " individuals in the population")

    def normalize_fitness(self):
        accumulator = 0
        ordered_tapes = []
        for elem in sorted(self.rank.items(), key=lambda kv: kv[1]):
            accumulator += elem[1]
            ordered_tapes.append(elem[0])
        for tape in ordered_tapes:
            self.rank[tape] /= accumulator

    def calculate_rank(self):
        score = self.compute_fitness()
        self.rank = {}
        counter = 0
        for elem in sorted(score.items(), key=lambda kv: kv[1]):
            if counter == 0:
                self.best_solution = elem[0]
                self.best_fitness = elem[1]
                if not self.best_fitness_ever or self.best_fitness_ever > self.best_fitness:
                    self.best_solution_ever =  self.best_solution
                    self.best_fitness_ever =  self.best_fitness
            counter += 1
            self.rank[elem[0]] = elem[1]
            # print(('{:<50s} {:<20s}'.format(lci.translate_to_bfl_code(str(elem[0])), str(elem[1]))))

    def initialize_population(self, level):
        self.generate_random_population((self.initial_population_size-len(self.population)))
        self.calculate_rank()
        self.population = []
        if len(self.rank) < self.initial_population_size:
            self.initialize_population(level)

    def compute_fitness(self):
        score = {}
        for individual in self.population:
            score[individual] = self.get_individual_score(individual)
        return score

    def generate_random_population(self, size):
        # add optimizing
        # manage probabilities
        pop_string = []
        for i in range(len(self.population), size):
            new_tape = self.generate_random_bfl_tape()
            new_tape_string = lci.translate_to_bf_code(str(new_tape))
            while new_tape_string in pop_string:
                new_tape = self.generate_random_bfl_tape()
                new_tape_string = lci.translate_to_bf_code(str(new_tape))
            pop_string.append(new_tape_string)
            self.population.append(new_tape)
        return self.population

    def generate_random_bfl_tape(self):
        size = int(abs(random.normal(self.min_program_size,(self.max_program_size-self.min_program_size+1)/4)))
        bfl_tape_text = self.generate_random_bfl_code(size)
        return Tape(None, None, input_text=bfl_tape_text)

    # def generate_random_bfl_code(self, size):
    #     instructions = [chr(2), chr(3),chr(4),chr(5),chr(6)]
    #     bfl_text = ''.join(numpy.random.choice(instructions, size))
    #     return bfl_text

    def generate_random_bfl_code(self, size):
        if self.level == 1:
            bfl_text = ''.join(numpy.random.choice(self.instructions, size))
        else:
            bfl_text = ''.join(numpy.random.choice(self.instructions_compleate, size))
        bfl_text = self.heal_code(bfl_text)
        return bfl_text

    def heal_code(self, code):
        for disease in self.diseases:
            while disease in code:
                code = self.heal_dna_base(disease, code)
        code = self.heal_brakets(code)
        return code


    def heal_brakets(self, code):
        code = self.heal_brakets_count(code)
        brackets_pos = [i for i,val in enumerate(code) if val==chr(5)]
        brackets_negs = [i for i,val in enumerate(code) if val==chr(6)]
        all_brackets = brackets_negs + brackets_pos
        counter = 0
        self.instructions.remove(chr(5))
        self.instructions.remove(chr(6))
        for i, bracket_index in enumerate(sorted(all_brackets)):
            brackets_left = (len(all_brackets) - i)
            if bracket_index in brackets_negs:# ]
                if counter == 0: # Flip
                    code = self.replace_char(code, chr(5), bracket_index)
                    counter += 1
                else: # No Flip
                    flip_probability = 1 - (counter/brackets_left)
                    if random.random() <= flip_probability:# flip
                        code = self.replace_char(code, chr(5), bracket_index)
                        counter += 1
                    else:
                        counter -= 1
            else: # [
                flip_probability = counter/brackets_left
                if random.random() <= flip_probability:# flip
                    code = self.replace_char(code, chr(6), bracket_index)
                    counter -= 1
                else: # No Flip
                    counter += 1
        self.instructions.append(chr(5))
        self.instructions.append(chr(6))
        return code



    def heal_brakets_count(self, code):
        brackets = code.count(chr(5)) + code.count(chr(6))
        if divmod(brackets,2)[1] == 1:
            mutating = random.randint(brackets)
            if mutating + 1 > code.count(chr(5)):
                index = self.findnth(code, chr(6), mutating-code.count(chr(5)))
            else:
                index =  self.findnth(code, chr(5), mutating)
            self.instructions.remove(chr(5))
            self.instructions.remove(chr(6))
            new_gene = random.choice(self.instructions)
            self.instructions.append(chr(5))
            self.instructions.append(chr(6))
            code = code[:index] + new_gene + code[index+1:]
        return code
# def heal_brakets( code):
#     brackets = code.count("[") + code.count("]")
#     if divmod(brackets,2)[1] == 1:
#         mutating = random.randint(brackets)
#         if mutating+1 > code.count("["):
#             index = findnth(code, "]", mutating-code.count("["))
#         else:
#             index =  findnth(code, "[", mutating)
#         instructions.remove("[")
#         instructions.remove("]")
#         new_gene = random.choice(instructions)
#         instructions.append("[")
#         instructions.append("]")
#         code = code[:index] + new_gene + code[index+1:]
#     return code, index, mutating


    def findnth(self, code, search, n):
        parts = code.split(search, n + 1)
        if len(parts)<=n+1:
            return -1
        return len(code) - len(parts[-1]) - len(search)

    def heal_dna_base(self, base, code):
        replaced = random.randint(len(base))
        self.instructions.remove(base[replaced])
        new_base = self.replace_char(base, random.choice(self.instructions),replaced)
        self.instructions.append(base[replaced])
        return code.replace(base, new_base, 1)

    def replace_char(self, string, replacement, index):
        return string[:index] + replacement + string[index+1:]


    def get_individual_score(self, individual):
        try:
            self.interpreter = lci.Interpreter(Tape(None, None, other_tape=individual), Tape(None, None, other_tape=self.input_tape), self.output_tape, self.level, self.level-1)
            iterations = self.interpreter.run()
            return len(str(individual)) * 0.01 + iterations * 0.01 + self.interpreter.error_tape.get_square_of_errors()
        except RecursionError:
            return self.maximum_fitness
        except KeyError:
            return self.maximum_fitness/2
        except IndexError:
            return self.maximum_fitness/2

    def print_code(self, code):
        print(lci.translate_to_bf_code(str(code)))
