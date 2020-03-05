from tapes import Tape
from numpy import random
import numpy
import lollocodeinterpreter as lci
import re

class GeneticAlgorithm:
    interpreter = None
    population = []
    level = 1
    input_tape = None
    output_tape = None
    initial_population_size = 30
    rank = {}
    maximum_fitness = 255
    accumulated_normalized_fitness = {}
    cross_over_probability = 0.8
    mutation_probability = 0.2
    best_solution = None
    best_fitness = None
    max_population = 200

    def __init__(self, input_tape, output_tape, initial_population_size):
        self.input_tape = input_tape
        self.output_tape = output_tape
        self.initial_population_size = initial_population_size
        self.maximum_fitness = len(str(self.input_tape))*255


    def apply_genetic_algorithm(self):
        self.initialize_population(1)
        self.population = self.rank.keys()
        # during this procedure all fitness are not normalized
        zero_score = self.get_individual_score(Tape(None, None))
        old_fitness = zero_score
        fitness = min(self.rank.values())
        generation = 0
        print("Fitness variation:" + str((old_fitness-fitness)/old_fitness))

        while fitness > 1:
        # while (old_fitness-fitness)/old_fitness < 0.9995:
            print("Starting generation n:" + str(generation))
            print("Best example" + lci.translate_to_bf_code(str(self.best_solution)) + " fitness: " + str(self.best_fitness))
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
            # print("[Mutation]  Finished, actual population:" + str(self.population) + " " + str(len(self.rank)))
            old_fitness = fitness
            self.calculate_rank()
            fitness = min(self.rank.values())
            generation += 1

    def repopulate(self):
        if len(self.population) < 0.2 * self.initial_population_size:
            self.generate_random_population(10)
            self.calculate_rank()
            self.normalize_fitness()

    def mutation(self):
        new_individuals = []
        remove_individuals = []
        for individual in self.population:
            if random.random() < self.mutation_probability:
                new_individual = self.mutate(individual)
                remove_individuals.append(individual)
                # print(lci.translate_to_bfl_code(str(new_individual)+ " <-- " + str(individual)))
                new_individuals.append(new_individual)
        self.population += new_individuals
        for individual in remove_individuals:
            self.population.remove(individual)

    def mutate(self, individual):
        text = str(individual)
        if text:
            index = random.choice(range(len(text)))
            while text[index] == chr(5) or text[index] == chr(6):
                index = random.choice(range(len(text)))
            text_array = list(text)
            text_array[index]= "Ā"
            text = ''.join(text_array)
        else:
            text = "Ā"
        while "Ā" in text:
            text = self.generate_random_bfl_tape_text(text, 0.2)
        return Tape(None, None, input_text=text)


    # def cross_over(self):
    #     fathers = []
    #     mothers = []
    #     counter = 0
    #     for i in range(math.floor(len(self.rank) / 2)):
    #         if random.random() <= self.cross_over_probability:
    #             fathers.append(list(self.rank.keys())[i])
    #     self.normalize_fitness()
    #     probabilities = numpy.array(list(self.rank.values()))
    #     probabilities = probabilities - (probabilities - probabilities.mean())
    #     for father in fathers:
    #         mother = random.choice(list(self.rank.keys()), p=probabilities)
    #         while mother in fathers or mother in mothers:
    #             mother = random.choice(list(self.rank.keys()), p=probabilities)
    #         mothers.append(mother)
    #         # print("[Cross-over] [mating] Searching a children for:" + lci.translate_to_bfl_code(str(father)) + " and " + lci.translate_to_bfl_code(str(mother)))
    #         children = self.generate_from_cross_over(father, mother)
    #         self.population.append(children)

    def cross_over(self):
        childrens = []
        self.normalize_fitness()
        for father in self.population:
            if random.random() <= self.cross_over_probability:
                probabilities = numpy.array(list(self.rank.values()))
                # probabilities = probabilities - (probabilities - probabilities.mean())
                mother = random.choice(list(self.rank.keys()), p=probabilities)
                while mother == father:
                    mother = random.choice(list(self.rank.keys()), p=probabilities)
                # print("[Cross-over] Searching a children for:" + lci.translate_to_bfl_code(str(father)) + " and " + lci.translate_to_bfl_code(str(mother)))
                children = self.generate_from_cross_over(father, mother)
                childrens.append(children)
        self.population += childrens

    def generate_from_cross_over(self, father, mother):
        father_text = str(father)
        mother_text = str(mother)
        children_text = self.get_children_text(father_text, mother_text)
        return Tape(None,None, input_text=children_text)

    # def get_children_text(self, father_text, mother_text):
    #     line = numpy.random.randint(min(len(father_text), len(mother_text))+1)
    #     if numpy.random.random() > 0.5:
    #         children_text = father_text[:line]+mother_text[line:]
    #     else:
    #         children_text = mother_text[:line]+father_text[line:]
    #     while children_text.count(chr(6)) != children_text.count(chr(5)) or (re.findall(chr(9), children_text) and re.findall("."+chr(9), children_text)):
    #         # print("It's not working: " + bf_child_text)
    #         line = numpy.random.randint(min(len(father_text),len(mother_text))+1)
    #         if numpy.random.random() > 0.5:
    #             children_text = father_text[:line]+mother_text[line:]
    #         else:
    #             children_text = mother_text[:line]+father_text[line:]
    #     return children_text

    def get_children_text(self, father_text, mother_text):
        father_structure = self.build_layers(father_text)
        mother_structure = self.build_layers(mother_text)
        if random.random() > 0.5:
            children_structure = father_structure
            other_structure = mother_structure
        else:
            children_structure = mother_structure
            other_structure = father_structure
        for i, children_structure_level in enumerate(children_structure):
            if len(other_structure) > i:
                for l, children_structure_base in enumerate(children_structure_level):
                    print(str(other_structure[i]))
                    if isinstance(other_structure[i], (list,)):
                        other_base = random.choice(other_structure[i])
                    else:
                        other_base = other_structure[i]
                    if isinstance(children_structure[i], (list,)):
                        children_structure[i][l] = self.get_splitted_text(children_structure[i][l], other_base)
                    else:
                        children_structure[i] = self.get_splitted_text(children_structure[i], other_base)
        children_text = self.layers_to_text(children_structure)
        return children_text

    def get_splitted_text(self, father_text, mother_text):
        line = numpy.random.randint(min(len(father_text), len(mother_text))+1)
        children_text = mother_text[:line]+father_text[line:]
        return children_text

    def layers_to_text(self, children_structure):
        children_structure.reverse()
        old_children_structure_layer = []
        new_children_structure = []
        for children_structure_layer in children_structure:
            while len(old_children_structure_layer) > 0:
                old_children_structure_base = old_children_structure_layer.pop()
                placed = False
                for i, children_structure_base in enumerate(children_structure_layer):
                    if random.random() < 1 / len(children_structure_layer):
                        print("text : " + str(chr(5) + old_children_structure_base + chr(6)))
                        children_structure_layer[i] += str(chr(5) + old_children_structure_base + chr(6))
                        placed = True
                if not placed:
                    ind = int(numpy.floor(random.random()*(len(children_structure_layer)+2)))
                    print("ind: " + str(ind))
                    print("text : " + str(children_structure_layer[:ind] + [chr(5) + old_children_structure_base + chr(6)] + children_structure_layer[ind:]))
                    children_structure_layer = str(children_structure_layer[:ind] + [chr(5) + old_children_structure_base + chr(6)] + children_structure_layer[ind:])
            new_children_structure.append(children_structure_layer)
            old_children_structure_layer = children_structure_layer
        new_children_structure.reverse()
        children_text = ''.join(new_children_structure[0])
        return children_text

    def build_layers(self, parent_text):
        layers = [[]]
        accumulator = ""
        pointer = 0
        for key in parent_text:
            if ord(key) == 5:
                if accumulator:
                    layers[pointer].append(accumulator)
                pointer += 1
                if len(layers) < pointer+1:
                    layers.append([])
                accumulator = ""
            elif ord(key) == 6:
                if accumulator:
                    layers[pointer].append(accumulator)
                pointer -= 1
                accumulator = ""
            else:
                accumulator += key
        if accumulator:
            layers[pointer] = accumulator
        print("LAYERS:")
        print(str(layers))
        return layers

    def selection(self):
        self.normalize_fitness()
        accumulator = 0
        self.accumulated_normalized_fitness = {}
        for elem in sorted(self.rank.items(), key=lambda kv: kv[1]):
            accumulator += elem[1]
            self.accumulated_normalized_fitness[elem[0]] = accumulator
        R = random.random()  # random.random()
        if len(self.population) > self.max_population:
            R = 0.1
        print("[selection] % of candidates surviving:" + str(R))
        new_candidates = {}
        for elem in sorted(self.rank.items(), key=lambda kv: kv[1]):
            if self.accumulated_normalized_fitness[elem[0]] < R:
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
        zero_score = Tape(None, None)
        zero_score.code_pointer = 1
        self.population.append(zero_score)
        score = self.compute_fitness()
        self.rank = {}
        # self.population.append(zero_score)
        # score[zero_score] = self.get_individual_score(zero_score)
        zero_score_score = score[zero_score]
        counter = 0
        for elem in sorted(score.items(), key=lambda kv: kv[1]):
            if counter == 0:
                self.best_solution = elem[0]
                self.best_fitness = elem[1]
            counter += 1
            # if elem[1] < zero_score_score:
            self.rank[elem[0]] = elem[1]
            # print(('{:<50s} {:<20s}'.format(lci.translate_to_bfl_code(str(elem[0])), str(elem[1]))))

    def initialize_population(self, level):
        self.level = level
        self.population = []
        self.restart_with_population()
        self.generate_random_population((self.initial_population_size-len(self.population))*6)
        self.calculate_rank()
        self.population = []
        if len(self.rank) < self.initial_population_size:
            self.initialize_population(level)
        else:
            print("candidates:")
            for elem in sorted(self.rank.items(), key=lambda kv: kv[1]):
               print(('{:<50s} {:<20s}'.format(lci.translate_to_bf_code(str(elem[0])), str(elem[1]))))
            print("paste in restart_with_population to start with this set of instruction as initial population:")
            for elem in sorted(self.rank.items(), key=lambda kv: kv[1]):
                print("self.population.append(Tape(None,None,lci.translate_from_bfl_code(\"" + lci.translate_to_bf_code(str(elem[0])) + "\")))")

    def restart_with_population(self):
        self.population.append(Tape(None, None, lci.translate_from_bf_code(".+>><+,++")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(".++>+.+")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("+>+-[++<+[>]+[+]-*]>+")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">>--+-+-")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("-[<].[>]>>+>>->>")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">,<>+")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">+[>,].")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">+")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("<,>,>-++")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">+")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">+[>].>[<]<")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">+[>].[>]-[+].")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("+,-+->+<>")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(",>.+.")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("->+")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">+[>].[+[,],>]+")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">[+[>],+[,+].]--+")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">+[>]++*")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("-,>.,.+[>]+")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(".>,[+>>]<+*[.]<>+")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("+>,>-,[>].[,]+[-]--")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(".,+>>>-.--[>]>")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">[+[.[>].>],]>*[,],[-]>,")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(">+[>]>>[+]->*")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code(",[>],[>].<-")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("-[>]<-<<[>].")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("<[.,],>>-++>+")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("+[->>]-+.[-]>*")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("+[+>-+]<-.")))
        self.population.append(Tape(None, None, lci.translate_from_bf_code("+[,->]-<-<")))
        pass
        # PASTE HERE SAVED POPULATION

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
        bfl_tape_text = ""
        if not bfl_tape_text:
            bfl_tape_text = "Ā"  # ŷ
        while "Ā" in bfl_tape_text:
            bfl_tape_text = self.generate_random_bfl_tape_text(bfl_tape_text,0.999)
        return Tape(None, None, input_text=bfl_tape_text)

    def generate_random_bfl_tape_text(self, bfl_tape_text,extend_probability):
        choices = [lci.translate_from_bf_code("Ā[Ā]Ā"), "ĀĀ", lci.translate_from_bf_code("+"), lci.translate_from_bf_code("-"), lci.translate_from_bf_code(">"), lci.translate_from_bf_code("<")]
        new_text = random.choice(choices)
        return bfl_tape_text.replace("Ā", new_text, 1)


    # def generate_random_bfl_tape_text(self, bfl_tape_text,extend_probability):
    #     choices = [lci.translate_from_bfl_code("Ā[Ā]Ā"), "ĀĀ", lci.translate_from_bfl_code("+"), lci.translate_from_bfl_code("-"), lci.translate_from_bfl_code(">"), lci.translate_from_bfl_code("<"), lci.translate_from_bfl_code("*"), lci.translate_from_bfl_code("."), lci.translate_from_bfl_code(",")]
    #     extend_probability = extend_probability * pow(0.9, len(bfl_tape_text))
    #     probabilities = [extend_probability*0.3, extend_probability*0.7, 0.2*(1-extend_probability), 0.2*(1-extend_probability), 0.15*(1-extend_probability), 0.15*(1-extend_probability), 0.16*(1-extend_probability), 0.07*(1-extend_probability), 0.07*(1-extend_probability)]
    #     if len(bfl_tape_text) == 1 or bfl_tape_text.find("Ā") == 0 or bfl_tape_text.find("Ā")-1 == bfl_tape_text.find(lci.translate_from_bfl_code("[Ā")) or bfl_tape_text.find("Ā")-1 == bfl_tape_text.find(lci.translate_from_bfl_code("]Ā")):
    #         probabilities = [extend_probability*0.3, extend_probability*0.7, 0.2*(1-extend_probability), 0.2*(1-extend_probability), 0.15*(1-extend_probability), 0.15*(1-extend_probability), 0, 0.15*(1-extend_probability), 0.15*(1-extend_probability)]
    #     new_text = random.choice(choices, p=probabilities)
    #     return bfl_tape_text.replace("Ā", new_text, 1)

    def get_distribution_info(self):
        sum = numpy.array([])
        for i in range(1000):
            newcode = self.generate_random_bfl_tape()
            sum = numpy.append(sum, len(str(newcode)))
        counter = 0
        for s in sum:
            if s == 1:
                counter += 1
        return counter/len(sum), sum.mean(), sum.std()

    def get_individual_score(self, individual):
        self.interpreter = lci.Interpreter(Tape(None, None, other_tape=individual), Tape(None, None, other_tape=self.input_tape), self.output_tape, self.level, self.level-1)
        # self.interpreter.tapes[self.level] = individual
        try:
            self.interpreter.run()
        except RecursionError:
            return self.maximum_fitness
        # except KeyError:
        #     print("Level: " + str(self.interpreter.level) + " code:" + str(self.interpreter.code) + " code_pointer:" + str(self.interpreter.code.code_pointer) + " starting/ending:" + str(self.interpreter.code.get_ending()) + "-" + str(self.interpreter.code.get_ending()))
        return self.interpreter.error_tape.get_square_of_errors()
