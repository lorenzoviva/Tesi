import numpy.random as random
import numpy as np

class Structure:
    layers = []
    levels = 0
    level = -1

    def __init__(self, base_layer, other_layers, levels):
        self.layers = []
        self.layers.append(Layer([""],[">","<"],["+","-"]))
        self.layers.append(base_layer)
        if isinstance(other_layers, (list,)):
            self.layers.append(other_layers)
        else:
            for level in range(levels-1):
                self.layers.append(other_layers)
        self.levels = levels
        self.level = 0

    def generate_tape(self):
        # print(self.layers)
        return self.layers[self.level].get_tape()

    def adjust_probabilities(self, codes, code_fitness):
        # print("Adjusting: " + str(self.level) + " layers probability")
        for i, layer in enumerate(self.layers[:self.level+1]):
            layer.adjust_probabilities(codes[i], code_fitness[i])

    def increase_level(self):
        self.level += 1

    def decrease_level(self):
        self.level -= 1

    def generate_all_tapes(self):
        tapes = []
        partecipating_layers = self.layers[:self.level+1]
        partecipating_layers.reverse()
        for layer in partecipating_layers:
            tapes.append(layer.get_tape())
        tapes.reverse()
        return tapes
    # def generate_all_tapes(self):
    #     tapes = []
    #     top_level = int(self.level)
    #     for _ in range(self.level):
    #         tapes.append(self.generate_tape())
    #         self.decrease_level()
    #     self.level = top_level
    #     return tape

    def __str__(self):
        string = ""
        for i,layer in enumerate(self.layers[:self.level+1]):
            string += ("LAYER N " + str(i) + ":\n " + str(layer)) + "\n"
        return string
class Layer:
    first_tapes = []
    first_tapes_probabilities = []
    second_tapes = []
    second_tapes_probabilities = []
    third_tapes = []
    third_tapes_probabilities = []
    fourth_tapes = []
    fourth_tapes_probabilities = []
    last_generated_tape = None

    def __init__(self, first_tapes, second_tapes, third_tapes):
        self.first_tapes = first_tapes
        self.second_tapes = second_tapes
        self.third_tapes = third_tapes
        self.fourth_tapes = [""]
        self.first_tapes_probabilities = []
        self.second_tapes_probabilities = []
        self.third_tapes_probabilities = []
        self.fourth_tapes_probabilities = [1.0]
        for _ in first_tapes:
            self.first_tapes_probabilities += [1 / float(len(first_tapes))]
        for _ in second_tapes:
            self.second_tapes_probabilities += [1 / float(len(second_tapes))]
        for _ in third_tapes:
            self.third_tapes_probabilities += [1 / float(len(third_tapes))]

    def get_tape(self):
        self.normalize_probabilities()
        first_part = random.choice(self.first_tapes, p=self.first_tapes_probabilities)
        second_part = random.choice(self.second_tapes, p=self.second_tapes_probabilities)
        third_part = random.choice(self.third_tapes, p=self.third_tapes_probabilities)
        fourth_part = random.choice(self.fourth_tapes, p=self.fourth_tapes_probabilities)
        self.last_generated_tape = [first_part, second_part, third_part, fourth_part]
        return ''.join(self.last_generated_tape)

    def normalize_probabilities(self):
        self.first_tapes_probabilities = np.array(self.first_tapes_probabilities)
        self.first_tapes_probabilities /= self.first_tapes_probabilities.sum()  # normalize
        self.second_tapes_probabilities = np.array(self.second_tapes_probabilities)
        self.second_tapes_probabilities /= self.second_tapes_probabilities.sum()  # normalize
        self.third_tapes_probabilities = np.array(self.third_tapes_probabilities)
        self.third_tapes_probabilities /= self.third_tapes_probabilities.sum()  # normalize
        self.fourth_tapes_probabilities = np.array(self.fourth_tapes_probabilities)
        self.fourth_tapes_probabilities /= self.fourth_tapes_probabilities.sum()  # normalize

    def adjust_probabilities(self, new_code, code_fitness):
        code_fitness = -code_fitness
        probability_multiplier = 1
        if code_fitness > 0:
            probability_multiplier += 1/len(self.first_tapes)
        if code_fitness < 0:
            probability_multiplier -= 1/len(self.first_tapes)
        if len(self.first_tapes) < 2:
            probability_multiplier = 1
        for i,tape in enumerate(self.first_tapes):
            if tape == self.last_generated_tape[0]:
                self.first_tapes_probabilities[i] *= probability_multiplier

        probability_multiplier = 1
        if code_fitness > 0:
            probability_multiplier += 1/len(self.second_tapes)
        if code_fitness < 0:
            probability_multiplier -= 1/len(self.second_tapes)
        if len(self.second_tapes) < 2:
            probability_multiplier = 1
        for i,tape in enumerate(self.second_tapes):
            if tape == self.last_generated_tape[1]:
                self.second_tapes_probabilities[i] *= probability_multiplier

        probability_multiplier = 1
        if code_fitness > 0:
            probability_multiplier += 1/len(self.third_tapes)
        if code_fitness < 0:
            probability_multiplier -= 1/len(self.third_tapes)
        if len(self.third_tapes) < 2:
            probability_multiplier = 1
        for i,tape in enumerate(self.third_tapes):
            if tape == self.last_generated_tape[2]:
                self.third_tapes_probabilities[i] *= probability_multiplier

        probability_multiplier = 1
        if code_fitness > 0:
            probability_multiplier += 1/len(self.fourth_tapes)
        if code_fitness < 0:
            probability_multiplier -= 1/len(self.fourth_tapes)
        if len(self.fourth_tapes) < 2:
            probability_multiplier = 1
        for i,tape in enumerate(self.fourth_tapes):
            if tape == self.last_generated_tape[3]:
                self.fourth_tapes_probabilities[i] *= probability_multiplier
        if new_code not in self.fourth_tapes and code_fitness > 0:
            self.append_to_fourth_tape(new_code)
        self.normalize_probabilities()
        self.clean_fourth_tape()
        self.normalize_probabilities()

    def clean_fourth_tape(self):
        mean = np.mean(self.fourth_tapes_probabilities)
        delete_list = []
        for i, probability in enumerate(self.fourth_tapes_probabilities):
            if probability <= 0.3 and self.fourth_tapes[i] != "":
                delete_list.append(i)
                # print("Removing : " + self.fourth_tapes[i] +" " + str(probability))
        delete_list.reverse()
        while delete_list:
            popped = delete_list.pop()
            self.fourth_tapes = self.fourth_tapes[:popped] + self.fourth_tapes[popped+1:]
            self.fourth_tapes_probabilities = np.append(self.fourth_tapes_probabilities[:popped], self.fourth_tapes_probabilities[popped+1:])

    def append_to_fourth_tape(self, new_code):
        new_probability = 1.0/float(len(self.fourth_tapes)+1)
        # print("appending : " + new_code + "(" + str(new_probability*100) + "%) to " + str(self))
        multiplier = 1 - new_probability
        self.fourth_tapes_probabilities *= multiplier
        self.fourth_tapes_probabilities = np.append(self.fourth_tapes_probabilities, [new_probability])
        self.fourth_tapes.append(new_code)

    def __str__(self):
        first_tapes_string = ''
        first_tapes_probabilities_string = ''
        for first_tape in self.first_tapes:
            first_tapes_string +='\t{:<10s}'.format(str(first_tape))
        for first_tapes_probability in self.first_tapes_probabilities:
            first_tapes_probabilities_string += str('\t{:<10s}'.format('{:.2f}'.format(first_tapes_probability*100)))

        second_tapes_string = ''
        second_tapes_probabilities_string = ''
        for second_tape in self.second_tapes:
            second_tapes_string +='\t{:<10s}'.format(str(second_tape))
        for second_tapes_probability in self.second_tapes_probabilities:
            second_tapes_probabilities_string += str('\t{:<10s}'.format('{:.2f}'.format(second_tapes_probability*100)))

        third_tapes_string = ''
        third_tapes_probabilities_string = ''
        for third_tape in self.third_tapes:
            third_tapes_string +='\t{:<10s}'.format(str(third_tape))
        for third_tapes_probability in self.third_tapes_probabilities:
            third_tapes_probabilities_string += str('\t{:<10s}'.format('{:.2f}'.format(third_tapes_probability*100)))

        fourth_tapes_string = ''
        fourth_tapes_probabilities_string = ''
        for fourth_tape in self.fourth_tapes:
            fourth_tapes_string +='\t{:<10s}'.format(str(fourth_tape))
        for fourth_tapes_probability in self.fourth_tapes_probabilities:
            fourth_tapes_probabilities_string += str('\t{:<10s}'.format('{:.2f}'.format(fourth_tapes_probability*100)))

        return first_tapes_string + "\n"  + first_tapes_probabilities_string + "\n" + second_tapes_string + "\n" + second_tapes_probabilities_string + "\n" + third_tapes_string + "\n" + third_tapes_probabilities_string+ "\n" + fourth_tapes_string + "\n" + fourth_tapes_probabilities_string
