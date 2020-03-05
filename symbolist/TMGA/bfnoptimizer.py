from bfnstructure import Layer,Structure
from tapes import Tape
from bfninterpreter import Interpreter
import numpy as np
import os
import io
import time
def get_absolute_error(error):
    while error > 127:
        error -= 256
    while error < -128:
        error += 256
    # junk, error = divmod(error, 128)
    return error


def get_error_tape(a, b):
    lista = a.get_list()
    listb = b.get_list()
    listc = []
    listal = len(lista)
    listbl = len(listb)
    for i in range(0, max(listal, listbl)):
        if listal > i and listbl > i:
            listc += [lista[i] - listb[i]]
        elif listal <= i < listbl:
            listc += [-listb[i]]
        elif listal > i >= listbl:
            listc += [lista[i]]
    for index, elem in enumerate(listc):
        listc[index] = get_absolute_error(elem)
    c = Tape(None, None, input_array=listc)
    return c


class Optimizer:

    structure = None
    input_tape = None
    target_tape = None
    thereshold = 1
    counter = 0
    # inplementation dependent
    number_of_times_error_not_change = 0
    best_total_error = 0
    code_fitness = 0
    lenght_cohefficient = 0.05
    n_of_levels = 50
    sum_of_error = 1
    def __init__(self):
        base_layer = Layer(["", "[']'"], ["", "<'", ">'"], ["", "+'<", "+^<", "-'<", "-^<"])
        other_layers = Layer(["", "[']'"], ["", "<'", ">'"], ["", "+'*'", "+'/'", "-'*'", "-*/'", "['*'", "['/'", "]'*'", "]'/'", "<'*'", "<'/'", ">'*'", ">'/'"])
        self.structure = Structure(base_layer,other_layers,self.n_of_levels)
        self.input_tape = Tape(None, None, "22")
        self.target_tape = Tape(None, None, "4")
        self.structure.level = 2

    def run(self):
        self.number_of_times_error_not_change = 0
        self.structure.level = 0
        level_repetititons = list(np.zeros(self.n_of_levels))
        while self.sum_of_error:
            while level_repetititons[self.structure.level] == self.thereshold and self.structure.level < self.n_of_levels - 1:
                level_repetititons[self.structure.level] = 0
                self.structure.level += 1
            if self.structure.level >= self.n_of_levels - 1:
                os.system("beep")
            level_repetititons[self.structure.level] += 1
            self.counter += 1
            print(str(level_repetititons) + " " + str(self.counter))
            self.step()
            self.structure.level = 0

    def step(self):
        # os.system("clear")
        # print("#######################################################\n\n\n\n\n")
        print("#######################################################")

        no_changes_counter = 0
        codes = self.structure.generate_all_tapes()
        print("generated codes: " + str(codes))
        before_codes = codes.copy()
        code = ""
        tape = Tape(None, None, other_tape=self.input_tape)
        before = Tape(None, None, input_array=list(np.abs(np.array(get_error_tape(tape, self.target_tape).get_list()))))
        self.code_fitness = list(np.zeros(len(codes)))
        for n, code in enumerate(codes):
            self.code_fitness[n] = len(code) * self.lenght_cohefficient
        for l, _ in enumerate(codes):
            tape = Tape(None, None, other_tape=self.input_tape)
            codes.reverse()
            i = 0
            while i <= l:
                real_pointer = -l + i - 1
                code = codes[real_pointer]
                if i != l:
                    input_tape = Tape(None, None, codes[real_pointer+1])
                else:
                    input_tape = tape
                # print("Applyng code: " + code + " on input: " + str(input_tape) + " i, l: " + str(i) + ", " + str(l))
                try:
                    interpreter = Interpreter(code, input_tape, self.target_tape, debug=False)
                    interpreter.run()
                except RecursionError:# i 1    l 2
                    self.code_fitness[l-i+1] += 1000
                if i != l:
                    codes[real_pointer+1] = str(input_tape)
                i += 1
            codes.reverse()
            # evaluation

            after_list = list(np.abs(np.array(get_error_tape(tape, self.target_tape).get_list())))
            after = Tape(None, None, input_array=after_list)
            difference_of_error = get_error_tape(before, after)

            sum_of_difference_of_error = sum(difference_of_error.get_list())
            self.code_fitness[l] -= sum_of_difference_of_error
            self.sum_of_error = sum(after_list)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("Code analysis for code: " + str(code) + " \n\tabsolute error before: " + str(before.get_list()) + " \n\tabsolute error after: " + str(after.get_list()) + " \n\terror gain: " + str(difference_of_error.get_list()) + " \n\ttotal error gain:" + str(sum_of_difference_of_error) + " \n\terror sum: " + str(self.sum_of_error) + " \n\tfitness: \033[1;32;48m " + str(self.code_fitness) + "\033[0m")
            if sum_of_difference_of_error == 0:
                no_changes_counter += 1
            print("before code: \t" + str(before_codes))
            print("after code: \t" + str(codes))
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            codes = before_codes.copy()
            # if l == 0:
            before = Tape(None, None, other_tape=after)
        # self.code_fitness.reverse()
        self.structure.adjust_probabilities(codes, self.code_fitness)
        self.print_structure()
        # print(str(self.structure))
        time.sleep(0.3)

    def print_structure(self):
        buffer_size = 0 # This makes it so changes appear without buffering
        with open('structure.log', 'w', -1) as f:
            f.write(str(self.structure))
            # time.sleep(1)
        f.close()

# def step(self):
#     no_changes_counter = 0
#     codes = self.structure.generate_all_tapes()
#     before_codes = codes.copy()
#     code = ""
#     codes_portion = []
#     for l, _ in enumerate(codes):
#         tape = Tape(None, None, other_tape=self.input_tape)
#         before = Tape(None, None, input_array=list(np.abs(np.array(get_error_tape(tape, self.target_tape).get_list()))))
#         self.code_fitness = list(np.zeros(len(codes)))
#         codes.reverse()
#         for i, code in enumerate(codes[-(l+1):]):
#             if i != l:
#                 input_tape = Tape(None, None, codes[-l+i])
#             else:
#                 input_tape = tape
#             print("Applyng code: " + code + " on input: " + str(input_tape) + " i, l: " + str(i) + ", " + str(l))
#             try:
#                 self.code_fitness[i] = len(code) * self.lenght_cohefficient
#                 interpreter = Interpreter(code, input_tape, self.target_tape, debug=False)
#                 interpreter.run()
#             except RecursionError:
#                 self.code_fitness[i] += 1000
#             if i != l:
#                 codes[-l+i] = str(input_tape)
#         codes.reverse()
#         # evaluation
#
#         after_list = list(np.abs(np.array(get_error_tape(tape, self.target_tape).get_list())))
#         after = Tape(None, None, input_array=after_list)
#         difference_of_error = get_error_tape(before, after)
#         sum_of_difference_of_error = sum(difference_of_error.get_list())
#         self.code_fitness[self.structure.level] -= sum_of_difference_of_error
#         sum_of_error = sum(after_list)
#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#         print("Code analysis for code: " + str(code) + " \n\tabsolute error before: " + str(before.get_list()) + " \n\tabsolute error after: " + str(after.get_list()) + " \n\terror gain: " + str(difference_of_error.get_list()) + " \n\ttotal error gain:" + str(sum_of_difference_of_error) + " \n\terror sum: " + str(sum_of_error) + " \n\tfitness: " + str(self.code_fitness))
#         if sum_of_difference_of_error == 0:
#             no_changes_counter += 1
#         self.structure.adjust_probabilities(self.code_fitness)
#         # print(str(self.structure))
#         print("before code: \t" + str(before_codes))
#         print("after code: \t" + str(codes))
#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#         codes = before_codes.copy()
