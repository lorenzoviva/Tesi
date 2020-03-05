import numpy.random as random
import numpy as np
from tapes import Tape
from bfninterpreter import  Interpreter
from time import sleep

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

def parenthesis_match(number):
    z = np.array([int(y) for y in bin(number)[2:]])
    z *= 2
    z -= 1
    for i, _ in enumerate(z):
        if sum(z[:i]) < 0:
            return False
    return sum(z) == 0


class Architecture:
    structures = []
    input_tapes = None
    output_tapes = None
    longhest_operation = 0
    last_structure_changed = -1
    level = 0
    current_error = []
    iterations = 0
    current_best_code = ""
    # index = 0
    flip = False

    def __init__(self, input_tapes, output_tapes):
        self.structures.append(Structure())
        self.input_tapes = input_tapes
        self.output_tapes = output_tapes
        self.longhest_operation = self.get_longhest_operation()
        self.last_structure_changed = -1
        self.current_error = []
        self.iterations = 0
        self.flip = False

    def run(self):
        code_writing_code = self.generate()
        code = self.get_code(code_writing_code)
        self.current_error = list(np.ones(len(self.input_tapes))*-1)
        self.current_error, error_after, difference_of_error, total_error_after = self.get_multiple_input_error(self.current_error, code)
        self.print_situation(error_after, difference_of_error, total_error_after, code_writing_code, code)
        self.set_structures_error(total_error_after)
        if difference_of_error >= 0:
            self.next()
            print(">= 0: next")

        else:
            self.current_best_code = code
            # self.reset_structure(self.last_structure_changed)
            self.current_error = error_after
            # self.structures[self.last_structure_changed].current_error = total_error_after
            print("< 0: reset all")
        self.print_situation(error_after, difference_of_error, total_error_after, code_writing_code, code)

        while total_error_after > 1:
            sleep(0.001)
            self.iterations += 1
            code_writing_code = self.generate()
            code = self.get_code(code_writing_code)
            self.current_error, error_after, difference_of_error, total_error_after = self.get_multiple_input_error(self.current_error, code)
            self.print_situation(error_after, difference_of_error, total_error_after, code_writing_code, code)
            self.set_structures_error(total_error_after)
            if difference_of_error >= 0:
                # if self.last_structure_changed != -1 and self.structures[self.last_structure_changed].is_better(total_error_after):
                #     self.reset_structure(self.last_structure_changed)
                #     self.structures[self.last_structure_changed].current_error = total_error_after
                #     print(">= 0: reset structure ")
                # else:
                self.next()
                print(">= 0: next")
            else:
                self.current_best_code = code
                # self.reset_structure(self.last_structure_changed)
                self.current_error = error_after
                # self.structures[self.last_structure_changed].current_error = total_error_after
                print("< 0: reset all")
            self.print_situation(error_after, difference_of_error, total_error_after, code_writing_code, code)

    def print_situation(self, error_after, difference_of_error, total_error_after, code_writing_code, code):
        last_changed_structure_current_error = ""
        if self.last_structure_changed != -1:
            last_changed_structure_current_error += " index: " + str(self.last_structure_changed) + " local error: " + str(self.structures[self.last_structure_changed].current_error) + " transofrmation applied: " + self.structures[self.last_structure_changed].generate() + "\n\t["
        for i, structure in enumerate(self.structures):
            if isinstance(structure, (Structure,)):
                last_changed_structure_current_error += str(structure.current_error) + " "
        print("{:<25s} {:<50s} {:<50s} {:<50s} {:<50s}\n{:s}".format(
            "Iteration number: " + str(self.iterations),
            " current error: " + str(self.current_error),
            " error after: " + str(error_after),
            " difference of error: " + str(difference_of_error),
            " total errror after: " + str(total_error_after),
            "LAST TRANSFORMATION " + last_changed_structure_current_error,
            ))
        print("\tgenerator code:\t\t " + self.get_structure_string())
        print("\tcode:\t\t\t\033[1;31;48m " + code_writing_code + "\033[0m")
        print("\tconfirmed code:\t\t " + self.get_confirmed_code())
        print("\tfinal code:\t\t " + code)
        print("\tcurrent best code:\t\033[1;32;48m " + self.current_best_code + "\033[0m")
        print("################################################\n\n\n################################################")

    def set_structures_error(self, total_error_after):
        for i, structure in enumerate(self.structures):
            if isinstance(structure, (Structure,)):
                if self.last_structure_changed != -1 and i == self.last_structure_changed and structure.is_better(total_error_after):
                    self.reset_structure(i)
                if structure.is_better(total_error_after):
                    structure.current_error = total_error_after

    def get_confirmed_code(self):
        code = ""
        for structure in self.structures:
            if isinstance(structure, (Structure,)):
                code += structure.confirmed_code
            else:
                code += structure
        return code

    def get_structure_string(self):
        result = ""
        for structure in self.structures:
            dot_comma = "\033[1;34;48m;\033[0m"

            if isinstance(structure, (Structure,)):
                result += str(structure) + dot_comma
            else:
                result += structure + dot_comma
        return result

    def reset_structure(self, index):
        if index != -1:
            self.structures[index].reset(confirm_code=True)
        # for structure in self.structures:
        #     if isinstance(structure, (Structure,)):
        #         structure.reset(confirm_code=True)

    def generate(self):
        code = ""
        for structure in self.structures:
            if isinstance(structure, (Structure,)):
                code += structure.generate()
            else:
                code += structure
        return code



    def get_optimization_queue(self):
        queue = []
        for i, structure in enumerate(self.structures):
            if isinstance(structure, (Structure,)):
                queue.append(i)
        # if self.flip:
        queue.reverse()
        # self.flip = not self.flip
        return queue


    # def get_optimization_queue(self):
    #     hierarchy = {}
    #     for i, structure in enumerate(self.structures):
    #         if isinstance(structure, (Structure,)):
    #             if structure.level not in hierarchy.keys():
    #                 hierarchy[structure.level] = []
    #             # hierarchy[structure.level].append(i)
    #             hierarchy[structure.level] = [i] + hierarchy[structure.level]
    #     queue = []
    #     keys = list(hierarchy.keys())
    #     # keys.sort(reverse=True)
    #     for key in keys:
    #         queue += hierarchy[key]
    #     return queue

    # def next(self):
    #     queue = self.get_optimization_queue()
    #     # if self.index >= len(queue):
    #     self.index = 0
    #     while self.structure_is_compleated(queue[self.index]):
    #         self.index += 1
    #         if self.index >= len(queue):
    #             self.index = 0
    #             self.next_structures()
    #             self.last_structure_changed = -1
    #         else:
    #             self.structures[queue[self.index]].reset()
    #             self.last_structure_changed = self.index
    #
    #     self.structures[queue[self.index]].next()
    #     self.last_structure_changed = queue[self.index]
    #     # self.index += 1
    #
    #
    # def structure_is_compleated(self, index):
    #     return len(self.structures[index].building_blocks) > self.longhest_operation*2 or self.structures[index].confirmed_code.count("<") + self.structures[index].confirmed_code.count(">") > self.longhest_operation*2

    def next(self):
        queue = self.get_optimization_queue()
        for index in queue:
            if len(self.structures[index].building_blocks) > self.longhest_operation*2 or self.structures[index].confirmed_code.count("<'") + self.structures[index].confirmed_code.count(">'") > self.longhest_operation*2:  #
                if index == queue[-1]:
                    self.next_structures()
                    self.last_structure_changed = -1
                else:
                    self.structures[index].reset()
                    self.last_structure_changed = index
            else:
                self.structures[index].next()
                self.last_structure_changed = index
                break

    def next_structures(self):
        self.level += 2
        while not parenthesis_match(self.level):
            self.level += 2
        self.structures = self.get_structures()
        self.current_error = list(np.ones(len(self.input_tapes))*-1)

    def get_structures(self):
        structures = []
        x = self.level
        current_level = 0
        while x > 0:
            if len(structures) == 0 or not isinstance(structures[0], (Structure,)):
                structures = [Structure(level=current_level)] + structures
            if x % 2:
                structures = ["['"] + structures
                current_level -= 1
            else:
                structures = ["]'"] + structures
                current_level += 1
            if not isinstance(structures[0], (Structure,)):
                structures = [Structure(level=current_level)] + structures
            x = x >> 1
        return structures




    def get_longhest_operation(self):
        longhest_op = 0
        for intput_tape in self.input_tapes:
            if len(str(intput_tape)) > longhest_op:
                longhest_op = len(str(intput_tape))
        for output_tape in self.output_tapes:
            if len(str(output_tape)) > longhest_op:
                longhest_op = len(str(output_tape))
        return longhest_op

    def get_code(self, code_writing_code):
        code_tape = Tape(None, None)
        interpreter = Interpreter(code_writing_code, code_tape, Tape(None, None), debug=False)
        interpreter.run()
        code = str(code_tape)
        return code

    def get_error(self, input_tape, error_before, output_tape, code):
        if error_before == -1:
            error_before = sum(list(np.abs(np.array(get_error_tape(input_tape, output_tape).get_list()))))
        tape = Tape(None, None, other_tape=input_tape)
        try:
            interpreter = Interpreter(code, tape, Tape(None, None, other_tape=output_tape), debug=False)
            interpreter.run()
        except RecursionError:
            return error_before, error_before+1000, 1000
        error_after = sum(list(np.abs(np.array(get_error_tape(tape, output_tape).get_list()))))
        error_after += float(interpreter.iterations*0.3) / interpreter.max_iterations
        error_after += float(len(code)*0.3) / interpreter.max_iterations
        difference_of_error = error_after - error_before
        return error_before, error_after, difference_of_error

    def get_multiple_input_error(self, errors_before, code):
        total_error_difference = 0
        total_error_after = 0
        errors_after = []
        for i, input_tape in enumerate(self.input_tapes):
            output_tape = self.output_tapes[i]
            error_before = errors_before[i]
            error_before, error_after, difference_of_error = self.get_error(input_tape, error_before, output_tape, code)
            errors_after.append(error_after)
            total_error_after += error_after
            total_error_difference += difference_of_error
        return errors_before, errors_after, total_error_difference, total_error_after



class Structure:
    building_blocks = None
    current_block = 0
    level = 0
    confirmed_code = ""
    current_error = -1

    def __init__(self, level=0):
        inc_dec_block = BuildingBlock(Tape(None, None, ".-'.+'."))
        move_block = BuildingBlock(Tape(None, None, ".<'.>'."))
        self.building_blocks = [inc_dec_block, move_block]
        self.current_block = 0  # or not??
        self.confirmed_codeconfirmed_code = ""
        self.level = level
        # if level > 0:
        #     self.next()

    def generate(self):
        instruction, was_empty = self.building_blocks[0].generate()
        i = 1
        while was_empty or i < self.current_block:
            if i >= len(self.building_blocks):
                inc_dec_block = BuildingBlock(Tape(None, None, ".-'.+'."))
                move_block = BuildingBlock(Tape(None, None, ".<'.>'."))
                self.building_blocks += [inc_dec_block, move_block]
            next_instruction, was_empty = self.building_blocks[i].generate()
            instruction = next_instruction + instruction
            i += 1
        if i > self.current_block:
            self.current_block = i
        return instruction  # self.confirmed_code +

    def next(self):
        i = 0
        was_empty = self.building_blocks[i].next()
        while was_empty:
            i += 1
            if i >= len(self.building_blocks):
                inc_dec_block = BuildingBlock(Tape(None, None, ".-'.+'."))
                move_block = BuildingBlock(Tape(None, None, ".<'.>'."))
                self.building_blocks += [inc_dec_block, move_block]
            was_empty = self.building_blocks[i].next()

    def is_better(self, new_error):
        return new_error < self.current_error or self.current_error == -1

    def reset(self, confirm_code=False):
        inc_dec_block = BuildingBlock(Tape(None, None, ".-'.+'."))
        move_block = BuildingBlock(Tape(None, None, ".<'.>'."))
        # move_block = BuildingBlock(Tape(None, None, ".<.>.<'.>'."))
        self.building_blocks = [inc_dec_block, move_block]
        self.current_block = 0
        if confirm_code:
            self.set_confirmed_code(self.generate())
        else:
            confirm_code = self.confirmed_code
            self.set_confirmed_code(confirm_code)
            self.current_error = -1

        # if self.level > 0:
        #     self.next()

    def set_confirmed_code(self, confirmed_code):
        self.confirmed_code = confirmed_code
        if confirmed_code:
            self.building_blocks.append(BuildingBlock(Tape(None, None, "." + confirmed_code + ".")))

    def __str__(self):
        comma = "\033[1;31;48m,\033[0m"
        str_blocks = [str(building_block) + comma for building_block in self.building_blocks]
        str_blocks.reverse()
        str_blocks = ''.join(str_blocks)
        if str_blocks:
            str_blocks = str_blocks[:-len(comma)]
        return str_blocks

class BuildingBlock:
    tape = None
    original_tape = None
    last_instruction = ""

    def __init__(self, tape):
        self.tape = Tape(None, None, other_tape=tape)
        self.original_tape = Tape(None, None, other_tape=tape)

    def generate(self, was_empty=False):
        tape_text = str(self.tape)
        found_dot = tape_text.find(".")
        if found_dot != -1:
            instruction = tape_text[:found_dot]
            self.last_instruction = instruction
        else:
            self.tape = Tape(None, None, other_tape=self.original_tape)
            return self.generate(was_empty=True)
        return instruction, was_empty

    def next(self):
        tape_text = str(self.tape)
        found_dot = tape_text.find(".")
        if found_dot != -1:
            tape_text = tape_text[found_dot+1:]
            self.tape = Tape(None, None, input_text=tape_text)
        return tape_text.find(".") == -1

    def __str__(self):
        return str(self.tape)
