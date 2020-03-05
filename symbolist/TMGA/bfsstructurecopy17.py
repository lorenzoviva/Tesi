import numpy as np
from tapes import Tape
from bfninterpreter2 import Interpreter
from time import sleep
import re

ESCAPE_LIST = ["'","*","^","/"]
BIG_NUMBER = 1000000
DEBUG = True

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


class Architect:


    @staticmethod
    def get_parent_all_derivate_blocks(architecture, structure):
        if architecture.layer < architecture.MAX_LAYERS:
            return Architecture(architecture.input_tapes,architecture.output_tapes, Architect.derivate_parent_blocks(structure), layer=architecture.layer+1)
        else:
            return None

    @staticmethod
    def get_parent_last_derivate_blocks(architecture, structure):
        if architecture.layer == architecture.MAX_LAYERS - 1:
            return Architecture(architecture.input_tapes,architecture.output_tapes, Architect.derivate_parent_blocks(structure), layer=architecture.layer+1)
        elif architecture.layer < architecture.MAX_LAYERS - 1:
            return Architecture(architecture.input_tapes,architecture.output_tapes, Architect.insert_parent_blocks(structure), layer=architecture.layer+1)
        else:
            return None

    @staticmethod
    def get_parent_last_derivate_structure(architecture, structure):
        if architecture.layer == architecture.MAX_LAYERS - 1:
            return Architecture(architecture.input_tapes,architecture.output_tapes, Architect.derivate_parent_structure(structure), layer=architecture.layer+1)
        elif architecture.layer < architecture.MAX_LAYERS - 1:
            return Architecture(architecture.input_tapes,architecture.output_tapes, Architect.insert_parent_structure(structure), layer=architecture.layer+1)
        else:
            return None

    @staticmethod
    def get_parent_all_derivate_structure(architecture, structure):
        if architecture.layer < architecture.MAX_LAYERS:
            return Architecture(architecture.input_tapes,architecture.output_tapes, Architect.derivate_parent_structure(structure), layer=architecture.layer+1)
        else:
            return None

    @staticmethod
    def insert_parent_blocks(structure):
        result = ""
        for block in structure.building_blocks:
            result += ".("+str(block) + ",)'."
        return Structure.from_str(result, 0)

    @staticmethod
    def insert_parent_structure(structure):
        result = ".("+str(structure) + ")'."
        return Structure.from_str(result, 0)

    @staticmethod
    def derivate_parent_blocks(structure):
        result = ""
        for block in structure.building_blocks:
            result += ".("+str(block) + ",)'."
        for block in structure.building_blocks:
            result += ".("+str(block) + ",)^."
        for block in structure.building_blocks:
            result += ".("+str(block) + ",)*."
        for block in structure.building_blocks:
            result += ".("+str(block) + ",)/."
        return Structure.from_str(result, 0)

    @staticmethod
    def derivate_parent_structure(structure):
        result = ""
        result += ".("+str(structure) + ")'."
        result += ".[<]>("+str(structure) + ")^."
        result += ".("+str(structure) + ")*."
        result += ".[<]>("+str(structure) + ")/."
        return Structure.from_str(result, 0)


class Architecture:
    structures = []
    input_tapes = None
    output_tapes = None
    longhest_operation = 0
    last_structure_changed = 0
    level = 0
    layer = 0
    current_error = []
    iterations = 0
    current_best_code = ""
    # index = 0
    optimization_queue_index = 0
    ground_blocks = []
    code = ""
    code_writing_code = ""
    MAX_LAYERS = 2
    horizontal = []
    horizontal_index = -1
    running_to_first_reset = False
    running = False
    has_to_reset = False
    original_horizontal = []


    def __init__(self, input_tapes, output_tapes, blocks, structures=None, layer=0, horizontal=False):
        if not horizontal:
            if isinstance(blocks, (list,)):
                self.ground_blocks = Structure.copy_blocks(blocks)
                blocks = Structure(blocks)
            else:
                blocks = Structure.from_str(str(blocks),0)
            self.ground_blocks = blocks.building_blocks
            self.layer = layer
            self.input_tapes = input_tapes
            self.output_tapes = output_tapes
            if not structures and structures != []:
                structure = Structure(self.ground_blocks)
                arch_structure = Architect.get_parent_last_derivate_structure(self, structure)
                structure.parent = arch_structure
                self.structures = [structure]
            else:
                self.structures = structures
        else:
            self.horizontal = []
            for architecture in horizontal:
                self.horizontal.append(architecture.copy())
                self.original_horizontal.append(architecture.copy())
            self.horizontal_index = -1
            self.ground_blocks = [BuildingBlock.from_str(".<.>.")]
            self.MAX_LAYERS = 0
            self.layer = layer
            self.input_tapes = input_tapes
            self.output_tapes = output_tapes
            if not structures and structures != []:
                structure = Structure(self.ground_blocks)
                self.structures = [structure]
            else:
                self.structures = structures
        self.running_to_first_reset = False
        self.longhest_operation = self.get_longest_operation()
        self.last_structure_changed = 0
        self.current_error = []
        self.iterations = 0
        self.running = False
        self.optimization_queue_index = 0
        self.has_to_reset = False

    def reset_architectures(self):
        self.horizontal = []
        for architecture in self.original_horizontal:
            self.horizontal.append(architecture.copy())
        self.current_error = list(np.ones(len(self.current_error))*-1)

    def run(self):
        self.running = True
        self.current_error = list(np.ones(len(self.input_tapes))*-1)
        total_error_after = self.step()
        while total_error_after > 1 and self.running:
            total_error_after = self.step()
        self.running = False

    def run_to_first_reset(self):
        self.running_to_first_reset = True
        self.run()

    def step(self):
        sleep(0.001)
        self.iterations += 1
        self.code_writing_code = self.generate()
        if not self.horizontal:
            self.code = self.get_code(self.code_writing_code)
        else:
            self.code = self.code_writing_code
        error_after, difference_of_error, total_error_after, penalties = self.get_multiple_input_error(self.current_error, self.code)
        if DEBUG:
            self.print_situation(error_after, difference_of_error, total_error_after, self.code)
        is_better = self.set_structures_error(error_after, penalties)
        if not is_better and difference_of_error >= 0:
            self.next(total_error_after)
            # is_better = self.set_structures_error(error_after, penalties)
            if DEBUG:
                print(">= 0: next")
        else:
            self.current_best_code = self.code
            self.current_error = error_after
            if DEBUG:
                print("< 0: reset all")
        if DEBUG:
            self.print_situation(error_after, difference_of_error, total_error_after, self.code)
        return total_error_after

    def generate_parent(self):
        self.running = True
        sleep(0.001)
        self.iterations += 1
        self.code_writing_code = self.generate()
        self.code = str(self.get_parent_output(self.code_writing_code, self.input_tapes[-1]))
        if DEBUG:
            print("generate parent: " + self.code + " cwc:" + self.code_writing_code + " input_tapes :" + str(self.input_tapes[-1]))
        return self.code

    def step_parent(self, error_after, penalties):
        self.running = True
        total_error_after = sum(error_after) + sum(penalties)
        if not self.current_error:
            self.current_error = list(np.ones(len(error_after))*-1)
            for i, error in enumerate(error_after):
                self.current_error[i] = (error_after[i] + penalties[i])
        difference_of_error = total_error_after - sum(self.current_error)
        if DEBUG:
            self.print_situation(error_after, difference_of_error, total_error_after, self.code, message="Parent l:" + str(self.layer))
        is_better = self.set_structures_error(error_after, penalties)

        if not is_better and difference_of_error >= 0:
            self.next(total_error_after)
            if self.last_structure_changed != -1:
                queue = self.get_optimization_queue()
                self.structures[self.last_structure_changed].update_error(error_after, penalties, self.last_structure_changed == queue[-1])
            # is_better = self.set_structures_error(error_after, penalties)
            if DEBUG:
                print(">= 0: next")
        else:
            self.current_best_code = self.code
            self.current_error = error_after
            if DEBUG:
                print("< 0: reset all")
        if DEBUG:
            self.print_situation(error_after, difference_of_error, total_error_after, self.code, message="Parent l:" + str(self.layer))
        return total_error_after

    # def step_parent(self, error_after, penalties):
    #     self.set_structures_error(error_after, penalties)
    #     if sum(error_after) + sum(penalties) >= self.current_error:
    #         self.next()
    #     else:
    #         self.current_error = sum(error_after)
    #         self.current_best_code = self.code
    #     self.print_situation(error_after, "PARENT", error_after, self.code)

    def set_structures_error(self, error_after, penalties):
        is_better = False
        lastone = self.last_structure_changed == self.get_optimization_queue()[-1]
        structure = self.structures[self.last_structure_changed]
        if self.last_structure_changed != -1 and structure.is_better(error_after, penalties, lastone):
            print("WULULUUU: the error is getting better after the tranformation on " + str(self))
            is_better = True
            structure.update_error(error_after, penalties, lastone)
            if not self.horizontal or self.horizontal_index == -1:
                if DEBUG:
                    print("\tResetting 1: " + str(structure) + " and confirming code: " + structure.generate())
                structure.reset(confirm_code=True)
                if DEBUG:
                    print("\tResetted 1: " + str(structure) + " new code: " + structure.generate())
                print("hululu: 1")
            else:
                architecture = self.optimize_current_horizontal_architecture(error_after, penalties)
                if DEBUG:
                    print("\tResetting 2: " + str(structure) + " and confirming code: " + structure.generate() + architecture.generate_parent())
                structure.reset(confirm_code=True, architecture=architecture)
                if DEBUG:
                    print("\tResetted 2: " + str(structure) + " new code: " + structure.generate()  + architecture.generate_parent())
                print("hululu: 2")
                # exit(1)
        return is_better


    def print_situation(self, error_after, difference_of_error, total_error_after,  code, message=""):
        if self.horizontal:
            print("HORIZONTAL " + str(self.horizontal) + " " + str(self.horizontal_index))
        last_changed_structure_current_error = "None"
        if self.last_structure_changed != -1:
            last_changed_structure_current_error = " index: " + str(self.last_structure_changed) + " local error: " + str(self.structures[self.last_structure_changed].current_error) + " transofrmation applied: " + self.structures[self.last_structure_changed].generate() + "\n\tStructure errors :"
        last_changed_structure_current_error += "["
        for i, structure in enumerate(self.structures):
            if isinstance(structure, (Structure,)):
                last_changed_structure_current_error += str(structure.current_error) + " "
        last_changed_structure_current_error += "]"
        print("{:<25s} {:<50s} {:<50s} {:<50s} {:<50s}\n{:s}".format(
            "Iteration number: " + str(self.iterations),
            " current error: " + str(self.current_error),
            " error after: " + str(error_after),
            " difference of error: " + str(difference_of_error),
            " total error after: " + str(total_error_after),
            "LAST TRANSFORMATION " + last_changed_structure_current_error,
            ))
        print("\tgenerator code:\t\t " + self.get_structure_string())
        print("\tcode:\t\t\t\033[1;31;48m " + self.code_writing_code + "\033[0m")
        print("\tconfirmed code:\t\t " + self.get_confirmed_code())
        print("\tfinal code:\t\t " + code)
        print("\tcurrent best code:\t\033[1;32;48m " + self.current_best_code + "\033[0m")
        print("\n\tOnly running to first reset:\t " + str(self.running_to_first_reset))
        print("\tThe architecture need to reset when restarting:\t " + str(self.has_to_reset))
        print("\tRunning:\t " + str(self.running))

    # print("\nBlocks " + str(self.ground_blocks))
        print("\n\nSTRUCTURE:\n" + str(self) + "\n")
        print("################################################\n\n"+message+"\n\n################################################")


    def optimize_current_horizontal_architecture(self, error_after, penalties):


        architecture = self.horizontal[self.horizontal_index]
        architecture.input_tapes = [Tape(None,None)]
        architecture.output_tapes = [Tape(None,None)]
        architecture.current_error = [sum(error_after) + sum(penalties)]
        # code = architecture.get_code(architecture.generate())
        architecture.running_to_first_reset = True
        return architecture


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
            dot_comma = ";"
            # dot_comma = "\033[1;34;48m;\033[0m"

            if isinstance(structure, (Structure,)):
                result += str(structure) + dot_comma
            else:
                result += structure + dot_comma
        return result

    def generate(self):
        code = ""
        for i, structure in enumerate(self.structures):
            if isinstance(structure, (Structure,)) :
                code += structure.generate()
                if self.horizontal and self.horizontal_index != -1 and self.last_structure_changed == i:
                    arch1 = self.horizontal[self.horizontal_index]
                    code += arch1.get_code(arch1.generate())
            else:
                code += structure
        return code

    def get_optimization_queue(self, structures=False):
        if structures == False:
            structures = list.copy(self.structures)
        queue = []
        closed = "]\'"
        opend = "[\'"
        if self.horizontal:
            closed = "]"
            opend = "["
        if closed in structures:
            first = structures.index(closed)
            if opend in structures[first + 1:] and closed in structures[first + 1:]:
                opend = structures[first+1:].index(opend)
                closed = structures[first+1:].index(closed)
                if opend < closed:
                    queue.append(self.structures.index(structures[first-1]))
                    queue.append(self.structures.index(structures[first+1]))
                    queue = queue + self.get_optimization_queue(structures[:first-2]+structures[first+2:])
                else:
                    queue.append(self.structures.index(structures[first-1]))
                    queue.append(self.structures.index(structures[first-3]))
                    queue = queue + self.get_optimization_queue(structures[:first-3]+structures[first+1:])
            else:  # if "[\'" not in structures[first+1:] and "]\'" not in structures[first+1:]:
                queue.append(self.structures.index(structures[first-1]))
                queue.append(self.structures.index(structures[first-3]))
                queue = queue + self.get_optimization_queue(structures[:first-3]+structures[first+1:])
        else:
            queue.append(self.structures.index(structures[0]))
        return queue

    def next(self, total_error_after=0):
        if self.horizontal and self.horizontal_index == -1:
            self.horizontal_index = 0
        else:
            queue = self.get_optimization_queue()
            index = queue[self.optimization_queue_index]
            regex = re.compile("[<>]([']|[*]|[\^]|[/]){" + str(self.layer+1) + "}[^'*/^]")
            if (not self.horizontal and len(regex.findall(self.structures[index].generate())) > self.longhest_operation+1) or (self.horizontal and len(self.structures[index].building_blocks) > self.longhest_operation+1):  # len(self.structures[index].building_blocks) > self.longhest_operation*2+1 or self.structures[index].generate().count("<'") + self.structures[index].generate().count(">'")
                # if self.structures[index].confirmed_code.count("<'") + self.structures[index].confirmed_code.count(">'") > self.longhest_operation*2+1:  # len(self.structures[index].building_blocks) > self.longhest_operation*2+1 or
                if index == queue[-1]:
                    if self.horizontal and self.horizontal_index < len(self.horizontal) - 1:
                        self.horizontal_index += 1
                        for structure in self.structures:
                            if isinstance(structure, (Structure,)):
                                structure.reset()
                                structure.current_error = -1
                                structure.confirmed_code = ""
                        self.current_error = list(np.ones(len(self.input_tapes))*-1)
                    else:
                        if self.running_to_first_reset:
                            if self.has_to_reset:
                                self.has_to_reset = False
                                print("Had to stop, but the guard prevented me!")
                            else:
                                self.running_to_first_reset = False
                                self.running = False
                                self.has_to_reset = True
                                print("Had to stop, and i stopped!")
                                return
                        self.next_structures()
                    self.last_structure_changed = self.get_optimization_queue()[0]
                    self.optimization_queue_index = 0
                else:
                    self.structures[index].reset()
                    self.last_structure_changed = index
                    self.optimization_queue_index += 1
                    self.reset_architectures()
                    self.horizontal_index = -1
                    self.last_structure_changed = queue[self.optimization_queue_index]
            else:
                self.structures[index].next(total_error_after)
                self.last_structure_changed = index


    def next_structures(self):

        self.horizontal_index = -1
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
                structure = Structure(self.ground_blocks, level=current_level)
                arch_structure = Architect.get_parent_last_derivate_structure(self, structure)
                # Architecture(self.input_tapes, self.output_tapes, self.derivate_parent_structure(structure), layer=self.layer + 1)
                structure.parent = arch_structure
                structures = [structure] + structures
            if x % 2:
                opend = "['"
                if self.horizontal:
                    opend = "["
                structures = [opend] + structures
                current_level -= 1
            else:
                closed = "]'"
                if self.horizontal:
                    closed = "]"
                structures = [closed] + structures
                current_level += 1
            if not isinstance(structures[0], (Structure,)):
                structure = Structure(self.ground_blocks, level=current_level)
                arch_structure = Architect.get_parent_last_derivate_structure(self, structure)
                structure.parent = arch_structure
                structures = [structure] + structures
            x = x >> 1
        return structures

    def get_longest_operation(self):
        longest_op = 0
        for input_tape in self.input_tapes:
            if len(str(input_tape)) > longest_op:
                longest_op = len(str(input_tape))
        for output_tape in self.output_tapes:
            if len(str(output_tape)) > longest_op:
                longest_op = len(str(output_tape))
        return longest_op

    def get_code(self, code_writing_code):
        code_tape = Tape(None, None)
        try:
            interpreter = Interpreter(code_writing_code, code_tape, Tape(None, None), debug=False)
            interpreter.run()
        except RecursionError:
            return ""
        code = str(code_tape)
        return code

    def get_parent_output(self, code, input_tape):
        output_tape = Tape(None, None, other_tape=input_tape)
        try:
            interpreter = Interpreter(code, output_tape, Tape(None, None), debug=False)
            interpreter.run()
        except RecursionError:
            return Tape(None,None)
        return output_tape

    def get_error(self, input_tape, error_before, output_tape, code):
        if error_before == -1:
            error_before = sum(list(np.abs(np.array(get_error_tape(input_tape, output_tape).get_list()))))
        tape = Tape(None, None, other_tape=input_tape)
        try:
            interpreter = Interpreter(code, tape, Tape(None, None, other_tape=output_tape), debug=False)
            interpreter.run()
            if DEBUG and self.horizontal:
                print("-" + str(self.iterations) + ": " + str(tape))
        except RecursionError:
            return error_before, error_before+BIG_NUMBER, BIG_NUMBER, BIG_NUMBER
        error_after = sum(list(np.abs(np.array(get_error_tape(tape, output_tape).get_list()))))
        penalty = float(interpreter.iterations*0.3) / interpreter.max_iterations + float(len(code)*0.3) / interpreter.max_iterations
        error_after += penalty
        difference_of_error = error_after - error_before
        return error_before, error_after, difference_of_error, penalty

    def get_multiple_input_error(self, errors_before, code):
        total_error_difference = 0
        total_error_after = 0
        errors_after = []
        penalties = []
        for i, input_tape in enumerate(self.input_tapes):
            output_tape = self.output_tapes[i]
            error_before = errors_before[i]
            error_before, error_after, difference_of_error, penalty = self.get_error(input_tape, error_before, output_tape, code)
            errors_after.append(error_after)
            penalties.append(penalty)
            total_error_after += error_after
            total_error_difference += difference_of_error
        return errors_after, total_error_difference, total_error_after, penalties

    @staticmethod
    def from_str(text, input_tapes, output_tapes, new_blocks):
        structures_text = text.split(";")
        while "" in structures_text:
            structures_text.remove("")
        structures = []
        level = 0
        for structure_text in structures_text:
            if structure_text == "]'":
                level = level - 1
                structures.append(structure_text)
            elif structure_text == "['":
                level = level + 1
                structures.append(structure_text)
            else:
                structures.append(Structure.from_str(structure_text, level))
        return Architecture(input_tapes, output_tapes, new_blocks, structures, layer=0)

    def __str__(self):
        result = ""
        for structure in self.structures:
            dot_comma = ";"
            # dot_comma = "\033[1;34;48m;\033[0m"

            if isinstance(structure, (Structure,)):

                result += str(structure)
                if structure.parent:
                    result += "\n"
                    for i in range(self.layer+1):
                        result += "\t"
                    result += "(" + str(structure.parent) + ");\n"
                    for i in range(self.layer):
                        result += "\t"
                else:
                    result += dot_comma
            else:
                result += structure + dot_comma + "\n"
        return result + " " + str(self.layer)

    @staticmethod
    def copy_structures(structures):
        copy = []
        for strucutre in structures:
            if isinstance(strucutre, (Structure, )):
                copy.append(strucutre.copy())
            else:
                copy.append(strucutre)
        return copy


    def copy(self):
        architecture = Architecture(self.input_tapes, self.output_tapes, Structure.copy_blocks(self.ground_blocks), structures=Architecture.copy_structures(self.structures), layer=self.layer, horizontal=self.horizontal)
        architecture.code = self.code
        architecture.code_writing_code = self.code_writing_code
        architecture.current_error = self.current_error
        architecture.current_best_code = self.current_best_code
        architecture.optimization_queue_index = self.optimization_queue_index
        architecture.structures = Architecture.copy_structures(self.structures)
        architecture.horizontal_index = self.horizontal_index
        architecture.last_structure_changed = self.last_structure_changed
        architecture.iterations = self.iterations
        return architecture

class Structure:
    building_blocks = None
    current_block = 0
    level = 0
    confirmed_code = ""
    current_error = -1
    original_blocks = []
    parent = None
    iterations = 0
    horizontal = False

    def __init__(self, blocks, level=0, parent=None):
        self.current_block = 0  # or not??s
        self.confirmed_code = ""
        self.level = level
        # self.original_blocks = blocks
        self.original_blocks = self.copy_blocks(blocks)
        self.hard_reset_blocks()
        self.parent = parent
        self.iterations = 0
        self.horizontal = False
        if DEBUG:
            print("Init blocks: " + str(self))

    # if level > 0:
        #     self.next()
    @staticmethod
    def copy_blocks(blocks):
        original_blocks = []
        for block in blocks:
            # original_blocks.append(BuildingBlock(Tape(None, None, other_tape=block.original_tape)))
            original_blocks.append(block.copy())
        return original_blocks

    def soft_reset_blocks(self):
        for building_block in self.building_blocks:
            building_block.reset()


    def hard_reset_blocks(self):
        if self.parent and not self.horizontal:
            # self.parent.step_parent([self.current_error], [float(self.iterations*0.3)])
            self.iterations = 0
            self.parent.input_tapes = [Tape(None,None,"")]
            self.building_blocks = Structure.from_str(self.parent.generate_parent(), 0).building_blocks
        else:
            self.building_blocks = self.copy_blocks(self.original_blocks)

    def append_blocks(self):
        if self.parent:
            self.soft_reset_blocks()
            self.parent.step_parent([self.current_error], [float(self.iterations*0.3)])
            self.iterations = 0
            if not self.horizontal:
                self.parent.input_tapes = [Tape(None,None,str(self))]
                self.building_blocks = Structure.from_str(self.parent.generate_parent(), 0).building_blocks
                print("hulalahihuuuu: 1")
            # else:
            #     if self.parent.running:
            #         print("hulalahihuuuu: 2 " + str(self.confirmed_code))
            #         self.parent.input_tapes = [Tape(None,None,"")]
            #         self.parent.generate_parent()
            #         print("hulalahihuuuu: 2 " + str(self.confirmed_code))
            #     else:
            #         print("hulalahihuuuu: 3 " + str(self.confirmed_code))
            #         self.parent.generate_parent()
            #         self.parent = None
            #         print("hulalahihuuuu: 3 " + str(self.confirmed_code))
            #         return
             #+ self.building_blocks
        else:
            blocks_text =  {} # self.copy_blocks(self.original_blocks)
            for i, block in enumerate(self.building_blocks):
                block_text = str(block.original_tape)
                if not block_text.endswith("."):
                    if block_text not in blocks_text.values():
                        blocks_text[i] = block_text
                    else:
                        for key, value in blocks_text.items():
                            if value == block_text:
                                del blocks_text[key]
                                break
                        blocks_text[i] = block_text

            if not blocks_text:
                self.building_blocks = self.building_blocks + self.copy_blocks(self.original_blocks)
                return
            keys = list(blocks_text.keys())
            keys.sort()
            keys.reverse()
            for key in keys:
                self.building_blocks.insert(key+1, BuildingBlock.from_str(blocks_text[key]))


    def generate(self):
        if self.horizontal and self.parent:
            return self.confirmed_code #+ self.parent.generate_parent()
        was_empty = True
        instruction = ""
        i = 0
        while was_empty or i < self.current_block:
            if i >= len(self.building_blocks):
                self.append_blocks()
                if DEBUG:
                    print("New blocks: " + str(self))
                i = i - 1
                break
            else:
                next_instruction, was_empty = self.building_blocks[i].generate()
                instruction = next_instruction + instruction
                i += 1
        if i > self.current_block:
            self.current_block = i
        # return instruction
        return self.confirmed_code + instruction

    def next(self, total_error_after=0):
        if self.horizontal and self.parent:
            self.parent.step_parent([self.current_error], [float(self.iterations*0.3)])
            self.iterations = 0
            if not self.parent.running:
                self.confirmed_code += self.parent.current_best_code
                self.parent = None
            else:
                return
        self.iterations += 1
        i = 0
        if self.building_blocks:
            was_empty = self.building_blocks[i].next()
            while was_empty:
                i += 1
                if i >= len(self.building_blocks):
                    self.append_blocks()
                    if DEBUG:
                        print("New blocks: " + str(self))
                    break
                else:
                    was_empty = self.building_blocks[i].next()

    def is_better(self, new_error, penalties, constant):
        if constant:
            return sum(new_error) + sum(penalties) < self.current_error or (self.current_error == -1 and sum(new_error) + sum(penalties) != -1)
        else:
            if max(penalties) >= BIG_NUMBER:
                return False
            else:
                if self.level == 0:
                    if DEBUG:
                        print("STRUCTURE ERRORS: sqrt: " + str(np.sqrt(np.std(new_error))) + " ,sums = " + str(sum(new_error) + sum(penalties)) + ", product:" + str(np.sqrt(np.std(new_error)) * (sum(new_error) + sum(penalties))))
                    return np.sqrt(np.std(new_error)) * (sum(new_error) + sum(penalties)) < self.current_error or self.current_error == -1
                else:
                    return np.std(new_error) * (1 + np.mean(penalties)) < self.current_error or self.current_error == -1

    def update_error(self, new_error, penalties, constant):
        if constant: # self.level == 0
            self.current_error = sum(new_error) + sum(penalties)
        else:
            if max(penalties) >= BIG_NUMBER or max(new_error) >= BIG_NUMBER:
                self.current_error = BIG_NUMBER
            else:
                # self.current_error = np.std(new_error) * (1 + np.mean(penalties))
                if self.level == 0:
                    self.current_error = np.sqrt(np.std(new_error)) * (sum(new_error) + sum(penalties))
                else:
                    self.current_error = np.std(new_error) * (1 + np.mean(penalties))


    def reset(self, confirm_code=False, architecture=None):
        if confirm_code:
            self.set_confirmed_code(self.generate())
            if architecture:
                # more_over = architecture.get_code(architecture.generate())
                architecture.running_to_first_reset = True
                self.parent = architecture
                self.horizontal = True
                self.building_blocks = []
                self.current_block = 0
                return
                # self.set_confirmed_code(self.generate()+more_over)
            # else:
        self.hard_reset_blocks()
        self.current_block = 0
        # if not confirm_code:
        #     self.current_error = -1


    def set_confirmed_code(self, confirmed_code):
        self.confirmed_code = confirmed_code

    @staticmethod
    def expand(matched_text):
        indexes = list(range(len(matched_text)-1))
        indexes.reverse()
        effect = ""
        index = indexes.pop(0)
        while index and matched_text[index] != ")":
            effect += matched_text[index]
            matched_text = matched_text[:index] + matched_text[index+1:]
            index = indexes.pop(0)
        matched_text = matched_text[:index] + matched_text[index+1:]
        index = indexes.pop(0)
        while index and matched_text[index] != "(":
            if not effect.startswith(matched_text[index]):
                matched_text = matched_text[:index+1] + effect + matched_text[index+1:]
            else:
               pass
            index = indexes.pop(0)
        matched_text = matched_text[:index] + matched_text[index+1:]
        return matched_text

    @staticmethod
    def split_blocks(text):
        blocks = []
        while text:
            found_comma = text.find(",")
            while found_comma != -1 and len(text) > found_comma+1 and text[found_comma+1] in ["'","*","^","/"]:
                found_comma = text.find(",", found_comma+1)
            if found_comma == -1:
                blocks.append(text)
                break
            else:
                blocks.append(text[:found_comma])
                text = text[found_comma+1:]
        return blocks


    @staticmethod
    def from_str(text, level):
        regex = re.compile("[(][^()]*[)]['*/^]*[.]")
        match = regex.search(text)
        while match:
            matched_text = text[match.start():match.end()]
            text = text[:match.start()] + Structure.expand(matched_text) + text[match.end():]
            match = regex.search(text)
        if text.endswith(","):
            text = text[:-1]

        blocks_text = Structure.split_blocks(text)
        while "" in blocks_text:
            blocks_text.remove("")
        blocks = []
        for block_text in blocks_text:
            blocks.append(BuildingBlock.from_str(block_text))
        blocks.reverse()
        return Structure(blocks, level)

    def copy(self):
        if self.parent:
            parent = self.parent.copy()
        else:
            parent = None
        structure = Structure(Structure.copy_blocks(self.building_blocks), level=self.level, parent=parent)
        structure.current_block = self.current_block
        structure.original_blocks = self.original_blocks
        structure.confirmed_code = self.confirmed_code
        # structure.current_error = self.current_error
        return structure

    def __str__(self):
        comma = ","
        # comma = "\033[1;31;48m,\033[0m"
        blocks_text = [str(building_block) + comma for building_block in self.building_blocks]
        blocks_text.reverse()
        blocks_text = ''.join(blocks_text)
        # if blocks_text:
        #     blocks_text = blocks_text[:-len(comma)]
        return blocks_text


class BuildingBlock:
    tape = None
    original_tape = None
    last_instruction = ""

    def __init__(self, tape):
        self.tape = Tape(None, None, other_tape=tape)
        self.original_tape = Tape(None, None, other_tape=tape)

    # def generate(self, was_empty=False):
    #     tape_text = str(self.tape)
    #     found_dot = tape_text.find(".")
    #     while found_dot != -1 and len(tape_text) > found_dot+1 and tape_text[found_dot+1] in self.escape_list:
    #         found_dot = tape_text.find(".", found_dot+1)
    #     if found_dot != -1:
    #         instruction = tape_text[:found_dot]
    #         self.last_instruction = instruction
    #     else:
    #         self.tape = Tape(None, None, other_tape=self.original_tape)
    #         return self.generate(was_empty=True)
    #     return instruction, was_empty

    def generate(self, was_empty=False):
        tape_text = str(self.tape)
        found_dot = tape_text.find(".")
        while found_dot != -1 and len(tape_text) > found_dot+1 and tape_text[found_dot+1] in ESCAPE_LIST:
            found_dot = tape_text.find(".", found_dot+1)
        if found_dot != -1:
            instruction = tape_text[:found_dot]
            self.last_instruction = instruction
        elif not was_empty:
            self.tape = Tape(None, None, other_tape=self.original_tape)
            return self.generate(was_empty=True)
        else:
            return "", was_empty
        return instruction, was_empty

    def reset(self):
        self.tape = Tape(None, None, other_tape=self.original_tape)

    @staticmethod
    def from_str(text):
        return BuildingBlock(Tape(None, None, input_text=text))

    def next(self):
        tape_text = str(self.tape)
        found_dot = tape_text.find(".")
        while found_dot != -1 and len(tape_text) > found_dot+1 and tape_text[found_dot+1] in ESCAPE_LIST:
            found_dot = tape_text.find(".", found_dot+1)
        if found_dot != -1:
            tape_text = tape_text[found_dot+1:]
            self.tape = Tape(None, None, input_text=tape_text)
        next = tape_text.find(".")
        while next != -1 and len(tape_text) > next+1 and tape_text[next+1] in ESCAPE_LIST:
            next = tape_text.find(".", next+1)
        return next == -1

    def copy(self):
        block = BuildingBlock(self.original_tape)
        block.tape = self.tape
        block.last_instruction = self.last_instruction
        return block

    def __str__(self):
        return str(self.tape)
