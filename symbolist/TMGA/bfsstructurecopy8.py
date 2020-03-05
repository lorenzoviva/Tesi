import numpy as np
from tapes import Tape
from bfninterpreter import Interpreter
from time import sleep


BIG_NUMBER = 1000000


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
    optimization_queue_index = 0
    blocks = []

    def __init__(self, input_tapes, output_tapes, blocks, structures=None):
        if not structures and structures != []:
            self.structures.append(Structure(blocks))
        else:
            self.structures = structures
        self.input_tapes = input_tapes
        self.output_tapes = output_tapes
        self.longhest_operation = self.get_longest_operation()
        self.last_structure_changed = -1
        self.current_error = []
        self.iterations = 0
        self.flip = False
        self.optimization_queue_index = 0
        self.blocks = blocks

    def run(self):
        code_writing_code = self.generate()
        code = self.get_code(code_writing_code)
        self.current_error = list(np.ones(len(self.input_tapes))*-1)
        self.current_error, error_after, difference_of_error, total_error_after, penalties = self.get_multiple_input_error(self.current_error, code)
        self.print_situation(error_after, difference_of_error, total_error_after, code_writing_code, code)
        self.set_structures_error(error_after, penalties)
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
            self.current_error, error_after, difference_of_error, total_error_after, penalties = self.get_multiple_input_error(self.current_error, code)
            self.print_situation(error_after, difference_of_error, total_error_after, code_writing_code, code)
            self.set_structures_error(error_after, penalties)
            if difference_of_error >= 0:
                self.next()
                print(">= 0: next")
            else:
                self.current_best_code = code
                self.current_error = error_after
                print("< 0: reset all")
            self.print_situation(error_after, difference_of_error, total_error_after, code_writing_code, code)

    def print_situation(self, error_after, difference_of_error, total_error_after, code_writing_code, code):
        last_changed_structure_current_error = ""
        if self.last_structure_changed != -1:
            last_changed_structure_current_error += " index: " + str(self.last_structure_changed) + " local error: " + str(self.structures[self.last_structure_changed].current_error) + " transofrmation applied: " + self.structures[self.last_structure_changed].generate() + "\n\tStructure errors :["
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
        print("\tcode:\t\t\t\033[1;31;48m " + code_writing_code + "\033[0m")
        print("\tconfirmed code:\t\t " + self.get_confirmed_code())
        print("\tfinal code:\t\t " + code)
        print("\tcurrent best code:\t\033[1;32;48m " + self.current_best_code + "\033[0m")
        print("\nBlocks " + str(self.blocks))
        print("################################################\n\n\n################################################")

    def set_structures_error(self, error_after, penalties):
        for i in self.get_optimization_queue():
            if isinstance(self.structures[i], (Structure,)):
                if self.last_structure_changed != -1 and i == self.last_structure_changed and self.structures[i].is_better(error_after, penalties):
                    print("\tResetting: " + str(self.structures[i]) + " and confirming code: " + self.structures[i].generate())
                    self.reset_structure(i)
                    print("\tResetted: " + str(self.structures[i]) + " new code: " + self.structures[i].generate())
                    self.structures[i].update_error(error_after, penalties)
                    break

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

    def reset_structure(self, index):
        if index != -1:
            self.structures[index].reset(confirm_code=True)

    def generate(self):
        code = ""
        for structure in self.structures:
            if isinstance(structure, (Structure,)):
                code += structure.generate()
            else:
                code += structure
        return code

    def get_optimization_queue(self, structures=False):
        if structures == False:
            structures = list.copy(self.structures)
        queue = []
        if "]\'" in structures:
            first = structures.index("]\'")
            if "[\'" in structures[first+1:] and "]\'" in structures[first+1:]:
                opend = structures[first+1:].index("[\'")
                closed = structures[first+1:].index("]\'")
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

    def next(self):
        queue = self.get_optimization_queue()
        index = queue[self.optimization_queue_index]
        if self.structures[index].generate().count("<'") + self.structures[index].generate().count(">'") > self.longhest_operation+1:  # len(self.structures[index].building_blocks) > self.longhest_operation*2+1 or
        # if self.structures[index].confirmed_code.count("<'") + self.structures[index].confirmed_code.count(">'") > self.longhest_operation*2+1:  # len(self.structures[index].building_blocks) > self.longhest_operation*2+1 or
            if index == queue[-1]:
                self.next_structures()
                self.last_structure_changed = -1
                self.optimization_queue_index = 0
            else:
                self.structures[index].reset()
                self.last_structure_changed = index
                self.optimization_queue_index += 1
        else:
            self.structures[index].next()
            self.last_structure_changed = index

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
                structures = [Structure(self.blocks, level=current_level)] + structures
            if x % 2:
                structures = ["['"] + structures
                current_level -= 1
            else:
                structures = ["]'"] + structures
                current_level += 1
            if not isinstance(structures[0], (Structure,)):
                structures = [Structure(self.blocks, level=current_level)] + structures
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
        return errors_before, errors_after, total_error_difference, total_error_after, penalties

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
        return Architecture(input_tapes, output_tapes, new_blocks, structures)

class Structure:
    building_blocks = None
    current_block = 0
    level = 0
    confirmed_code = ""
    current_error = -1
    original_blocks = []

    def __init__(self, blocks, level=0):
        print("Init blocks: " + str(blocks))
        self.current_block = 0  # or not??s
        self.confirmed_code = ""
        self.level = level
        # self.original_blocks = blocks
        self.original_blocks = self.copy_blocks(blocks)
        self.reset_blocks()
        # if level > 0:
        #     self.next()

    def copy_blocks(self, blocks):
        original_blocks = []
        for block in blocks:
            original_blocks.append(BuildingBlock(Tape(None, None, other_tape=block.original_tape)))
        return original_blocks

    # def reset_blocks(self):
    #     self.building_blocks = self.copy_blocks(self.original_blocks)
    #     if self.confirmed_code:
    #         self.building_blocks = [BuildingBlock(Tape(None, None, "." + self.confirmed_code + "."))] + self.building_blocks
    #
    # def append_blocks(self):
    #     blocks = self.copy_blocks(self.original_blocks)
    #     if not self.confirmed_code or len(self.building_blocks) < 2:
    #         self.building_blocks = self.building_blocks + blocks
    #     else:
    #         self.building_blocks = [BuildingBlock(Tape(None, None, "." + self.confirmed_code + "."))] + self.building_blocks[1:] + blocks

    # def reset_blocks(self):
    #     self.building_blocks = self.copy_blocks(self.original_blocks)
    #
    # def append_blocks(self):
    #     blocks = self.copy_blocks(self.original_blocks)
    #     self.building_blocks = self.building_blocks + blocks

    def reset_blocks(self):
        self.building_blocks = self.copy_blocks(self.original_blocks)

    def append_blocks(self):
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
        instruction, was_empty = self.building_blocks[0].generate()
        i = 1
        while was_empty or i < self.current_block:
            if i >= len(self.building_blocks):
                self.append_blocks()
            next_instruction, was_empty = self.building_blocks[i].generate()
            instruction = next_instruction + instruction
            i += 1
        if i > self.current_block:
            self.current_block = i
        # return instruction
        return self.confirmed_code + instruction

    def next(self):
        i = 0
        was_empty = self.building_blocks[i].next()
        while was_empty:
            i += 1
            if i >= len(self.building_blocks):
                self.append_blocks()
                print("New blocks: " + str(self))
            was_empty = self.building_blocks[i].next()

    def is_better(self, new_error, penalties):
        if self.level == 0:
            return sum(new_error) + sum(penalties) < self.current_error or self.current_error == -1
        else:
            if max(penalties) >= BIG_NUMBER:
                return False
            else:
                return np.std(new_error) * (1 + np.mean(penalties)) < self.current_error or self.current_error == -1

    def update_error(self, new_error, penalties):
        if self.level == 0:
            self.current_error = sum(new_error) + sum(penalties)
        else:
            if max(penalties) >= BIG_NUMBER:
                self.current_error = BIG_NUMBER
            else:
                self.current_error = np.std(new_error) * (1 + np.mean(penalties))

    def reset(self, confirm_code=False):
        if confirm_code:
            self.set_confirmed_code(self.generate())
        self.reset_blocks()
        self.current_block = 0
        if not confirm_code:
            self.current_error = -1


    def set_confirmed_code(self, confirmed_code):
        self.confirmed_code = confirmed_code

    @staticmethod
    def from_str(text, level):
        blocks_text = text.split(",")
        while "" in blocks_text:
            blocks_text.remove("")
        blocks = []
        for block_text in blocks_text:
            blocks.append(BuildingBlock.from_str(block_text))
        blocks.reverse()
        return Structure(blocks, level)

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

    @staticmethod
    def from_str(text):
        return BuildingBlock(Tape(None, None, input_text=text))

    def next(self):
        tape_text = str(self.tape)
        found_dot = tape_text.find(".")
        if found_dot != -1:
            tape_text = tape_text[found_dot+1:]
            self.tape = Tape(None, None, input_text=tape_text)
        return tape_text.find(".") == -1

    def __str__(self):
        return str(self.tape)
