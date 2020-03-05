from tapes import Tape
import tapes
import os
import numpy.random as random




def translate_to_bf_code(bf_code):
    new_code = ""
    for instruction in bf_code:
        if instruction == chr(1):
            new_code += "+"
        elif instruction == chr(2):
            new_code += "-"
        elif instruction == chr(3):
            new_code += "<"
        elif instruction == chr(4):
            new_code += ">"
        elif instruction == chr(5):
            new_code += "["
        elif instruction == chr(6):
            new_code += "]"
        elif instruction == ";":
            new_code += instruction
        elif instruction == chr(7):
            new_code += "."
        elif instruction == chr(8):
            new_code += ","
        elif instruction == chr(9):
            new_code += "*"
        elif instruction == "b":
            new_code += instruction
        else:
            new_code += instruction
    return new_code

def translate_from_bf_code(bf_code):
    new_code = ""

    for instruction in bf_code:
        if instruction == "+":
            new_code += chr(1)
        elif instruction == "-":
            new_code += chr(2)
        elif instruction == "<":
            new_code += chr(3)
        elif instruction == ">":
            new_code += chr(4)
        elif instruction == "[":
            new_code += chr(5)
        elif instruction == "]":
            new_code += chr(6)
        elif instruction == ".":
            new_code += chr(7)
        elif instruction == ",":
            new_code += chr(8)
        elif instruction == "*":
            new_code += chr(9)
        else:
            new_code += instruction
    return new_code

def compute_linear_variations(interpreter):
    code = str(interpreter.code)
    choice = [translate_from_bf_code("+"), translate_from_bf_code("-")]
    error = interpreter.error_tape.get_square_of_errors()
    while error != 0:
        code += choice[0]
        interpreter.code = Tape(None, None, input_text=code)
        new_input_tape = Tape(None, None, other_tape=interpreter.current_tape)
        interpreter.focus_tape()
        interpreter.run()
        interpreter.error_tape = tapes.get_error_tape(interpreter.current_tape, interpreter.target_tape)
        interpreter.current_tape = new_input_tape
        interpreter.unfocus_tape()
        new_error = interpreter.error_tape.get_square_of_errors()
        if new_error > error:
            choice.reverse()
            code = code[:-1]
        elif new_error == error:
            code = code[:-1] + translate_from_bf_code(">")
        error = new_error
    return code

def compute_non_linear_variations(interpreters):
    codes = []
    error = 0
    all_equals = True
    max_temporal_complexity = 0
    max_spatial_complexity = 0

    for i, interpreter in enumerate(interpreters):
        codes[i] = str(interpreter[i].code)
        error += interpreter[i].error_tape.get_square_of_errors()
        if i > 0:
            all_equals &= codes[i] == codes[i-1]
        code_temporal_complexity = codes[i].count(chr(5))
        code_spatial_complexity = get_spatial_complexity(interpreter)
        if code_temporal_complexity > max_temporal_complexity:
            max_temporal_complexity = code_temporal_complexity
        if code_spatial_complexity > max_spatial_complexity:
            max_spatial_complexity = code_spatial_complexity
    if all_equals:
        return codes[0]
    estimated_temporal_complexity = max_temporal_complexity
    estimated_spatial_complexity = max_spatial_complexity
    skeleton = get_skeleton(estimated_temporal_complexity)
    body = translate_from_bf_code(get_organs(skeleton, estimated_spatial_complexity))
    for i, interpreter in enumerate(interpreters):




    choice = [translate_from_bf_code("+"), translate_from_bf_code("-")]
    while error != 0:
        code += choice[0]
        interpreter.code = Tape(None, None, input_text=code)
        new_input_tape = Tape(None, None, other_tape=interpreter.current_tape)
        interpreter.focus_tape()
        interpreter.run()
        interpreter.error_tape = tapes.get_error_tape(interpreter.current_tape, interpreter.target_tape)
        interpreter.current_tape = new_input_tape
        interpreter.unfocus_tape()
        new_error = 0
        for i, interpreter in enumerate(interpreters):
            new_error += interpreter[i].error_tape.get_square_of_errors()
        if new_error > error:
            choice.reverse()
            code = code[:-1]
        elif new_error == error:
            code = code[:-1] + translate_from_bf_code(">")
        error = new_error
    return code


def get_spatial_complexity(interpreter):
    junk, cells_visited = interpreter.run()
    return cells_visited


def get_skeleton(estimated_complexity):
    """
    adds the []
    """
    structure = []
    while estimated_complexity > 0:
        randoms_partition = structure
        while len(randoms_partition) > 0 and random.randint(2):
            if len(randoms_partition) == 1:
                randoms_partition = randoms_partition[0]
            else:
                randoms_partition = randoms_partition[random.randint(len(randoms_partition))]
        randoms_partition.append([])
        estimated_complexity -= 1
    # skeleton = str(structure).replace(", ","")[1:-1].replace("[",chr(5)).replace("]",chr(6))
    # return (skeleton.replace(chr(6), "Ā"+chr(6)).replace(chr(5), "Ā"+chr(5)) + "Ā").replace("ĀĀ", "Ā")
    return structure


def get_organs(skeleton, estimated_complexity):
    """
    adds the <>
    """
    body = get_organ(skeleton, estimated_complexity)
    return str(body).replace(", ", "").replace("'","")[1:-1]


def get_organ(skeleton, estimated_complexity):
    if isinstance(skeleton, (list,)):
        if len(skeleton) == 0:
            print("empy list: " + str(skeleton))
            skeleton.append(get_sequence(estimated_complexity))
        else:
            print("not empy list: " + str(skeleton))
            for i in range(len(skeleton) + 1):
                skeleton = skeleton[:i*2] + [get_sequence(estimated_complexity)] + skeleton[i*2:]
        new_skeleton = []
        for sub_skeleton in skeleton:
            new_skeleton += [get_organ(sub_skeleton, estimated_complexity)]
            # new_skeleton = [get_organ(sub_skeleton,esimated_complexity)] + new_skeleton
            # new_skeleton = new_skeleton + [get_organ(sub_skeleton,esimated_complexity)]
        print("new skeleton: " + str(new_skeleton))
        return new_skeleton
    return skeleton

def get_sequence(estimated_complexity):
    organ = ""
    true_complexity = (estimated_complexity - 1)
    cells_touched = random.binomial(true_complexity, 1 / true_complexity)
    left = random.randint(cells_touched + 1)
    for _ in range(left):
        organ += "<"
    for _ in range(cells_touched):
        organ += ">"
    if random.randint(2):
        # even
        for _ in range(cells_touched-left):
            organ += "<"
    else:
        if random.randint(2):
            organ.replace("<",chr(257))
            organ.replace(">","<")
            organ.replace(chr(257),">")
    return  organ

class Interpreter:
    code = ""
    level = 0
    brace_map = {}
    current_tape = None
    ground_tape = None
    target_tape = None
    error_tape = None
    max_levels = 4
    tapes = []
    errors = []
    max_iterations = 1000
    iterations = 0

    def __init__(self, code, input_tape, target_tape, max_levels, level):
        self.max_levels = max_levels
        self.error_tape = tapes.get_error_tape(input_tape, target_tape)
        if isinstance(code, (Tape,)):
            self.code = code
        else:
            self.code = Tape(None, None, input_text=code)
        self.tapes = [input_tape]
        for i in range(max_levels):
            self.tapes.append(Tape(None, None))
        self.tapes[level+1] = self.code
        self.current_tape = input_tape
        self.ground_tape = input_tape
        self.target_tape = target_tape
        self.level = level
        self.focus_tape()

    def run(self):
        cells_visited = []
        while self.code.code_pointer < self.code.get_ending()+1 and len(str(self.code)) > 0:
            if self.current_tape.pointer not in cells_visited:
                cells_visited.append(self.current_tape.pointer)
            self.step()
            # self.print_situation()
        # print(str(self.tape) + " " + str(self.tape.pointer))
        return self.iterations, len(cells_visited)

    def step(self):
        if self.iterations > self.max_iterations:
            raise RecursionError
        str_code = self.code.translate("")
        # print("adsasddas aa_"+str(translate_to_bfl_code(str(self.code))))
        instruction = chr(self.code.tape[self.code.code_pointer])
        # instruction = str_code[self.code.code_pointer]
        derivate = self.code.get_ending() > self.code.code_pointer+1 and self.code.tape[self.code.code_pointer+1] == "*"
        # derivate = len(str_code) > self.code.code_pointer+1 and str_code[self.code.code_pointer+1] == "*"
        # print("executing: " + translate_to_bfl_code(str(instruction)) + "(" + str(self.code.code_pointer) + ") of code: " + translate_to_bfl_code(str_code) + " to level: " + str(self.level))
        if not derivate:
            if ord(instruction) == 1:  # +
                self.current_tape.inc()
            if ord(instruction) == 2:  # -
                self.current_tape.dec()
            if ord(instruction) == 3:
                self.current_tape.left()  # <
            if ord(instruction) == 4:
                self.current_tape.right()  # >
            if ord(instruction) == 5 and self.current_tape.check():  # [
                self.code.code_pointer = self.brace_map[self.code.code_pointer]
            if ord(instruction) == 6 and not self.current_tape.check():  # ]
                self.code.code_pointer = self.brace_map[self.code.code_pointer]
            if ord(instruction) == 7 and self.level + 1 < self.max_levels:   # . # E.g; max lvl 3 --> [0 1 2]
                self.increase_level()
            if ord(instruction) == 8:  # ,
                self.decrease_level()
            if ord(instruction) == 9:  # *
                self.current_tape.set(chr(self.code.tape[self.code.code_pointer-1]))
                if self.level > 1:
                    self.current_tape.set(chr(self.code.tape[self.code.code_pointer]))
            # Enable for debugging purposes (getting on everyone nerves)
            # if instruction == "b":
            #     os.system("play -n synth 0.1 tri  500.0")
        self.code.code_pointer += 1
        self.iterations += 1

    def increase_level(self):
        self.unfocus_tape()
        self.tapes[self.level] = self.current_tape  # Updates the previous current level
        self.level += 1
        if len(self.tapes) <= self.level+1:
            new_code = Tape(None, None, other_tape=self.code)  # Creates a new level (in case it's not there)
            new_code.code_pointer = 0
            self.tapes.append(new_code)
        self.current_tape = self.code  # sets the current tape to the previously set code
        self.code = self.tapes[self.level+1]  # Sets the current code to the upper level
        self.focus_tape()

    def decrease_level(self):
        # print("Level: " + str(self.level))
        if self.level > 0:
            self.unfocus_tape()
            if self.level == 1:
                self.tapes[0] = Tape(None, None, other_tape=self.ground_tape)  # if we are going to operate on level 0 we reset it to input tape
            self.tapes[self.level] = self.current_tape  # Saves the previous level and current code
            self.level -= 1
            self.code = self.current_tape
            self.current_tape = self.tapes[self.level]
            self.focus_tape()
        elif self.level == 0:
            self.error_tape = tapes.get_error_tape(self.current_tape, self.target_tape)
            # new_error_tape = tapes.get_error_tape(self.current_tape, self.target_tape)
            # if new_error_tape.is_any_better(self.error_tape):


    def print_situation(self):
        print("#######################################################")
        print("\033[1;34;48m " + str(self.ground_tape) + "\033[0m\t\t\033[1;32;48m " + str(self.target_tape) + "\033[0m\t\033[1;31;48m" + str(self.error_tape) + "\033[0m\t\t\033[1;34;48m " + str(self.tapes[0].get_list()) + "\033[0m\t\t\033[1;31;48m " + str(self.error_tape.get_list()) + "\033[0m")
        print("-------------------------------------------------------")
        i = 0
        for tape in self.tapes:
            lvl_pointer = " "
            if i == self.level:
                lvl_pointer = "™"
            if i == self.level + 1:
                lvl_pointer = "©"
            tape_pointer = "    "
            code_pointer = "    "
            for l in range(0, tape.pointer):
                tape_pointer += " "
            for m in range(0, tape.code_pointer):
                code_pointer += " "
            tape_pointer += "⇑"
            code_pointer += "⇣"
            print(code_pointer)
            print('{:<50s} {:<20s}'.format(lvl_pointer + str(i) + ": " + translate_to_bf_code(str(tape)), str(tape.get_starting()) + " - " + str(tape.get_ending()) + " \033[2;37;40m" + str(tape.pointer) + "\033[0m"))
            # print(lvl_pointer + str(i) + ": " + translate_to_bf_code(str(tape)) + "\t" + str(tape.get_starting()) + " - " + str(tape.get_ending()))
            print(tape_pointer)
            i += 1
        print("#######################################################")

    def focus_tape(self):
        print(translate_to_bf_code(str(self.code)))
        if not self.code.get_starting() == self.code.get_ending():
            self.code.strip()
        else:
            self.code.code_pointer = 0
        # self.code.append(chr(8))
        # self.code.prepend(chr(7))
        # self.code.code_pointer -= 1
        self.brace_map = self.build_brace_map()
        self.iterations = 0

    def unfocus_tape(self):
        # self.code.substr(1, 2)
        # self.code.code_pointer -= 2
        pass

    def build_brace_map(self):
        temp_brace_stack, brace_map = [], {}
        for key in range(self.code.get_starting(), self.code.get_ending()+1):
            if key in self.code.tape.keys() and self.code.tape[key] == 5:
                temp_brace_stack.append(key)
            if key in self.code.tape.keys() and self.code.tape[key] == 6:
                start = temp_brace_stack.pop()
                brace_map[start] = key
                brace_map[key] = start
        return brace_map

# def get_organs(skeleton, estimated_complexity):
#     """
#     adds the <>
#     """
#     body = get_organ(skeleton, estimated_complexity)
#     return str(body).replace(", ", "").replace("'","")[1:-1]

    # minimum_current_point = 0
    # unvisited_skeleton = skeleton.copy()
    # skeleton_part = unvisited_skeleton
    # while len(unvisited_skeleton):
    #     while len(skeleton_part):
    #         skeleton_part = unvisited_skeleton.pop()
    #         if random.randint(2):
    #             pass
    #
    # # while "Ā" in skeleton:
    # #     first_marker = skeleton.find("Ā")
    # #     current_temporal_complexity = skeleton[:skeleton.find("Ā")].count(chr(5)) - skeleton[:skeleton.find("Ā")].count(chr(6))
    # return skeleton


# def get_organ(skeleton, esimated_complexity):
#     if len(skeleton) == 0:
#         organ = ""
#         true_complexity = (esimated_complexity - 1)
#         cells_touched = random.binomial(true_complexity, 1 / true_complexity)
#         left = random.randint(cells_touched + 1)
#         for _ in range(left):
#             organ += chr(3)
#         for _ in range(cells_touched):
#             organ += chr(4)
#         if random.randint(2):
#             # even
#             for _ in range(cells_touched-left):
#                 organ += chr(3)
#         # else:
#         if random.randint(2):
#             organ.replace(chr(3),chr(257))
#             organ.replace(chr(4),chr(3))
#             organ.replace(chr(257),chr(4))
#         skeleton.append(organ)
#         return skeleton
#     else:
#         for i,sub_skeleton in enumerate(skeleton):
#             get_organ(sub_skeleton, esimated_complexity)
#         skeleton = get_organ(skeleton, esimated_complexity)
#         return skeleton
#
#         # shift = random.binomial(cells_touched*2, 0.5) - true_complexity
#         # last_cell = random.binomial(cells_touched*2, 0.5) - true_complexity
#         # if last_cell < 0:
#         #
#         # go_left = cells_touched
#         # for _ in range(go_right):
#         #     organ += chr(3)
#         # for _ in range(go_left):
#         #     organ += chr(4)
#         # if random.randint(2):
#         #     # even
#         #     for _ in range(cells_touched - go_left):

# skeleton.append(organ)

# def get_skeleton(estimated_complexity):
#     structure = random_ints_with_sum(estimated_complexity)
#     skeleton = ""
#     for number_of_brackets in structure:
#         for i in range(number_of_brackets):
#             skeleton += chr(5)
#         for i in range(number_of_brackets):
#             skeleton += chr(6)
#     return (skeleton.replace(chr(6), "Ā"+chr(6)).replace(chr(5), "Ā"+chr(5)) + "Ā").replace("ĀĀ", "Ā")

# def random_ints_with_sum(n):
#     """
#     Generate positive random integers summing to `n`, sampled
#     uniformly from the ordered integer partitions of `n`.
#     """
#     m = 0
#     randoms = []
#     while m < n:
#         p = 0
#         for _ in range(n - 1):
#             p += 1
#             if random.randint(2):
#                 m += p
#                 randoms.append(p)
#                 p = 0
#         m += p + 1
#         randoms.append(p + 1)
#     return randoms
