from tapes import Tape
import os

def get_absolute_error(error):
    while error > 127:
        error -= 256
    while error < -128:
        error += 256
    # junk, error = divmod(error, 128)
    return error + 128

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


class Interpreter:
    code = ""
    level = 0
    bracemap = {}
    current_tape = None
    ground_tape = None
    target_tape = None
    error_tape = None
    code_tape = None
    max_levels = 4
    tapes = []
    errors = []

    def __init__(self, code, input_tape, target_tape, max_levels):
        self.max_levels = max_levels
        self.error_tape = Tape(None, None)
        if isinstance(code, (Tape,)):
            self.code = code
        else:
            self.code = Tape(None, None, input_text=code)
        self.tapes = [input_tape, code]
        self.current_tape = input_tape
        self.ground_tape = input_tape
        self.target_tape = target_tape
        self.code_tape = code
        self.focus_tape()

    def run(self):
        while self.code.code_pointer < len(str(self.code)):
            self.step()
        # print(str(self.tape) + " " + str(self.tape.pointer))

    def step(self):
        str_code = self.code.translate("")
        instruction = str_code[self.code.code_pointer]
        derivate = len(str_code) > self.code.code_pointer+1 and str_code[self.code.code_pointer+1] == "*"
        # print("executing: " + str(instruction) + "(" + str(self.code.code_pointer) + ") of code: " + str_code + " to level: " + str(self.level))
        if not derivate:
            if instruction == "+":
                self.current_tape.inc()
            if instruction == "-":
                self.current_tape.dec()
            if instruction == "<":
                self.current_tape.left()
            if instruction == ">":
                self.current_tape.right()
            if instruction == "[" and self.current_tape.check():
                self.code.code_pointer = self.bracemap[self.code.code_pointer]
            if instruction == "]" and not self.current_tape.check():
                self.code.code_pointer = self.bracemap[self.code.code_pointer]
            if instruction == ";" and not self.error_tape.is_all_zero():
                print("executing: " + str(instruction) + "(" + str(self.code.code_pointer) + ") of code: " + str_code + " to level: " + str(self.level))
                self.print_situation()
                self.code.reset_code_pointer()
            if instruction == "." and self.level + 1 < self.max_levels:  # E.g; max lvl 3 --> [0 1 2]
                self.increase_level()
            if instruction == ",":
                self.decrease_level()
            if instruction == "*":
                self.current_tape.set(str_code[self.code.code_pointer-1])
                if self.level > 1:
                    self.current_tape.set(str_code[self.code.code_pointer])
            if instruction == "b":
                os.system("play -n synth 0.1 tri  500.0")
        self.code.code_pointer += 1

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
            self.error_tape = get_error_tape(self.current_tape, self.target_tape)

    def print_situation(self):
        print("#######################################################")
        print(" " + str(self.ground_tape) + "\t\t" + str(self.target_tape) + "\t" + str(self.error_tape) + "\t\t" + str(self.tapes[0].get_list()) + "\t\t" + str(self.error_tape.get_list()))
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
            print('{:<50s} {:<10s}'.format(lvl_pointer + str(i) + ": " + str(tape), str(tape.get_starting()) + " - " + str(tape.get_ending())))
            print(tape_pointer)
            i += 1
        print("#######################################################")

    def focus_tape(self):
        self.code.append(",;")
        self.code.prepend(".")
        # self.code.code_pointer -= 1
        self.bracemap = self.build_brace_map()

    def unfocus_tape(self):
        self.code.substr(1, 2)
        # self.code.code_pointer -= 2

    def build_brace_map(self):
        temp_brace_stack, brace_map = [], {}
        for position, command in enumerate(str(self.code)):
            if command == "[":
                temp_brace_stack.append(position)
            if command == "]":
                start = temp_brace_stack.pop()
                brace_map[start] = position
                brace_map[position] = start
        return brace_map

