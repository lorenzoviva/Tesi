from tapes import Tape
import tapes
import os




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

        while self.code.code_pointer < self.code.get_ending()+1 and len(str(self.code)) > 0:
            self.step()
            # self.print_situation()
        # print(str(self.tape) + " " + str(self.tape.pointer))
        return self.iterations

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
            # if instruction == ";" and not self.error_tape.is_all_zero():
            #     # print("executing: " + str(ord(instruction)) + "(" + str(self.code.code_pointer) + ") of code: " + translate_to_bf_code(str_code) + " to level: " + str(self.level))
            #     # self.print_situation()
            #     self.code.reset_code_pointer()
            #     self.code.code_pointer -= 1
            if ord(instruction) == 7 and self.level + 1 < self.max_levels:   # . # E.g; max lvl 3 --> [0 1 2]
                self.increase_level()
            if ord(instruction) == 8:  # ,
                self.decrease_level()
            if ord(instruction) == 9:  # *
                # self.current_tape.set(str_code[self.code.code_pointer-1])
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
        # print(str(self.code))
        if not self.code.get_starting() == self.code.get_ending():
            self.code.strip()
        else:
            self.code.code_pointer = 0
        self.code.append(chr(8))
        self.code.prepend(chr(7))
        # self.code.code_pointer -= 1
        self.brace_map = self.build_brace_map()
        self.iterations = 0

    def unfocus_tape(self):
        self.code.substr(1, 2)
        # self.code.code_pointer -= 2

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

