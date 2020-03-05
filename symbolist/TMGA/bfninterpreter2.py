from tapes import Tape
import tapes
import os
import numpy.random as random



class Interpreter:
    code = ""
    tape = None
    brace_map = {}
    ground_tape = None
    target_tape = None
    error_tape = None
    error = None
    max_iterations = 1000
    iterations = 0
    code_pointer = 0

    debug = False

    def __init__(self , code, input_tape, target_tape, debug=True):
        self.error_tape = tapes.get_error_tape(input_tape, target_tape)
        if isinstance(code, (Tape,)):
            self.code = str(code)
        else:
            self.code = code
        self.tape = input_tape
        self.tape.pointer = self.tape.get_ending()
        self.ground_tapes = Tape(None,None, other_tape=input_tape)
        self.target_tape = target_tape
        self.brace_map = self.build_brace_map()
        self.code_pointer = 0
        self.debug = debug

    def run(self):
        while self.code_pointer < len(self.code) and len(self.code) > 0:
            try:
                self.step()
            except KeyError:
                raise RecursionError
            if self.debug:
                self.print_situation()

    def step(self):
        if self.iterations > self.max_iterations:
            raise RecursionError
        instruction = self.code[self.code_pointer]
        escape_list = ["*","'","^","/"]
        jumping = False
        if len(self.code) > self.code_pointer+1 and self.code[self.code_pointer+1] in escape_list:
            i = 2
            l = 0
            if not self.code[self.code_pointer] == self.code[self.code_pointer+1]:
                l = 1
            while len(self.code) > self.code_pointer+i and self.code[self.code_pointer+i] in escape_list:
                if not self.code[self.code_pointer+i] == self.code[self.code_pointer+i-1]:
                    l = l+1
                i = i+1
            if l % 2:
                jumping = True

        # jumping = (len(self.code) > self.code_pointer+1 and self.code[self.code_pointer+1] in escape_list and instruction not in escape_list)
        if self.debug:
            print("executing: " + instruction + "(" + str(ord(instruction)) + ") of code: " + self.code + " has to be executed? " + str(not jumping))
        if not jumping:
            if instruction == "+":  # +
                self.tape.inc()
            if instruction == "-":  # -
                self.tape.dec()
            if instruction == "<":
                self.tape.left()  # <
            if instruction == ">":
                self.tape.right()  # >
            if instruction == "[" and self.tape.check():  # [
                self.code_pointer = self.brace_map[self.code_pointer]
            if instruction == "]" and not self.tape.check():  # ]
                self.code_pointer = self.brace_map[self.code_pointer]
            if instruction == "'":  # insert
                self.tape.insert(str(self.code[self.code_pointer-1]))
            if instruction == "^" and chr(self.tape.get()) == self.code[self.code_pointer-1]:  # (chr(self.tape.get()) == self.code[self.code_pointer-1] or chr(self.tape.get()) == "^"
                # if chr(self.tape.get()) == "^":
                #     self.code_pointer -= 1
                #     self.tape.remove()
                #     self.tape.left()
                # else:
                    self.tape.remove()
            if instruction == "*" and self.tape.check():
                self.tape.set(str(self.code[self.code_pointer-1]))
            if instruction == "/" and chr(self.tape.get()) == self.code[self.code_pointer-1]:
                # if chr(self.tape.get()) == "/":
                #     self.code_pointer -= 1
                #     self.tape.remove()
                #     self.tape.left()
                # else:
                self.tape.set(chr(0))
            # Enable for debugging purposes (getting on everyone nerves)
            # if instruction == "b":
            #     os.system("play -n synth 0.1 tri  500.0")
        self.code_pointer += 1
        self.iterations += 1


    def print_situation(self):
        print("#######################################################")
        print("\033[1;34;48m " + str(self.ground_tapes) + "\033[0m\t\t\033[1;32;48m " + str(self.target_tape) + "\033[0m\t\033[1;31;48m" + str(self.error_tape) + "\033[0m\t\t\033[1;34;48m " + str(self.tape) + "\033[0m\t\t\033[1;31;48m " + str(self.error_tape.get_list()) + "\033[0m")
        print("-------------------------------------------------------")
        lvl_pointer = "™"
        tape_pointer = "    "
        for l in range(0, self.tape.pointer):
            tape_pointer += " "
        tape_pointer += "⇑"
        print('{:<50s} {:<20s}'.format(lvl_pointer + " : " + str(self.tape), str(self.tape.get_starting()) + " - " + str(self.tape.get_ending()) + " \033[2;37;40m" + str(self.tape.pointer) + "\033[0m"))
        print(tape_pointer)
        lvl_pointer = "©"
        code_pointer = "   "
        for m in range(0, self.code_pointer):
            code_pointer += " "
        code_pointer += "⇣"
        print(code_pointer)
        print('{:<50s} {:<20s}'.format(lvl_pointer + " : " + self.code, str(len(self.code))))
        print("#######################################################")

    def build_brace_map(self):
        try:
            temp_bracestack, bracemap = [], {}
            escape_list = ["*","'","^","/"]
            for position, command in enumerate(self.code):
                if command == "[" and not (len(self.code) > position + 1 and self.code[position + 1] in escape_list):
                    temp_bracestack.append(position)
                if command == "]" and not (len(self.code) > position + 1 and self.code[position + 1] in escape_list):
                    start = temp_bracestack.pop()
                    bracemap[start] = position
                    bracemap[position] = start
            return bracemap
        except IndexError:
            raise RecursionError
