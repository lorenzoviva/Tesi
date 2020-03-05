from tapes import Tape

class Interpreter:
    code = ""
    tape = None
    pointer = 0
    bracemap = {}

    def __init__(self, code, tape):
        if isinstance(code, (Tape,)):
            self.code = code.get_list()
        else:
            self.code = code
        self.tape = tape
        self.bracemap = self.buildbracemap()

    def run(self):
        while self.pointer < len(self.code):
            self.step()
        # print(str(self.tape) + " " + str(self.tape.pointer))

    def step(self):
        instruction = self.code[self.pointer]
        if instruction == "+":
            self.tape.inc()
        if instruction == "-":
            self.tape.dec()
        if instruction == "<":
            self.tape.left()
        if instruction == ">":
            self.tape.right()
        if instruction == "[" and self.tape.check():
            self.pointer = self.bracemap[self.pointer]
        if instruction == "]" and not self.tape.check():
            self.pointer = self.bracemap[self.pointer]
        if instruction == ".":
            print(self.tape.print(self.tape.pointer, ""))
        if instruction == "*":
            self.tape.set(self.code[self.pointer-1])
        self.pointer += 1
        # print(str(self.tape) + " " + str(self.tape.pointer))

    # print(self.pointer)

    def buildbracemap(self):
        temp_bracestack, bracemap = [], {}

        for position, command in enumerate(self.code):
            if command == "[":
                temp_bracestack.append(position)
            if command == "]":
                start = temp_bracestack.pop()
                bracemap[start] = position
                bracemap[position] = start
        return bracemap
