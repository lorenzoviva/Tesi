from bfinterpreter import Interpreter
from tapes import Tape
from numpy import dot
from numpy.linalg import norm
from math import exp
from numpy import array as nparray


def point_product(lista ,listb):
    return [a*b for a, b in zip(lista, listb)]


def get_similarity(a, b):
    k = min(len(a), len(b))
    weights = []
    for i in range(0,k):
        weights.append(exp(-i))
    a = point_product(weights, a)
    b = point_product(weights, b)
    numerator = dot(a[0:k], b[0:k])
    denominator = (norm(a[0:k])*norm(b[0:k]))
    return numerator / denominator


def get_absolute_error(error):
    while error > 127:
        error -= 256
    while error < -128:
        error += 256
    # junk, error = divmod(error, 128)
    return error + 128


def code_writing_symbols(symbols):
    code_to_write_symbols = ""
    ascii_codes = symbols.encode('ascii', 'strict')
    for ascii_code in ascii_codes:
        for i in range(0, ascii_code):
            code_to_write_symbols += "+"
        code_to_write_symbols += ">"
    if code_to_write_symbols.endswith(">"):
        code_to_write_symbols = code_to_write_symbols[0:-1]
    return code_to_write_symbols


class RecursiveNeuralNetwork:

    beta = 1
    max_level = 9
    def __init__(self):
        self.main_writing_codes = {"+": code_writing_symbols("+"), "-": code_writing_symbols("-"), ">": code_writing_symbols(">"), "<": code_writing_symbols("<"), "[": code_writing_symbols("["), "]": code_writing_symbols("]"), ".": code_writing_symbols("."), ",": code_writing_symbols(",")}
        self.input_tape = Tape(None, None)
        self.working_tape = Tape(None, None)
        self.target_tape = Tape(None, None)
        self.diff_tape = Tape(None, None)
        self.memory = self.get_init_tapes()
        self.interpreter = None
        self.working_code = None

    def inplace(self, old, new):
        plusold = code_writing_symbols(old)
        minusold = code_writing_symbols(old).replace("+", "-")
        plusnew = code_writing_symbols(new)
        return "[<]>" + minusold + "[>" + minusold + "]" + plusnew + "<[" + plusold + "<]"

    def left(self, old, new):
        plusold = code_writing_symbols(old)
        minusold = code_writing_symbols(old).replace("+", "-")
        plusnew = code_writing_symbols(new)
        return "[<]>"+minusold+"[[-<+>]>"+minusold+"]"+plusold+"<"+plusnew+"<["+plusold+"<]"

    def right(self, old, new):
        plusold = code_writing_symbols(old)
        minusold = code_writing_symbols(old).replace("+", "-")
        plusnew = code_writing_symbols(new)
        return "[>]<" + minusold + "[[->+<]<" + minusold + "]" + plusold + ">" + plusnew + ">[" + plusold + ">]"

    def pop(self, old):
        plusold = code_writing_symbols(old)
        minusold = code_writing_symbols(old).replace("+", "-")
        return "[>]<"+minusold+"[<"+minusold+"]>[[-<+>]<"+plusold+">>]"

    def append(self, new):
        plusnew = code_writing_symbols(new)
        return "[>]" + plusnew

    def nthcopy(self, n):
        forward = ""
        backward = ""
        code = ""
        for i in range(0, n+1):
            forward += ">"
            backward += "<"
        for i in range(0, n):
            code += "[>]"+backward[:-1]+"["+forward[:-1]+"+>+"+backward+"-]"+forward+"[-"+backward+"+"+forward+"]"+backward
        return code

    def nthsymbol(self, n, symbol):
        forward = ""
        code = ""
        for i in range(0, n):
            forward += "*"
        for i in range(0, pow(2, n)):
            code += symbol + forward
        return code

    def train(self, inputs, targets):
        if isinstance(inputs, (list,)):
            self.input_tape = Tape(None, None, input_array=inputs)
        else:
            self.input_tape = Tape(None, None, input_text=inputs)
        if isinstance(targets, (list,)):
            self.target_tape = Tape(None, None, input_array=targets)
        else:
            self.target_tape = Tape(None, None, input_text=targets)
        self.search()

    # def search(self):
    #     level = 0
    #     self.diff_tape = Tape(None, None, input_array=[1])
    #     stack = []
    #     while not self.diff_tape.is_all_zero():
    #         tmp_level = level
    #         while level > 1:
    #             stack.append(self.working_tape)
    #             self.working_tape = self.working_code
    #             print("level : " + str(level))
    #             self.working_code = self.get_from_memory([level, get_absolute_error(0)])
    #             print("working code : " + str(self.working_code))
    #             print("working tape : " + str(self.working_tape))
    #             self.run()
    #             print("working tape*: " + str(self.working_tape))
    #             self.set_to_memory([level, get_absolute_error(0)])
    #             level -= 1
    #         print("level : " + str(level))
    #         self.working_code = self.get_from_memory([level, self.working_tape])
    #         print("working code : " + str(self.working_code))
    #         print("working tape : " + str(self.working_tape))
    #         self.run()
    #         print("working tape*: " + str(self.working_tape))
    #         print("target tape  : " + str(self.target_tape))
    #         self.diff_tape = self.get_difference_tape(self.target_tape,self.working_tape)
    #         print("derivate tape: " + str(self.diff_tape.get_list()))
    #         print("---------------------------------------------------")
    #         level = tmp_level + 1

    def search(self):
        self.diff_tape = Tape(None, None, input_array=[1])
        delta_level = 0
        cumulative_tape = Tape(None, None)
        while not self.diff_tape.is_all_zero() and delta_level < self.max_level:
            for i in range(0, delta_level+1):
                level = self.max_level - i
                print("level : " + str(level))
                self.working_code = self.get_from_memory([level, get_absolute_error(0)])
                while level > 1:
                    self.working_tape = cumulative_tape
                    print("working code : " + str(self.working_code))
                    print("working tape : " + str(self.working_tape))
                    self.run()
                    print("computed tape: " + str(self.working_tape))
                    self.working_code = self.working_tape
                    self.working_tape = cumulative_tape
                    level -= 1
                print("level : " + str(level))
                self.working_code = self.get_from_memory([level, self.input_tape])
                print("working code : " + str(self.working_code))
                print("working tape : " + str(self.input_tape))
                self.run()
                print("working tape*: " + str(self.input_tape))
                print("target tape  : " + str(self.target_tape))
                self.diff_tape = self.get_error_tape(self.target_tape, self.input_tape)
                print("derivate tape: " + str(self.diff_tape.get_list()))
                print("---------------------------------------------------")
            delta_level += 1

            while level > 1:
                self.input_tape = self.working_code
                print("level : " + str(level))
                self.working_code = self.get_from_memory([level, get_absolute_error(0)])
                print("working code : " + str(self.working_code))
                print("working tape : " + str(self.input_tape))
                self.run()
                print("working tape*: " + str(self.input_tape))
                self.set_to_memory([level, get_absolute_error(0)])
                level -= 1

            level = tmp_level + 1

    def get_error_tape(self, a, b):
        lista = a.get_list()
        listb = b.get_list()
        listc = []
        listal = len(lista)
        listbl = len(listb)
        for i in range(0, max(listal, listbl)):
            if listal > i and listbl > i:
                listc += [lista[i] - listb[i]]
            elif listal <= i and listbl > i:
                listc += [-listb[i]]
            elif listal > i and listbl <= i:
                listc += [lista[i]]
        for index, elem in enumerate(listc):
            listc[index] = get_absolute_error(elem)
        c = Tape(None,None, input_array=listc)
        return c

    # def run(self):
    #     self.interpreter = Interpreter(self.working_code, self.input_tape)
    #     self.interpreter.run()
    def run(self):
        self.interpreter = Interpreter(self.working_code, self.working_tape)
        self.interpreter.run()

    def get_from_memory(self, key):
        esp = {}
        weight = 0
        for tape in self.memory.keys():
            esp[self.memory[tape]] = exp(self.beta * get_similarity(tape.get_list(), key))
            weight += esp[self.memory[tape]]
        for tape in self.memory.keys():
            esp[self.memory[tape]] /= weight
        esp = sorted(esp.items(), key=lambda kv: kv[1], reverse=True)
        for tape, similarity in esp:
            print(str(tape) + " " + str(similarity))
        esp = esp[0][0]
        return esp

    def set_to_memory(self, key, value):
        esp = {}
        weight = 0
        for tape in self.memory.keys():
            esp[tape] = exp(self.beta * get_similarity(tape.get_list(), key))
            weight += esp[tape]
        for tape in self.memory.keys():
            esp[tape] /= weight
        esp = sorted(esp.items(), key=lambda kv: kv[1], reverse=True)
        for tape, similarity in esp:
            print(str(tape) + " " + str(similarity))
        self.memory[esp[0][0]] = value

    def get_init_tapes(self):
        memory = {}
        # error_0 = get_absolute_error(0)
        # error_plus = get_absolute_error(1)
        # error_minus = get_absolute_error(-1)

        memory[Tape(None, None, input_array=[1])] = Tape(None, None)
        # memory[Tape(None, None, input_array=[2, get_absolute_error(1)])] = Tape(None, None, input_text=self.append("+"))
        # memory[Tape(None, None, input_array=[2, get_absolute_error(-1)])] = Tape(None, None, input_text=self.append("-"))
        for level in range(2, self.max_level+1):
            n = level - 2
            memory[Tape(None, None, input_array=[level, get_absolute_error(pow(2, n))])] = Tape(None, None, input_text=self.nthsymbol(n, "+"))
            memory[Tape(None, None, input_array=[level, get_absolute_error(-pow(2, n))])] = Tape(None, None, input_text=self.nthsymbol(n, "-"))


        # for i in range(3, 9):
        #     memory[Tape(None, None, input_array=[i,  get_absolute_error(pow(2, i - 2))])] = Tape(None, None, input_text=self.nthcopy(i-2))
        return memory

###########################################################################
#######################            IDEAS            #######################
###########################################################################
# Edit language:
# +* writes a +
# +** writes +*
# ...


#   Level  | Error b  | Error a  | Content (correction)
#     1    |    25    |    24    | [>]++        Example block
# 00000001 | 00011001 | 00011000 | [            Example row
# 00000001 | 00011001 | 00011000 | >            Example row
# 00000001 | 00011001 | 00011000 | ]            Example row
# 00000001 | 00011001 | 00011000 | +            Example row
# 00000001 | 00011001 | 00011000 | +            Example row


# MODEL:
#   Level  |   input  ||   code
#     0    |    ""    ||    ""
#     1    |     0    ||    +*
#     2    |     0    ||    +**

# OLD MODEL:
#   Level  | Error b  | Error a  | Content (correction)
#     1    |    25    |    24    | [>]++        Example block
# 00000001 | 00011001 | 00011000 | [            Example row
# 00000001 | 00011001 | 00011000 | >            Example row
# memory.append(Tape(None, None, input_array=[0, error_0, error_0]))  # Matches all, initially
# memory.append(Tape(None, None, input_array=[1, error_plus, error_0] + list(self.main_writing_codes["+"].encode('ascii', "strict"))))
# memory.append(Tape(None, None, input_array=[1, error_minus, error_0] + list(self.main_writing_codes["-"].encode('ascii', "strict"))))
# memory.append(Tape(None, None, input_array=[1, error_0, error_0] + list(self.main_writing_codes[">"].encode('ascii', "strict"))))
# memory.append(Tape(None, None, input_array=[1, error_0, error_0] + list(self.main_writing_codes["<"].encode('ascii', "strict"))))

# Error codes?
# 00011000 + here : +
# 00010000 - here : -
# 11000000 + left : <+
# 10000000 - left : <-
# 00000011 + right : >+
# 00000010 - right : >-