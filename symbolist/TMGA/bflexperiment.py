from bflinterpreter import Interpreter
from tapes import Tape
from time import sleep


def nthcopy(n):
    forward = ""
    backward = ""
    code = ""
    for i in range(0, n+1):
        forward += ">"
        backward += "<"
    for i in range(0, n):
        code += "[>]"+backward[:-1]+"["+forward[:-1]+"+>+"+backward+"-]"+forward+"[-"+backward+"+"+forward+"]"+backward[:-1]
    return code


def code_writing_symbols(symbols):
    code_to_write_symbols = ""
    for ascii_char in symbols:
        for i in range(0, ord(ascii_char)):
            code_to_write_symbols += "+"
        code_to_write_symbols += ">"
    if code_to_write_symbols.endswith(">"):
        code_to_write_symbols = code_to_write_symbols[0:-1]
    return code_to_write_symbols

def n_numeric_counter(n, start):
    plus_n = ""
    for i in range(0, n):
        plus_n += "+"
    minus_n = plus_n.replace("+", "-")
    end = code_writing_symbols(chr(start+n))
    return "[>]" + plus_n + ">-<<[+" + end.replace("+", "-") + "[+" + end + plus_n + "[-" + minus_n + ">+]]" + end + "<]+<[>->" + end.replace("+", "-") + "<]>[+" + end + "[" + minus_n + "->+]]<<<[>]<"

def translate_bf_code(bf_code):
    new_code = ""
    for instruction in bf_code:
        if instruction == "+":
            new_code += chr(1)
        if instruction == "-":
            new_code += chr(2)
        if instruction == "<":
            new_code += chr(3)
        if instruction == ">":
            new_code += chr(4)
        if instruction == "[":
            new_code += chr(5)
        if instruction == "]":
            new_code += chr(6)
        if instruction == ";":
            new_code += instruction
        if instruction == ".":
            new_code += chr(7)
        if instruction == ",":
            new_code += chr(8)
        if instruction == "*":
            new_code += chr(9)
        if instruction == "b":
            new_code += instruction
    return new_code

max_levels = 1
input_tape = Tape(None,None,"10")
target_tape = Tape(None,None,"125")
# code_tape = Tape(None,None, "+")  #,-*,>*+*")
other_tapes = Tape(None,None)
# last_tape = Tape(None,None,"+**>*,!-*")
# code_tape = Tape(None,None,"<1*0*[-]")

# code_tape = Tape(None,None,"[>]<+" + code_writing_symbols(":").replace("+", "-") + ">[-]+>[-]<<[>-<[>>+<<-]]>>[<<+>>-]<[<<1*<0*<[-]]<" + code_writing_symbols(":").replace("+", "-"))
# code_tape = Tape(None,None,"[>]<+" + code_writing_symbols(":").replace("+", "-") + ">[-]+>[-]<<[>-<[>>+<<-]]>>[<<+>>-]<[<<1*0*[-]]<" + code_writing_symbols(":").replace("+", "-"))
# DECIMAL INCREMENT 2 DIGITS:
# code_tape = Tape(None,None,"[>]<+" + code_writing_symbols(":").replace("+", "-") + ">[-]+>[-]<<["+code_writing_symbols(":")+"+>-<[>>+<<-]]>>[<<+>>-]<[<<+>0*[-]]<")
# 255 - system
# code_tape = Tape(None,None,"[>]->-<<[+[[>+]>+]<]+<[>->]>[+[>+]>+]<<<[>]<")

# DECIMAL INCREMENT infinite digits:
code_tape = Tape(None,None,n_numeric_counter(10,48))



# code_tape = Tape(None,None,"[>]<+>[-]+>[-]<<[b>-<[>>+<<-]]>>[<<+>>-]<[-]<")
# last_tape = Tape(None,None,"+*,>*")
# last_tape = Tape(None,None,"+*")
third_tape = Tape(None,None, nthcopy(1))  #"[>]+*<[<]>>")

# second_tape = Tape(None,None,"[>]+[*-*]*+*,+[*-*]*-*")
# third_tape = Tape(None,None,"[>]+[*-*]*+*.*,*")

interpreter = Interpreter(code_tape, input_tape, target_tape, max_levels)
# for i in range(2, max_levels):
#     interpreter.tapes.append(Tape(None, None, other_tape=other_tapes))
# interpreter.tapes.append(last_tape)
# interpreter.tapes.append(third_tape)
x = 0
while interpreter.code.code_pointer < len(str(interpreter.code)):
    # try:
    # print(str(x) + ": ", end='')
    interpreter.step()
    # except Exception as e:
    #     # interpreter.increase_level()
    #     # interpreter.code.code_pointer += 1
    #     print("exception in code: " + str(interpreter.code) + " " + str(e))
    x += 1
    # interpreter.print_situation()
    # sleep(0.001)
    # sleep(0.1)

