from lollocodeinterpreter import Interpreter
from tapes import Tape
import lollocodeinterpreter
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
    return "[>]" + plus_n + ">-<<[+" + end.replace("+", "-") + "[+" + end + plus_n + "[-" + minus_n + ">+]]" + end + "<]+<[>->" + end.replace("+", "-") + "<]>[+" + end + "[" + minus_n + "->+]]<<" # <<<[>]<



max_levels = 2
input_tape = Tape(None,None,"1")
target_tape = Tape(None,None,"3")
other_tapes = Tape(None,None)
code_tape = Tape(None, None, chr(3)+chr(3))
third_tape = Tape(None,None, lollocodeinterpreter.translate_from_bf_code(n_numeric_counter(9, 1)))
interpreter = Interpreter(code_tape, input_tape, target_tape, max_levels)
# interpreter = Interpreter(third_tape, input_tape, target_tape, max_levels)
interpreter.tapes.append(third_tape)
x = 0
while interpreter.code.code_pointer < len(str(interpreter.code)):
    try:
        print(str(x) + ": ", end='')
        interpreter.step()
    except Exception as e: # KeyError
        interpreter.increase_level()
        interpreter.code.code_pointer += 1
        print("exception in code: " + str(interpreter.code) + " " + str(e))
        print(str(interpreter.brace_map))

    x += 1
    interpreter.print_situation()
    # sleep(0.003)
    # sleep(0.1)

print(str(interpreter.brace_map))
