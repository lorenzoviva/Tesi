from bfsstructurecopy2 import Structure, BuildingBlock
from tapes import Tape
from bfninterpreter import Interpreter
import numpy as np

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

def get_code(code_writing_code, other_code):
    code_tape = Tape(None, None, input_text=other_code)
    interpreter = Interpreter(code_writing_code, code_tape, Tape(None, None), debug=False)
    interpreter.run()
    code = str(code_tape)
    return code

def get_error(input_tape, error_before, output_tape, code):
    if not error_before:
        error_before = sum(list(np.abs(np.array(get_error_tape(input_tape, output_tape).get_list()))))
    tape = Tape(None, None, other_tape=input_tape)
    interpreter = Interpreter(code, tape, Tape(None, None, other_tape=output_tape), debug=False)
    interpreter.run()
    error = interpreter.error_tape
    error_after = sum(list(np.abs(np.array(get_error_tape(tape, output_tape).get_list()))))
    difference_of_error = error_after - error_before
    return error_before, error_after, difference_of_error

def get_multiple_input_error(input_tapes, errors_before, output_tapes, code):
    total_error_difference = 0
    total_error_after = 0
    errors_after = []
    for i, input_tape in enumerate(input_tapes):
        output_tape = output_tapes[i]
        error_before = errors_before[i]
        error_before, error_after, difference_of_error = get_error(input_tape, error_before, output_tape, code)
        errors_after.append(error_after)
        total_error_after += error_after
        total_error_difference += difference_of_error
    return errors_before, errors_after, total_error_difference, total_error_after

confirmed_code = ""
input_tapes = [Tape(None, None, input_text="hello"), Tape(None, None, input_text="ciao")]
output_tapes = [Tape(None, None, input_text="world"), Tape(None, None, input_text="mondo")]

structure = Structure()

code_writing_code = structure.generate()
code = get_code(code_writing_code, confirmed_code)
error_before, error_after, difference_of_error, total_error_after = get_multiple_input_error(input_tapes, list(np.zeros(len(input_tapes))), output_tapes, code)
if difference_of_error >= 0:
    structure.next()
else:
    structure = Structure()
    confirmed_code = code
    error_before = error_after

while total_error_after > 0:
    code_writing_code = structure.generate()
    code = get_code(code_writing_code, confirmed_code)
    error_before, error_after, difference_of_error, total_error_after = get_multiple_input_error(input_tapes, error_before, output_tapes, code)
    print("code: " + code_writing_code + " error: " + str(error_after) + " difference of error: " + str(difference_of_error) + " confirmed code: " + confirmed_code + " final code:" + code)

    if difference_of_error >= 0:
        structure.next()
    else:
        structure = Structure()
        confirmed_code = code
        error_before = error_after



