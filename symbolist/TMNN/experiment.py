from tapes import Tape
from bfinterpreter import Interpreter


def get_single_char():
    var = input("Input, one letter at a time:")
    return var


input_tape = Tape(0, None)
imaginary_tape = Tape(None, None)
middle_tape = (None, None)
output_tape = Tape(None, None)


var = get_single_char()
while len(var) > 0:
    input_tape.set(var)
    print("input tape: " + str(input_tape))
    print("Sub lists: " + str(input_tape.sub_lists()))
    var = get_single_char()

