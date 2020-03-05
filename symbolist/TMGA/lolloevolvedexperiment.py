from tapes import Tape
from evolvedinterpreter import Interpreter
import tapes
import evolvedinterpreter

input_tape = Tape(None, None, "coda")
target_tape = Tape(None, None, "cane")
code = Tape(None, None, "")

interpreter = Interpreter(code, input_tape, target_tape, 1, 0)
single_variation = evolvedinterpreter.compute_linear_variations(interpreter)
print(str(input_tape) + " -> " + str(target_tape) + ": " + evolvedinterpreter.translate_to_bf_code(single_variation))



