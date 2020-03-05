from tapes import Tape
from bfinterpreter import Interpreter
from recursive import RecursiveNeuralNetwork


def nthcopy(n):
    forward = ""
    backward = ""
    code = ""
    for i in range(0, n+1):
        forward += ">"
        backward += "<"
    for i in range(0, n):
        code += "[>]"+backward[:-1]+"["+forward[:-1]+"+>+"+backward+"-]"+forward+"[-"+backward+"+"+forward+"]"+backward
    return code


rnn = RecursiveNeuralNetwork()
input_tape = Tape(None, None, input_text="xAxAxAx")
# interpreter = Interpreter("[>]<[>+>+<<-]>>[-<<+>>]<<", input_tape) #  rnn.pop("A")
# interpreter = Interpreter(nthcopy(7), input_tape)  # rnn.pop("A")
# interpreter = Interpreter(rnn.pop("A"), input_tape)  #
interpreter = Interpreter("+***", input_tape)  # rnn.pop("A")

while interpreter.pointer < len(interpreter.code):
    interpreter.step()
    print(str(interpreter.tape))
print(str(interpreter.tape.get_list()))

