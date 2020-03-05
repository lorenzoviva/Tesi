from evolvedevolution import GeneticAlgorithm
from tapes import Tape
import lollocodeinterpreter

input_tape = Tape(None, None, "coda")
target_tape = Tape(None, None, "cane")
ga = GeneticAlgorithm(input_tape, target_tape, 30)
ga.population.append(Tape(None, None, lollocodeinterpreter.translate_from_bf_code(">------------->++++++++++>++++")))
ga.apply_genetic_algorithm()


