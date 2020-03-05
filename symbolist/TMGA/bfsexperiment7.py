import bfsstructurecopy as bfsstructure
from tapes import Tape




# 1 cipher sum (388 - 1684 operand compatible)
input_tapes = [Tape(None, None, input_text="22"), Tape(None, None, input_text="44"), Tape(None, None,  input_text="27")]
output_tapes = [Tape(None, None, input_text="4"), Tape(None, None, input_text="8"), Tape(None, None,  input_text="9")]

# 1 cipher increment
input_tapes1 = [Tape(None, None, input_text="2"), Tape(None, None, input_text="B"), Tape(None, None,  input_text="5")]
output_tapes1 = [Tape(None, None, input_text="3"), Tape(None, None, input_text="C"), Tape(None, None,  input_text="6")]
# 1 cipher decrement
input_tapes2 = [Tape(None, None, input_text="2"), Tape(None, None, input_text="B"), Tape(None, None,  input_text="5")]
output_tapes2 = [Tape(None, None, input_text="1"), Tape(None, None, input_text="A"), Tape(None, None,  input_text="4")]

bb = ".<'.>'.,.-'.+'.,"
a1 = bfsstructure.Architecture.from_str(".<'.>'.,.-'.+'.,", input_tapes1, output_tapes1, bb)
a1.run()
a2 = bfsstructure.Architecture.from_str(".<'.>'.,.-'.+'.,", input_tapes2, output_tapes2, bb)
a2.run()

h_architecture = bfsstructure.Architecture(input_tapes, output_tapes, None, horizontal=[a1, a2])
h_architecture.run_to_first_reset()


# 1 cipher increment, 1 cipher decrement
# input_tapes1 = [Tape(None, None, input_text="22"), Tape(None, None, input_text="44"), Tape(None, None,  input_text="27")]
# output_tapes1 = [Tape(None, None, input_text="13"), Tape(None, None, input_text="35"), Tape(None, None,  input_text="18")]


# 1 cipher sum (388 - 1684 operand compatible)
# input_tapes2 = [Tape(None, None, input_text=chr(2)+chr(2)), Tape(None, None, input_text=chr(3)+chr(5)), Tape(None, None,  input_text=chr(1)+chr(8))]
# output_tapes2 = [Tape(None, None, input_text=chr(4)), Tape(None, None, input_text=chr(8)), Tape(None, None,  input_text=chr(9))]

# last char move left
# input_tapes1 = [Tape(None, None, input_text="2"+chr(0)+"2"), Tape(None, None, input_text="B"+chr(2)+chr(5)), Tape(None, None,  input_text=chr(0)+"1")]
# output_tapes1 = [Tape(None, None, input_text="25"), Tape(None, None, input_text="B"+chr(10)), Tape(None, None,  input_text="4")]

# input_tapes = [Tape(None, None, input_text="22"), Tape(None, None, input_text="44"), Tape(None, None,  input_text="27")]
# output_tapes = [Tape(None, None, input_text="-<>-----------------------------------------------[-<+>]"), Tape(None, None, input_text="-<>-----------------------------------------------[-<+>]"), Tape(None, None,  input_text="-<>-----------------------------------------------[-<+>]")]

# nonsense operation
# input_tapes = [Tape(None, None, input_text="2"), Tape(None, None, input_text="5"), Tape(None, None,  input_text="3")]
# output_tapes = [Tape(None, None, input_text="+"), Tape(None, None, input_text="+"), Tape(None, None,  input_text="+")]

# 1 cipher 0 operand sum (15976)
# input_tapes = [Tape(None, None, input_text="2"+chr(0)+"2"), Tape(None, None, input_text="4"+chr(0)+"4"), Tape(None, None,  input_text="2"+chr(0)+"7")]
# output_tapes = [Tape(None, None, input_text="4"), Tape(None, None, input_text="8"), Tape(None, None,  input_text="9")]

# 1 cipher + operand sum (16303)
# input_tapes = [Tape(None, None, input_text="2+2"), Tape(None, None, input_text="4+4"), Tape(None, None,  input_text="2+7")]
# output_tapes = [Tape(None, None, input_text="4"), Tape(None, None, input_text="8"), Tape(None, None,  input_text="9")]

# Copy once
# input_tapes = [Tape(None, None, input_text="2"), Tape(None, None, input_text="x"), Tape(None, None,  input_text="6")]
# output_tapes = [Tape(None, None, input_text="22"), Tape(None, None, input_text="xx"), Tape(None, None,  input_text="66")]

# unitary conversion
# input_tapes = [Tape(None, None, input_text="2"), Tape(None, None, input_text="3"), Tape(None, None,  input_text="6")]
# output_tapes = [Tape(None, None, input_text="11"), Tape(None, None, input_text="111"), Tape(None, None,  input_text="111111")]

# 1 cipher double
# input_tapes = [Tape(None, None, input_text="2"), Tape(None, None, input_text="3"), Tape(None, None,  input_text="4")]
# output_tapes = [Tape(None, None, input_text="4"), Tape(None, None, input_text="6"), Tape(None, None,  input_text="8")]


# 1 cipher product ()
# input_tapes = [Tape(None, None, input_text="22"), Tape(None, None, input_text="33"), Tape(None, None,  input_text="42")]
# output_tapes = [Tape(None, None, input_text="4"), Tape(None, None, input_text="9"), Tape(None, None,  input_text="8")]