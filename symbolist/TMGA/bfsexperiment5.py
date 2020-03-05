import bfsstructurecopy12 as bfsstructure
from tapes import Tape


# 1 cipher increment, 1 cipher decrement
# input_tapes = [Tape(None, None, input_text="22"), Tape(None, None, input_text="44"), Tape(None, None,  input_text="27")]
# output_tapes = [Tape(None, None, input_text="13"), Tape(None, None, input_text="35"), Tape(None, None,  input_text="18")]

# 1 cipher sum (388 - 1684 operand compatible)
# input_tapes = [Tape(None, None, input_text="22"), Tape(None, None, input_text="44"), Tape(None, None,  input_text="27")]
# output_tapes = [Tape(None, None, input_text="4"), Tape(None, None, input_text="8"), Tape(None, None,  input_text="9")]

# input_tapes = [Tape(None, None, input_text="22"), Tape(None, None, input_text="44"), Tape(None, None,  input_text="27")]
# output_tapes = [Tape(None, None, input_text="-<>-----------------------------------------------[-<+>]"), Tape(None, None, input_text="-<>-----------------------------------------------[-<+>]"), Tape(None, None,  input_text="-<>-----------------------------------------------[-<+>]")]

# nonsense operation
# input_tapes = [Tape(None, None, input_text="2"), Tape(None, None, input_text="5"), Tape(None, None,  input_text="3")]
# output_tapes = [Tape(None, None, input_text="+"), Tape(None, None, input_text="+"), Tape(None, None,  input_text="+")]

# 1 cipher 0 operand sum (15976)
# input_tapes = [Tape(None, None, input_text="2"+chr(0)+"2"), Tape(None, None, input_text="4"+chr(0)+"4"), Tape(None, None,  input_text="2"+chr(0)+"7")]
# output_tapes = [Tape(None, None, input_text="4"), Tape(None, None, input_text="8"), Tape(None, None,  input_text="9")]

# 1 cipher + operand sum (16303)
input_tapes = [Tape(None, None, input_text="2+2"), Tape(None, None, input_text="4+4"), Tape(None, None,  input_text="2+7")]
output_tapes = [Tape(None, None, input_text="4"), Tape(None, None, input_text="8"), Tape(None, None,  input_text="9")]

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




bb = ".<'.>'.,.-'.+'.,"
architecture = bfsstructure.Architecture.from_str(".<'.>'.,.-'.+'.,", input_tapes, output_tapes, bb)
# bb1 = ".(.<'.>'.,.-'.+'.,)'.,"
# architecture1 = bfsstructure.Architecture.from_str(".(.<'.>'.,.-'.+'.,)'.,", input_tapes, output_tapes, bb1)
# architecture.structures[0].parent = architecture1
architecture.run()
# ".<'.>'.(,.-'.+'.),;[';.-'.+'.(,.<'.>'.),;]';.<'.>'.(,.-'.+'.),;"