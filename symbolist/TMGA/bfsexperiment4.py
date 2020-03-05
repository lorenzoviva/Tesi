import bfsstructurecopy8 as bfsstructure
from tapes import Tape

# 1 cipher sum (388 - 1684 operand compatible)
# input_tapes = [Tape(None, None, input_text="22"), Tape(None, None, input_text="44"), Tape(None, None,  input_text="27")]
# output_tapes = [Tape(None, None, input_text="4"), Tape(None, None, input_text="8"), Tape(None, None,  input_text="9")]

# nput_tapes = [Tape(None, None, input_text="22"), Tape(None, None, input_text="44"), Tape(None, None,  input_text="27")]
# output_tapes = [Tape(None, None, input_text="-<>-----------------------------------------------[-<+>]"), Tape(None, None, input_text="-<>-----------------------------------------------[-<+>]"), Tape(None, None,  input_text="-<>-----------------------------------------------[-<+>]")]

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

# 1 cipher product ()
# input_tapes = [Tape(None, None, input_text="22"), Tape(None, None, input_text="33"), Tape(None, None,  input_text="42")]
# output_tapes = [Tape(None, None, input_text="4"), Tape(None, None, input_text="9"), Tape(None, None,  input_text="8")]

# 1 cipher double
# input_tapes = [Tape(None, None, input_text="2"), Tape(None, None, input_text="3"), Tape(None, None,  input_text="4")]
# output_tapes = [Tape(None, None, input_text="4"), Tape(None, None, input_text="6"), Tape(None, None,  input_text="8")]


blocks = [bfsstructure.BuildingBlock(Tape(None, None, ".-'.+'.")), bfsstructure.BuildingBlock(Tape(None, None, ".<'.>'."))] # 700
# blocks = [bfsstructure.BuildingBlock(Tape(None, None, ".-'.+'.-.+.")), bfsstructure.BuildingBlock(Tape(None, None, ".<'.>'.<.>."))] # 2791
# blocks = [bfsstructure.BuildingBlock(Tape(None, None, ".-'.+'.-^.+^.-*.+*.-/.+/.")), bfsstructure.BuildingBlock(Tape(None, None, ".<'.>'."))] # 9000
# blocks = [ bfsstructure.BuildingBlock(Tape(None, None, "'./.*.^.")), bfsstructure.BuildingBlock(Tape(None, None, ".]'.['.>'.<'.-'.+'."))] # 9000
# blocks = [bfsstructure.BuildingBlock(Tape(None, None, ".-'.+'.-^.+^.-*.+*.-/.+/.")), bfsstructure.BuildingBlock(Tape(None, None, ".<'.>'.<^.>^.<*.>*.</.>/."))] # 17000
architecture = bfsstructure.Architecture(input_tapes, output_tapes, blocks)
architecture.run()