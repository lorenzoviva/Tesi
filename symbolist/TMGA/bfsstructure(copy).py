import numpy.random as random
import numpy as np
from tapes import Tape

class Structure:
    building_blocks = None
    involved_tapes = 0
    current_block = 0

    def __init__(self):
        inc_dec_block = BuildingBlock(Tape(None, None, ".-'.+'."))
        move_block = BuildingBlock(Tape(None, None, ".<'.>'."))
        self.building_blocks = [inc_dec_block, move_block]

    def get_next(self):
        instruction, next_building_block = self.building_blocks[0].generate()
        i = 1
        while next_building_block:
            if i >= len(self.building_blocks):
                inc_dec_block = BuildingBlock(Tape(None, None, ".-'.+'"))
                move_block = BuildingBlock(Tape(None, None, ".<'.>'"))
                self.building_blocks += [inc_dec_block, move_block]
            next_instruction, next_building_block = self.building_blocks[i].generate()
            instruction = next_instruction + instruction
            i += 1
        self.involved_tapes = i
        if self.involved_tapes > self.current_block:
            self.current_block = self.involved_tapes
        return instruction

    def generation_was_good(self):
        for i in range(self.involved_tapes):
            self.building_blocks[i].generation_was_good()

    def generation_was_neutral(self):
        self.building_blocks[self.involved_tapes-1].generation_was_neutral()


class BuildingBlock:
    tape = None
    original_tape = None
    last_instruction = ""

    def __init__(self, tape):
        self.tape = Tape(None, None, other_tape=tape)
        self.original_tape = Tape(None, None, other_tape=tape)

    def generate(self, next_building_block=False):
        tape_text = str(self.tape)
        found_dot = tape_text.find(".")
        if found_dot != -1:
            instruction = tape_text[:found_dot]
            self.last_instruction = instruction
            tape_text = tape_text[found_dot+1:]
        else:
            self.tape = self.original_tape
            return self.generate(next_building_block=True)
        self.tape = Tape(None, None, input_text=tape_text)
        return instruction,next_building_block

    def generation_was_good(self):
        self.tape = Tape(None, None, input_text=(self.last_instruction + "." + str(self.tape)))

    def genetation_was_bad(self):
        pass

    def generation_was_neutral(self):
        self.tape = Tape(None, None, input_text=(self.last_instruction + "." + str(self.tape)))
