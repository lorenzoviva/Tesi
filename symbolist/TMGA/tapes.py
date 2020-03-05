from math import sqrt

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

def get_absolute_error(error):
    while error > 127:
        error -= 256
    while error < -128:
        error += 256
    # junk, error = divmod(error, 128)
    return error + 128

class Tape:
    begin = None
    end = None
    tape = {}
    pointer = 0
    code_pointer = 0
    # delta = 0

    def __init__(self, begin, end, input_text = None, input_array = None, other_tape = None):
        self.begin = begin
        self.end = end
        self.tape = {}
        if begin or begin == 0:
            # Begin (might end or not) |INPUT### or |INPUT###|
            self.pointer = begin
            # self.delta = -begin
        elif end or end == 0:
            # Only end but not beginning ####INPUT|
            if input_text:
                self.pointer = end - len(input_text)+1
            else:
                self.pointer = end
        if input_text:
            index = 0
            for char in input_text:
                self.tape[self.pointer + index] = ord(char)
                index += 1
        if input_array:
            index = 0
            for code in input_array:
                self.tape[self.pointer + index] = code
                index += 1
        if other_tape:
            self.tape = other_tape.tape.copy()
            self.begin = other_tape.begin
            self.end = other_tape.end
        self.code_pointer = self.pointer
        # self.code_pointer = self.pointer - 1

    def left(self):
        if not (self.begin or self.begin == 0) or self.begin < self.pointer:
            self.pointer -= 1

    def right(self):
        if not (self.end or self.end == 0) or self.end > self.pointer:
            self.pointer += 1

    def inc(self):
        if self.pointer not in self.tape.keys() or not self.tape[self.pointer]:
            self.tape[self.pointer] = 0
        if self.tape[self.pointer] < 255:
            self.tape[self.pointer] += 1
        else:
            self.tape[self.pointer] = 0

    def dec(self):
        if self.pointer not in self.tape.keys() or not self.tape[self.pointer]:
            self.tape[self.pointer] = 0
        if self.tape[self.pointer] > 0:
            self.tape[self.pointer] -= 1
        else:
            self.tape[self.pointer] = 255

    def set(self, char):
        # print("char: " + str(char) + " translation: " + chr(char))
        self.tape[self.pointer] = ord(char)
        self.pointer += 1

    def get(self,delta=0):
        if self.pointer+delta not in self.tape.keys() or not self.tape[self.pointer+delta]:
            self.tape[self.pointer+delta] = 0
        return self.tape[self.pointer+delta]

    def append(self, string):
        ending = self.get_ending()
        index = 0
        for char in string:
            self.tape[ending + 1 + index] = ord(char)
            index += 1

    def prepend(self, string):
        starting = self.get_starting()
        index = 0
        for char in string:  # ascii
            self.tape[starting - len(string) + index] = ord(char)
            index += 1

    def insert(self, char):
        new_tape = {}
        for key in self.tape.keys():
            if key <= self.pointer:
                new_tape[key] = self.tape[key]
            else:
                new_tape[key+1] = self.tape[key]
        self.pointer += 1
        new_tape[self.pointer] = ord(char)
        self.tape = new_tape

    def remove(self, delta=0):
        new_tape = {}
        for key in self.tape.keys():
            if key < self.pointer + delta:
                new_tape[key] = self.tape[key]
            elif key > self.pointer + delta:
                new_tape[key-1] = self.tape[key]
        self.tape = new_tape

    def reset_code_pointer(self):
        self.code_pointer = self.get_starting()

    def check(self):
        # print(str(self.pointer) + " " + str(self.tape[self.pointer]))
        return self.pointer not in self.tape.keys() or not self.tape[self.pointer]

    def is_all_zero(self):
        if not self.tape:
            return True
        for value in self.tape.values():
            if value != 128:
                return False
        return True

    def get_square_of_errors(self):
        square_of_errors = 0
        for value in self.tape.values():
            square_of_errors += pow(value - 128, 2)
        return sqrt(square_of_errors)

    def is_any_better(self, previous):
        square_of_errors = self.get_square_of_errors()
        return square_of_errors < previous.get_square_of_errors()

    def sub_lists(self):
        list1 = self.get_list()
        sublist = {}
        for i in range(len(list1) + 1):
            for j in range(i + 1, len(list1) + 1):
                sub = list1[i:j]
                hash_code = self.hash_list(sub)
                sublist[hash_code] = sub
        return sublist

    @staticmethod
    def hash_list(list1):
        hash_code = 0
        for i, b in enumerate(list1):
            hash_code += pow(256, i) * b
        return hash_code

    # def sub_lists(self):
    #     list1 = self.get_list()
    #     sublist = []
    #     for i in range(len(list1) + 1):
    #         for j in range(i + 1, len(list1) + 1):
    #             sub = list1[i:j]
    #             sublist.append(sub)
    #     return sublist


    def get_list(self):
        list1 = []
        if not self.tape:
            return list1
        starting = self.get_starting()
        ending = self.get_ending()
        for index in range(starting, ending+1):
            if index in self.tape.keys() and self.tape[index]:
                list1.append(self.tape[index])
            else:
                list1.append(0)
        return list1

    def print(self, index, replacement):
        if index in self.tape.keys() and self.tape[index]:
            try:
                return chr(self.tape[index])
            except Exception:
                return replacement
        else:
            return replacement

    def translate(self, replacement):
        translation = ""
        if self.tape:
            starting = self.get_starting()
            ending = self.get_ending()
            for i in range(starting, ending+1):
                translation += self.print(i, replacement)
        return translation

    def get_ending(self):
        ending = 0
        if self.tape:
            ending = max(self.tape.keys())
        if self.end or self.end == 0:
            ending = max(ending, self.end)
        return ending

    def get_starting(self):
        starting = 0
        if self.tape:
            starting = min(self.tape.keys())
        if self.begin or self.begin == 0:
            starting = min(starting, self.begin)
        return starting

    def substr(self, begin, end):
        starting = self.get_starting()
        ending = self.get_ending()
        dstarting = 0
        dending = 0
        while dstarting < begin:
            del self.tape[starting]
            starting = self.get_starting()
            dstarting += 1
        # if self.pointer < starting:
        #         self.pointer = starting
        while dending < end:
            del self.tape[ending]
            ending = self.get_ending()
            dending += 1
        # if self.pointer > ending:
        #     self.pointer = ending

    def strip(self):

        starting = self.get_starting()
        ending = self.get_ending()
        while not self.tape[starting] or self.tape[starting] == 0:
            del self.tape[starting]
            starting = self.get_starting()
        # if self.pointer < starting:
        #         self.pointer = starting
        while not self.tape[ending] or self.tape[ending] == 0:
            del self.tape[ending]
            ending = self.get_ending()
        # if self.pointer > ending:
        #     self.pointer = ending

    def __str__(self):
            return self.translate("")

    def get_size(self):
        return self.get_ending() - self.get_starting()



    # def insert(self, char):
    #     new_tape = {}
    #     for key in self.tape.keys():
    #         if key < self.pointer:
    #             new_tape[key] = self.tape[key]
    #         elif key > self.pointer:
    #             new_tape[key+1] = self.tape[key]
    #     # self.pointer += 1
    #     new_tape[self.pointer] = ord(char)
    #     self.tape = new_tape
    #
    # def remove(self):
    #     new_tape = []
    #     for key in self.tape.keys():
    #         if key < self.pointer:
    #             new_tape[key] = self.tape[key]
    #         else:
    #             new_tape[key-1] = self.tape[key]
