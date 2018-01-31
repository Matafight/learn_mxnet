class ReusableGenerator:
    def __init__(self, generator_factory,i):
        self.generator_factory = generator_factory
        self.i = i

    def __iter__(self):
        return self.generator_factory(self.i)
myiter = lambda: (x * x for x in range(5))
def myfun(i):
    for x in range(i):
        yield(x)
    
squares = ReusableGenerator(myfun,4)

for x in squares: print(x)
for x in squares: print(x)
for x in myiter: print(x)