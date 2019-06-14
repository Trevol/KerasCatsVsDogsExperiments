class TestObj:
    pass


inst = TestObj()


def myMethod(self):
    print('11111', self)


inst.mmm = myMethod.__get__(inst, inst.__class__)

inst.mmm()
