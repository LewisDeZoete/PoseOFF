class tmp:
    def __init__(self, fn):
        self.fn = fn
    
    def __call__(self, input):
        return self.fn(input)

def fn(input):
    return input**2

if __name__=='__main__':
    cls = tmp(fn)
    print(cls(3)) # 9