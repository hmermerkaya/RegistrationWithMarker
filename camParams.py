# id=021401644747 intrinsic parameters
def cam021401644747():
    cx=2.59602295e+02
    cy=2.04101303e+02
    fx=3.67825592e+02
    fy=3.67825592e+02
    return cx,cy,fx,fy

#id=291296634347 intrinsic parameters
def  cam291296634347():
    cx=2.57486603e+02
    cy=1.99466003e+02
    fx=3.64963013e+02
    fy=3.64963013e+02
    return cx,cy,fx,fy

# id=003726334247 intrinsic parameters
def cam003726334247():
    cx=2.54701508e+02
    cy=2.01162506e+02
    fx=3.64941895e+02
    fy=3.64941895e+02
    return cx,cy,fx,fy

# id=021401644747 intrinsic parameters
class cam021401644747:
    cx=2.59602295e+02
    cy=2.04101303e+02
    fx=3.67825592e+02
    fy=3.67825592e+02


#id=291296634347 intrinsic parameters
class cam291296634347:
    cx=2.57486603e+02
    cy=1.99466003e+02
    fx=3.64963013e+02
    fy=3.64963013e+02

# id=003726334247 intrinsic parameters
class cam003726334247:
    cx=2.54701508e+02
    cy=2.01162506e+02
    fx=3.64941895e+02
    fy=3.64941895e+02

class camera:
    def __init__(self, camserial):
        if camserial == "021401644747":
            self.cx=2.59602295e+02
            self.cy=2.04101303e+02
            self.fx=3.67825592e+02
            self.fy=3.67825592e+02
        elif camserial == "291296634347":
            self.cx=2.57486603e+02
            self.cy=1.99466003e+02
            self.fx=3.64963013e+02
            self.fy=3.64963013e+02
        elif camserial == "003726334247":
            self.cx=2.54701508e+02
            self.cy=2.01162506e+02
            self.fx=3.64941895e+02
            self.fy=3.64941895e+02
        else:
            print("It's not defined camera serial")
