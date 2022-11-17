import display_models

def DirecModel():
    d = display_models.DisplayDirec()
    return d.shot_display()

def InverseModel():
    i = display_models.DisplayInverse()
    return i.inverse_display()