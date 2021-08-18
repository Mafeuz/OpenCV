points = []

def verify_arms_UP(points):
    # Arm LEFT
    # position horizontal
    head_V = 0
    wristleft_H = 0
    # position veritical
    shoulderleft_H = 0
    wristleft_V = 0

    # Arm RIGHT
    # position horizontal
    wristright_H = 0
    # position veritical
    wristright_V = 0
    shoulderright_H = 0

    for indx, p in enumerate(points):
        # ARMS
        # 0 ~ 8 - left
        # 0 ~ 5 - RIGHT
        # p[0] horizontal
        # p[1] vertical

        # HEAD
        if indx == 0:
            head_V = p[1]

        # shoulder RIGHT
        elif indx == 2:
            shoulderright_H = p[0]

        # wrist RIGHT
        elif indx == 4:
            wristright_H = p[0]
            wristright_V = p[1]

        # shoulder left
        elif indx == 5:
            shoulderleft_H = p[0]

        # wrist left
        elif indx == 7:
            wristleft_H = p[0]
            wristleft_V = p[1]

    # as lower the height, lower the position is
    # as higher the height, higher the position is
    
    # VERITICAL COMPARING:
    if wristleft_V and wristright_V < head_V:
        
        # HORIZONTAL COMPARING:
        if (wristleft_H <= shoulderleft_H) and (wristright_H >= shoulderright_H):

            return True
    else:
        return False


def verify_arms_DOWN(points):
    # Armleft
    # position horizontal
    wristleft_H = 0
    # position veritical
    shoulderleft_H = 0
    shoulderleft_V = 0
    wristleft_V = 0

    # ArmRIGHT
    # position horizontal
    # position veritical
    wristright_V = 0
    shoulderright_H = 0

    for indx, p in enumerate(points):
        # shoulder RIGHT
        if indx == 2:
            shoulderright_H = p[0]
            shoulderright_V = p[1]

        # wrist RIGHT
        if indx == 4:
            wristright_V = p[1]

        # shoulder left
        if indx == 5:
            shoulderleft_H = p[0]
            shoulderleft_V = p[1]

        # wrist left
        if indx == 7:
            wristleft_V = p[1]
            wristleft_H = p[0]

    if (wristleft_V >= shoulderleft_V) and (wristright_V >= shoulderright_V):
        if (wristleft_H <= shoulderleft_H) and (wristleft_H >= shoulderright_H):
            print('initial position')
            return True
    else:
        return False


def verify_legs_OPEN(points):
    # legs
    # 11 ~ 14 left
    # 8 ~ 11 right
    ankleleft_H = 0
    ankleright_H = 0
    hipleft_H = 0
    hipright_H = 0

    for indx, p in enumerate(points):
        # print(indx, p)
        # p[0] horizontal
        # p[1] vertical
        
        if indx == 0:
            hipright_H = p[0]
        if indx == 2:
            ankleright_H = p[0]
        if indx == 3:
            hipleft_H = p[0]
        if indx == 5:
            ankleleft_H = p[0]
            
    if (ankleright_H < hipright_H) and (ankleleft_H > hipleft_H):
        return True
    else:
        return False


def verify_legs_CLOSE(points):
    # legs
    # 11 ~ 14 left
    # 8 ~ 11  right
    ankleleft_H = 0
    ankleright_H = 0
    hipleft_H = 0
    hipright_H = 0

    for indx, p in enumerate(points):
        # p[0] horizontal
        # p[1] vertical
        if indx == 0:
            hipright_H = p[0]
        if indx == 2:
            ankleright_H = p[0]
        if indx == 3:
            hipleft_H = p[0]
        if indx == 5:
            ankleleft_H = p[0]
            
    if (ankleright_H >= hipright_H) and (ankleleft_H <= hipleft_H):
        return True
    else:
        return False
