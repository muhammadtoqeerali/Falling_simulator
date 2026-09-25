DATA_AUGMENTATION_SUBJECTS = ["1000", "999"]

FALL_TASKS = [
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    37,
    38,
    39,
    40,
    41,
    42,
]

ACTIVITY_TASKS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    35,
    36,
    43,
    44,
    88
]

BACKWARD_FALLS = [21, 27, 34, 37, 38, 40, 41, 42] # 8
FORWARD_FALLS = [20, 23, 25, 28, 29, 30, 31, 32, 39] # 9
LATERAL_FALLS = [22, 24, 26, 33] # 4
FALLS_FROM_HEIGHT = [39, 40] # 2

TASKS_DESCRIPTIONS = {
    1: "Stand for 30 seconds",
    2: "Stand, slowly bend the back with or without bending at knees, tie shoe lace, and get up",
    3: "Pick up an object from the floor",
    4: "Gently jump (try to reach an object)",
    5: "Stand, sit to the ground, wait a moment, and get up with normal speed",
    6: "Walk normally with turn ",
    7: "Walk quickly with turn",
    8: "Jog normally with turn",
    9: "Jog quickly with turn",
    10: "Stumble with Mattress while walking",
    11: "Sit on a chair for 30 seconds",
    12: "Walk downstairs normally",
    13: "Sit down to a chair normally, and get up from a chair normally",
    14: "Sit down to a chair quickly, and get up from a chair quickly",
    15: "Sit a moment, trying to get up, and collapse into a chair",
    16: "Walk downstairs quickly",
    17: "Lie on the bed for 30 seconds",
    18: "Sit a moment, lie down to the bed normally, and get up normally",
    19: "Sit a moment, lie down to the bed quickly, and get up quickly",
    20: "Forward fall when trying to sit down",
    21: "Backward fall when trying to sit down",
    22: "Lateral fall when trying to sit down",
    23: "Forward fall when trying to get up",
    24: "Lateral fall when trying to get up",
    25: "Forward fall while sitting, caused by fainting",
    26: "Lateral fall while sitting, caused by fainting",
    27: "Backward fall while sitting, caused by fainting",
    28: "Vertical (forward) fall while walking caused by fainting",
    29: "Fall while walking, use of hands to dampen fall, caused by fainting",
    30: "Forward fall while walking caused by a trip",
    31: "Forward fall while jogging caused by a trip",
    32: "Forward fall while walking caused by a slip",
    33: "Lateral fall while walking caused by a slip",
    34: "Backward fall while walking caused by a slip",
    35: "Walk upstairs normally",
    36: "Walk upstairs quickly",
    37: "Backward fall while slowly moving back",
    38: "Backward fall while quickly moving back",
    39: "Forward fall from height",
    40: "Backward fall from height",
    41: "Backward fall while trying to climb up the stairs",
    42: "Backward fall while trying to climb down the stairs",
    43: "Climb up and climb down the stairs",
    44: "Walk slowly and jump over the obstacle",
    88: "On field activities without falling"
}


# Added for simulated dataset Protechto-compatible task remapping.
# Original simulator scenarios:
#   250 -> 45
#   290 -> 46
#   291 -> 47
TASKS_DESCRIPTIONS.update({
    45: "Forward fall while standing, caused by fainting (original simulator scenario 250)",
    46: "Fall backward while walking, hands used to dampen (original simulator scenario 290)",
    47: "Fall lateral while walking, hands used to dampen (original simulator scenario 291)",
})

