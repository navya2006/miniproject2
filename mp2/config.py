"""Constants shared by every stage: variable groups, cleaning ranges, model formulas, labels."""

SEED = 20261019

# ---- variable groups ------------------------------------------------------------------------
BIN = ["nonprev", "female", "married", "metro", "norcen", "south", "west", "itreq", "itprog", "itqual",
       "itarch", "itai", "itclient", "itrd", "ittech", "wrkhm"]
REGION = ["norcen", "south", "west"]
FUNC = ["itreq", "itprog", "itqual", "itarch", "itai", "itclient"]
DIV = ["itrd", "ittech"]
FAM = ["momed", "daded", "sib"]
CORE_VARS = (["sal", "edu", "exper", "ten", "nonprev", "female", "married", "dep", "metro"]
             + REGION + FUNC + DIV + ["wrkhm"])

# ---- Rule 2: impossible values (lo, hi, reason) ------------------------------------------------
RANGE = {
    "sal":   (0.0001, 100, "salary must be positive; all other values <= 60"),
    "edu":   (0, 25,  "years of education; all other values <= 18"),
    "exper": (0, 60,  "years of experience"),
    "ten":   (0, 60,  "years with company"),
    "dep":   (0, 15,  "# dependents"),
    "momed": (0, 25,  "mother's education; all other values <= 15"),
    "daded": (0, 25,  "father's education; all other values <= 19"),
    "sib":   (0, 10,  "# siblings; all other values <= 5"),
}

# ---- model formulas ----------------------------------------------------------------------------
HC = "edu + exper + exper2 + ten + ten2"
PERS = "female + nonprev + married + dep"
LOC = "metro + norcen + south + west"
JOB = "itreq + itprog + itqual + itarch + itai + itclient + itrd + ittech + wrkhm"
FORM = {
    "M1": f"lsal ~ {HC}",
    "M2": f"lsal ~ {HC} + {PERS}",
    "M3": f"lsal ~ {HC} + {PERS} + {LOC}",
    "M4": f"lsal ~ {HC} + {PERS} + {LOC} + {JOB}",
    "M5": f"lsal ~ {HC} + {PERS} + {LOC} + {JOB} + edu2",
    "M6": f"lsal ~ {HC} + {PERS} + {LOC} + {JOB} + edu2 + female:married",
}
MODEL_NAMES = {"M1": "Human capital", "M2": "+ Personal", "M3": "+ Location", "M4": "+ Job assignment",
               "M5": "+ Education squared", "M6": "+ Female x Married"}

LABEL = {"Intercept": "Intercept", "edu": "Education (years)", "edu2": "Education squared", "exper": "Experience (years)",
         "exper2": "Experience squared", "ten": "Tenure (years)", "ten2": "Tenure squared", "female": "Female",
         "nonprev": "Non-privileged", "married": "Married", "dep": "# dependents", "metro": "Metro area",
         "norcen": "North-central", "south": "South", "west": "West", "itreq": "IT requirements",
         "itprog": "IT programming", "itqual": "IT quality assurance", "itarch": "IT architecture",
         "itai": "IT AI applications", "itclient": "IT client-only", "itrd": "IT R&D division",
         "ittech": "IT tech support", "wrkhm": "Works from home", "female:married": "Female x Married",
         "momed": "Mother's education", "daded": "Father's education", "sib": "# siblings"}


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---- interaction candidates ---------------------------------------------------------------------
QUAD_TERMS = ["edu2", "dep2"]
INTER = {"female:edu": "Gender differences in returns to education",
         "female:exper": "Gender differences in returns to experience",
         "female:ten": "Gender differences in returns to tenure",
         "female:married": "Marriage premium differs by gender",
         "female:dep": "Dependents penalty differs by gender",
         "edu:exper": "Education-experience complementarity",
         "edu:ten": "Education-tenure complementarity",
         "metro:edu": "Urban premium to education",
         "itai:edu": "AI work x education",
         "itrd:edu": "R&D division x education",
         "nonprev:edu": "Returns to education by privilege status",
         "wrkhm:female": "Working from home x gender"}

# ---- terms tracked in the robustness / sensitivity tables ---------------------------------------
KEYT = ["edu", "edu2", "exper", "exper2", "ten", "ten2", "female", "married", "female:married", "nonprev",
        "metro", "itarch", "itai", "itrd", "wrkhm"]
