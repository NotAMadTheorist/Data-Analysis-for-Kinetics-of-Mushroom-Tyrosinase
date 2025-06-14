from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
path_KineticsData = ROOT_DIR / "Instrument Output Files" / "Tyrosinase Kinetics Data, Group 3.csv"
path_AbsorbanceTimeTable = ROOT_DIR / "Program Output Files" / "Tyrosinase Kinetics, Absorbance-Time Table.csv"
path_MichaelisMentenParams = ROOT_DIR / "Program Output Files" / "Tyrosinase Kinetics, Michaelis-Menten Parameters.csv"
path_Models_and_ReactionRates = ROOT_DIR / "Program Output Files" / ("Tyrosinase Kinetics, Summarized Model and "
                                                                     "Reaction Rates.csv")
path_TTest_for_LineweaverBurk = ROOT_DIR / "Program Output Files" / ("Tyrosinase Kinetics, T-Test for Slopes and "
                                                                     "Intercepts in Lineweaver-Burk Plot")
