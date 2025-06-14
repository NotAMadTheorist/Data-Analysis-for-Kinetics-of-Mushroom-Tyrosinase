import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from math import exp, sqrt
from Source_MTKinetics.rootDir import *

# ---------------------------------------------------
# Process the data into a numpy or Pandas data table
# ---------------------------------------------------
TyrKineticsDF = pd.read_csv(path_KineticsData)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

sampleTrialsSize = len(TyrKineticsDF.index)
groupedTrialsSize = sampleTrialsSize // 3
rowTrialsSize = groupedTrialsSize // 3
colTrialsSize = 9


# ---------------------------------------------------
# Divide the table into trio of trials
# ---------------------------------------------------
TyrosinaseOnly_KineticsDF_List = []
pHBA_and_Tyrosinase_KineticsDF_List = []
Vinegar_and_Tyrosinase_KineticsDF_List = []
DividedKineticsDF_List = [TyrosinaseOnly_KineticsDF_List,
                          pHBA_and_Tyrosinase_KineticsDF_List,
                          Vinegar_and_Tyrosinase_KineticsDF_List]

for rowIndex in range(rowTrialsSize):
    Index1 = rowIndex*colTrialsSize
    Index2 = Index1 + colTrialsSize
    rowKineticsDF = TyrKineticsDF.iloc[Index1:Index2, 0:]
    row_TyrosinaseOnlyDF = rowKineticsDF.iloc[0:3, 0:]
    row_pHBA_and_TyrosinaseDF = rowKineticsDF.iloc[3:6, 0:]
    row_Vinegar_and_TyrosinaseDF = rowKineticsDF.iloc[6:9, 0:]

    TyrosinaseOnly_KineticsDF_List.append(row_TyrosinaseOnlyDF)
    pHBA_and_Tyrosinase_KineticsDF_List.append(row_pHBA_and_TyrosinaseDF)
    Vinegar_and_Tyrosinase_KineticsDF_List.append(row_Vinegar_and_TyrosinaseDF)

# ---------------------------------------------------
# Prepare Data for Absorbance vs Time Curves for Each Trio
# ---------------------------------------------------
TyrosinaseOnly_KineticsDF_DataPoints = []
pHBA_and_Tyrosinase_KineticsDF_DataPoints = []
Vinegar_and_Tyrosinase_KineticsDF_DataPoints = []
DividedKinetics_DF_DataPoints = [TyrosinaseOnly_KineticsDF_DataPoints,
                                 pHBA_and_Tyrosinase_KineticsDF_DataPoints,
                                 Vinegar_and_Tyrosinase_KineticsDF_DataPoints]


# ---------------------------------------------------
# Fill in the average data points for each trio of trials
# ---------------------------------------------------
UnshiftedTimeValues = [float(x[11:]) for x in list(TyrKineticsDF.columns)[12:]]
loop_index_k = 0
for OriginalDF, DFDataPoints in zip(DividedKineticsDF_List, DividedKinetics_DF_DataPoints):
    loop_index_i = 0
    for trio_ConditionDF in OriginalDF:
        # Shifted Time Values
        Shift = trio_ConditionDF.iat[0,7]
        ShiftedTimeValues = [t + Shift for t in UnshiftedTimeValues]

        # Averaged Absorbance Readings
        AbsorbanceTrial1 = list(trio_ConditionDF.iloc[0, 12:])
        AbsorbanceTrial2 = list(trio_ConditionDF.iloc[1, 12:])
        AbsorbanceTrial3 = list(trio_ConditionDF.iloc[2, 12:])
        AbsorbanceAverage = [(A1 + A2 + A3) / 3 for A1, A2, A3 in
                             zip(AbsorbanceTrial1, AbsorbanceTrial2, AbsorbanceTrial3)]

        # Final Concentration of Benzoquinone in mM
        mM_Benzoquinone_ConcValue = trio_ConditionDF.iat[0, 11]
        mM_Benzoquinone_Conc = [mM_Benzoquinone_ConcValue for C in range(len(ShiftedTimeValues))]

        # Construct the Data Frame for DFDataPoints
        NewDF = pd.DataFrame({'Final Conc. Benzoquinone (mM)': mM_Benzoquinone_Conc,
                 'Shifted Time Values (s)': ShiftedTimeValues,
                 'Averaged Absorbance (A.U.)':AbsorbanceAverage})
        DFDataPoints.append(NewDF)

# ---------------------------------------------------
# Data Points to Exclude (Manually performed based on initial scatter plots)
# ---------------------------------------------------
ExclusionMatrix = [[4, 3, 5, 2, 2, 3],
                   [0, 0, 0, 5, 5, 0],
                   [0, 0, 0, 0, 1, 4]]

# ---------------------------------------------------
# Trim the Data for Each Data Frame
# ---------------------------------------------------
loop_index_k = 0
for DFDataPoints in DividedKinetics_DF_DataPoints:
    loop_index_i = 0
    New_DFDataPoints = []
    for averagedCurve_DataPoints in DFDataPoints:
        number_DPs_excluded = ExclusionMatrix[loop_index_k][loop_index_i]
        DFDataPoints[loop_index_i] = averagedCurve_DataPoints.iloc[number_DPs_excluded:]
        loop_index_i += 1
    DFDataPoints = New_DFDataPoints
    loop_index_k += 1


# ---------------------------------------------------
# Get the exponential regression parameters for each absorbance vs time curve
# Equation:  Y = A- B*exp(-X/C)
# Derivative: dY/dX = (B/C)*exp(-X/C)
# ---------------------------------------------------

# Collection for parameters of the exponential model including the R2 value
# [A, B, C, R^2]
TyrosinaseOnly_ExpoParameters = []
pHBA_and_Tyrosinase_ExpoParameters = []
Vinegar_and_Tyrosinase_ExpoParameters= []
Divided_ExpoParameters = [TyrosinaseOnly_ExpoParameters,
                          pHBA_and_Tyrosinase_ExpoParameters,
                          Vinegar_and_Tyrosinase_ExpoParameters]

def ExpoModel(xArray, A, B, C):
    return A - B*np.exp(-xArray/C)
InitialParameters = (1.7, 1.7, 68)  # Manually determined initial parameters near what we expect for the final model

loop_index_k = 0
for DFDataPoints, ListExpoParameters in zip(DividedKinetics_DF_DataPoints, Divided_ExpoParameters):
    loop_index_i = 0
    for averagedCurve_DataPoints in DFDataPoints:
        X_ShiftedTimeValues = np.array(list(averagedCurve_DataPoints['Shifted Time Values (s)']))
        Y_AverageAbsorbance = np.array(list(averagedCurve_DataPoints['Averaged Absorbance (A.U.)']))

        # Determine Parameters A and B
        ExpoParameters, ExpoCV = scipy.optimize.curve_fit(ExpoModel, X_ShiftedTimeValues, Y_AverageAbsorbance, InitialParameters)
        ExpoParameters = list(ExpoParameters)
        ParameterA, ParameterB, ParameterC = ExpoParameters

        # Determine R Squared Value
        SquaredDiffs_Y = np.square(Y_AverageAbsorbance - ExpoModel(X_ShiftedTimeValues, ParameterA, ParameterB, ParameterC))
        SquaredDiffsFromMean_Y = np.square(Y_AverageAbsorbance-np.mean(Y_AverageAbsorbance))
        ParameterR2 = 1 - np.sum(SquaredDiffs_Y) / np.sum(SquaredDiffsFromMean_Y)

        ExpoParameters.append(ParameterR2)
        ListExpoParameters.append(ExpoParameters)

# Scatter Plotting of Trimmed, Modelled Data
loop_index_k = 0
for DFDataPoints in DividedKinetics_DF_DataPoints:
    colors = ["red", "orange", "yellow", "green", "blue", "violet"]
    titles = ["Absorbance Readings for Tyrosinase-Only Samples",
              "Absorbance Readings for Tyrosinase + p-Hydroxybenzoic Acid Samples",
              "Absorbance Readings for Tyrosinase + Commercial Vinegar Samples"]
    loop_index_i = 0
    for averagedCurve_DataPoints in DFDataPoints:
        # Get exponential parameters
        ParameterA, ParameterB, ParameterC, ParameterR2 = Divided_ExpoParameters[loop_index_k][loop_index_i]

        # Scatter Plot of Data Points
        mM_Benzoquinone_ConcValue = averagedCurve_DataPoints['Final Conc. Benzoquinone (mM)'].iloc[0]
        X_ShiftedTimeValues = np.array(averagedCurve_DataPoints['Shifted Time Values (s)'])
        Y_AverageAbsorbance = np.array(averagedCurve_DataPoints['Averaged Absorbance (A.U.)'])
        plt.scatter(X_ShiftedTimeValues, Y_AverageAbsorbance, color=colors[loop_index_i],
                    label=f'{mM_Benzoquinone_ConcValue} mM Catechol, R² = {ParameterR2:.2f}')

        # Plot the Exponential Model
        X_min, X_max = min(X_ShiftedTimeValues), max(X_ShiftedTimeValues)
        NumberOfPoints = 100
        X_ContinuousTime = np.linspace(X_min, X_max, NumberOfPoints)

        plt.plot(X_ContinuousTime, ExpoModel(X_ContinuousTime, ParameterA, ParameterB, ParameterC), '--', color='black')
        loop_index_i += 1

    plt.title(titles[loop_index_k])
    plt.legend(fontsize="small")
    plt.axvline(x = 300, color = 'gray', linestyle = 'dotted')
    plt.xlabel("Time since Addition of Tyrosinase (s)")
    plt.ylabel("Average Absorbance at 410 nm (A.U.)")
    plt.xlim(80, 630)
    plt.show()
    loop_index_k += 1

# ---------------------------------------------------
# Use exponential model to fit absorbance curve estimate absorbance rate:
# Equation:  Y = A - B*exp(-X/C)
# Derivative: dY/dX = (B/C)*exp(-X/C)
# Note that parameter A (max absorbance) is used to convert absorbance readings into reaction rates (mM/s unit)
# Reaction Rate = Absorbance Rate * (Max Concentration / Max Absorbance)
# ---------------------------------------------------

# Collection of Initial Reaction Rates at Common Time t = 300 s
TyrosinaseOnly_ReactionRate = []
pHBA_and_Tyrosinase_ReactionRate = []
Vinegar_and_Tyrosinase_ReactionRate= []
Divided_ReactionRate = [TyrosinaseOnly_ReactionRate,
                        pHBA_and_Tyrosinase_ReactionRate,
                        Vinegar_and_Tyrosinase_ReactionRate]

TyrosinaseOnly_AbsorbanceRate = []
pHBA_and_Tyrosinase_AbsorbanceRate = []
Vinegar_and_Tyrosinase_AbsorbanceRate= []
Divided_AbsorbanceRate = [TyrosinaseOnly_AbsorbanceRate,
                        pHBA_and_Tyrosinase_AbsorbanceRate,
                        Vinegar_and_Tyrosinase_AbsorbanceRate]

# --------------------------------------
timeCommon = 300 # seconds (manually determined; considers shifting of time due to delay in measurement)
# --------------------------------------

loop_index_k = 0
for ListReactionRates, ListExpoParameters, OriginalDF, ListAbsorbanceRates in zip(Divided_ReactionRate, Divided_ExpoParameters, DividedKineticsDF_List, Divided_AbsorbanceRate):
    loop_index_i = 0
    for ExpoParameters, trio_ConditionDF in zip(ListExpoParameters, OriginalDF):
        ParameterA, ParameterB, ParameterC, ParameterR2 = ExpoParameters
        AbsorbanceRate = ParameterB/ParameterC*exp(-timeCommon/ParameterC)
        Max_mM_Benzoquinone_ConcValue = trio_ConditionDF.iat[0, 11]
        ReactionRate = AbsorbanceRate*Max_mM_Benzoquinone_ConcValue/ParameterA
        ListAbsorbanceRates.append(AbsorbanceRate)
        ListReactionRates.append(ReactionRate)
        loop_index_i += 1
    loop_index_k += 1

# ---------------------------------------------------
# Create a shifted time vs average absorbance table with the following:
#  - Cells Included (e.g., A1, A2, A3)
#  - Conditions (type of inhibition, concentration of catechol)
#  - Variable (shifted time or average absorbance)
#  - Values of Variable
# ---------------------------------------------------

# ---------------------------------------------------
# Create a summary table with the following:
#  - Cells Included (e.g., A1, A2, A3)
#  - Conditions (type of inhibition, concentration of catechol)
#  - Exponential Parameters A, B, and C
#  - R Squared Value
#  - Max Concentration of Benzoquinone (in mM)
#  - Max Estimated Absorbance (in A.U.)
#  - Estimated Reaction Rate at {timeCommon} seconds (in mM / s)
#  - Averaged Absorbance Readings
# ---------------------------------------------------

TyrKinetics_TimeAbsTable_TyrOnly = pd.DataFrame()
TyrKinetics_TimeAbsTable_TyrPHBA = pd.DataFrame()
TyrKinetics_TimeAbsTable_TyrVinegar = pd.DataFrame()
TyrKinetics_TimeAbsTableList = [TyrKinetics_TimeAbsTable_TyrOnly,
                            TyrKinetics_TimeAbsTable_TyrPHBA,
                            TyrKinetics_TimeAbsTable_TyrVinegar]
TyrKinetics_TimeAbsTable = pd.DataFrame()

TyrKinetics_Summary_TyrOnly = pd.DataFrame()
TyrKinetics_Summary_TyrPHBA = pd.DataFrame()
TyrKinetics_Summary_TyrVinegar = pd.DataFrame()
TyrKinetics_SummaryList = [TyrKinetics_Summary_TyrOnly,
                           TyrKinetics_Summary_TyrPHBA,
                           TyrKinetics_Summary_TyrVinegar]
TyrKinetics_Summary = pd.DataFrame()

loop_index_k = 0
DividedSummary = zip(DividedKineticsDF_List, DividedKinetics_DF_DataPoints, Divided_ExpoParameters, Divided_AbsorbanceRate, Divided_ReactionRate, TyrKinetics_SummaryList)
for groupedTrio_ConditionDF, groupedDataPointsDF, groupedExpoParameters, groupedAbsorbanceRate, groupedReactionRate, groupedSummaryDF in DividedSummary:
    ListNumberDataCols = []
    NumberDataCols = 0
    timeAbsCols = ['Cells Included', 'Conditions', 'Initial Conc. mM Catechol', 'Variable']
    timeAbsDataCols = []
    SummaryCols = ['Cells Included', 'Conditions', 'Initial Conc. mM Catechol', 'Expo Parameter A',
                   'Expo Parameter B', 'Expo Parameter C', 'R Squared Value', 'Max Conc. mM Benzoquinone',
                   'Max Absorbance', 'Absorbance Rate (A.U/s)', f'Initial Reaction Rate (mM/s) at {timeCommon} s']
    groupSummary = zip(groupedTrio_ConditionDF, groupedDataPointsDF, groupedExpoParameters, groupedAbsorbanceRate, groupedReactionRate)

    trioCellsIncluded = []
    trioConditions = []
    InitialConcCatechol = []
    ExpoParamA = []
    ExpoParamB = []
    ExpoParamC = []
    RSquaredValue = []
    MaxConcBenzoquinone = []
    MaxAbsorbance = []
    AbsorbanceRateCol = []
    InitialReactionRate = []
    VariablesCol = []
    SummaryValueCols = [trioCellsIncluded, trioConditions, InitialConcCatechol, ExpoParamA, ExpoParamB, ExpoParamC,
                        RSquaredValue, MaxConcBenzoquinone, MaxAbsorbance, AbsorbanceRateCol, InitialReactionRate]
    TimeAbsValueCols = [trioCellsIncluded, trioConditions, InitialConcCatechol, VariablesCol]
    TimeValues = []
    AbsValues = []

    loop_index_i = 0
    for trio_ConditionDF, dataPointsDF, ExpoParameters, AbsorbanceRate, ReactionRate in groupSummary:
        # Cells Included Column
        trioCellsIncluded.append(", ".join(list(trio_ConditionDF.iloc[:, 2])))

        # Condition Column
        conditionString = "Tyrosinase" if trio_ConditionDF.iat[0,4] else ""
        conditionString = conditionString + " with " + "p-Hydroxybenzoic Acid" if trio_ConditionDF.iat[0,5] else conditionString
        conditionString = conditionString + " with " + "Commercial Vinegar" if trio_ConditionDF.iat[0,6] else conditionString
        trioConditions.append(conditionString)

        # Initial Catechol Concentration Column
        InitialConcCatechol.append(trio_ConditionDF.iat[0,10])

        # Exponential Parameters A, B, C, R Squared Columns
        ParameterA, ParameterB, ParameterC, ParameterR2 = ExpoParameters
        ExpoParamA.append(ParameterA)
        ExpoParamB.append(ParameterB)
        ExpoParamC.append(ParameterC)
        RSquaredValue.append(ParameterR2)

        # Max Benzoquinone Concentration Column
        MaxConcBenzoquinone.append(trio_ConditionDF.iat[0,11])

        # Max Absorbance Column
        MaxAbsorbance.append(ParameterA)

        # Absorbance Rate Column
        AbsorbanceRateCol.append(AbsorbanceRate)

        # Initial Reaction Rate Column
        InitialReactionRate.append(ReactionRate)
        loop_index_i += 1

        # Time and absorbance columns
        ListNumberDataCols.append(len(dataPointsDF))
        TimeValues.append(list(dataPointsDF["Shifted Time Values (s)"]))
        AbsValues.append(list(dataPointsDF["Averaged Absorbance (A.U.)"]))

    # Create the dataframe for the summary table
    groupedSummaryDict = {}
    for column, values in zip(SummaryCols, SummaryValueCols):
        groupedSummaryDict[column] = values
    groupedSummaryDF = pd.DataFrame(groupedSummaryDict)
    TyrKinetics_SummaryList[loop_index_k] = groupedSummaryDF

    # Adjust the absorbance and time values for the absorbance-time table
    NumberDataCols = max(ListNumberDataCols)
    def padList(aList, length):
        aList.extend([0]*length)
        aList = aList[:length]
        return aList

    TimeValues = [padList(T, NumberDataCols) for T in TimeValues]
    TimeValues = [["" if T==0 else T for T in TT] for TT in TimeValues]
    AbsValues = [padList(A, NumberDataCols) for A in AbsValues]
    AbsValues = [["" if A==0 else A for A in AA] for AA in AbsValues]

    # Further adjustments for absorbance-time table
    VariablesCol1 = ["Absorbance (A.U.)" for i in AbsValues]
    VariablesCol2 = ['Time Since Tyrosinase Addition (s)' for i in TimeValues]
    for Col1, Col2 in zip(VariablesCol1, VariablesCol2):
        VariablesCol.append(Col1)
        VariablesCol.append(Col2)
    TimeValues = np.array(TimeValues).T.tolist()
    AbsValues = np.array(AbsValues).T.tolist()
    timeAbsCols = timeAbsCols + [f"Data-{x+1}" for x in range(NumberDataCols)]
    DataValues = []
    for Col1, Col2 in zip(AbsValues, TimeValues):
        TempDataValues = []
        for AbsEntry, TimeEntry in zip(Col1, Col2):
            TempDataValues.append(AbsEntry)
            TempDataValues.append(TimeEntry)
        DataValues.append(TempDataValues)
    for DV in DataValues:
        TimeAbsValueCols.append(DV)

    TempTrioCellsIncluded = []
    TempTrioConditions = []
    TempInitialConcCatechol = []
    for Cells, Conditions, Conc in zip(trioCellsIncluded, trioConditions, InitialConcCatechol):
        TempTrioCellsIncluded.append(Cells)
        TempTrioCellsIncluded.append(Cells)
        TempTrioConditions.append(Conditions)
        TempTrioConditions.append(Conditions)
        TempInitialConcCatechol.append(Conc)
        TempInitialConcCatechol.append(Conc)
    TimeAbsValueCols[0] = TempTrioCellsIncluded
    TimeAbsValueCols[1] = TempTrioConditions
    TimeAbsValueCols[2] = TempInitialConcCatechol

    # Construct table with absorbance; alternate rows of absorbance and time
    groupedAbsorbanceDict = {}
    for column, values in zip(timeAbsCols, TimeAbsValueCols):
        groupedAbsorbanceDict[column] = values
    groupedAbsorbanceDF = pd.DataFrame(groupedAbsorbanceDict)
    TyrKinetics_TimeAbsTableList[loop_index_k] = groupedAbsorbanceDF

    loop_index_k += 1

# Finalize data frame for summary
TyrKinetics_Summary_TyrOnly, TyrKinetics_Summary_TyrPHBA, TyrKinetics_Summary_TyrVinegar = TyrKinetics_SummaryList
TyrKinetics_Summary = pd.concat(TyrKinetics_SummaryList, ignore_index=True)

# Finalize data frame for absorbance-time plot
NoRows = TyrKinetics_TimeAbsTableList[0].shape[0]
targetNoColumns = max([DF.shape[1] for DF in TyrKinetics_TimeAbsTableList])
EmptyCol = ['' for i in range(NoRows)]
for DF in TyrKinetics_TimeAbsTableList:
    lastColumnName = DF.columns[-1]
    NoColumns = DF.shape[1]
    DataIndex = int(lastColumnName[5:])
    while NoColumns != targetNoColumns:
        DataIndex += 1
        DF.insert(NoColumns, f"Data-{DataIndex}", EmptyCol, True)
        NoColumns += 1

TyrKinetics_TimeAbsTable = pd.concat(TyrKinetics_TimeAbsTableList, ignore_index=True)
print(TyrKinetics_TimeAbsTable)
print(TyrKinetics_Summary)

# Convert Time-Absorbance and Summary Files into CSV Format
TyrKinetics_TimeAbsTable.to_csv(path_AbsorbanceTimeTable, index=False)
TyrKinetics_Summary.to_csv(path_Models_and_ReactionRates, index=False)

# ---------------------------------------------------
# Determination of Michaelis-Menten Parameters: VMax and K
# and Summary Table for Michaelis-Menten Parameters
# ---------------------------------------------------

# Construct DF which summarizes all three cases
TyrMichaelisMenten_SummaryDict = {"Cells Included":[],
                                  "Conditions":[],
                                  "V-C Expo Parameter A":[],
                                  "V-C Expo Parameter B":[],
                                  "V-C Expo Parameter C":[],
                                  "V-C R Squared Value":[],
                                  "V⁻¹-C⁻¹ Intercept A":[],
                                  "V⁻¹-C⁻¹ Slope B":[],
                                  "V⁻¹-C⁻¹ R Squared Value":[],
                                  "V⁻¹-C⁻¹ Residual Standard Error":[],
                                  "V⁻¹-C⁻¹ Std. Error of Slope":[],
                                  "V⁻¹-C⁻¹ Std. Error of Intercept":[],
                                  "Maximum Reaction Velocity V (mM/s)":[],
                                  "Uncertainty in V (mM/s)":[],
                                  "Michaelis-Menten Constant Km (mM)":[],
                                  "Uncertainty in Km (mM)":[],
                                  "Remarks":[]}
InitialParameters = (0, -1.2E-5, -0.8)  # Manually determined initial parameters near what we expect for the final model
loop_index_i = 0
for SummaryDF in TyrKinetics_SummaryList:
    # Cells included and conditions
    cellsIncluded = ", ".join(list(SummaryDF.iloc[:, 0]))
    labelConditions = SummaryDF.iat[0, 1]
    TyrMichaelisMenten_SummaryDict["Cells Included"].append(cellsIncluded)
    TyrMichaelisMenten_SummaryDict["Conditions"].append(labelConditions)

    # Calculate exponential parameters A, B, and C for exponential model between V and C
    # Equation:  Y = A- B*exp(-X/C)
    X_initConcCatechol = np.array(SummaryDF['Initial Conc. mM Catechol'])
    Y_initReactionRate = np.array(SummaryDF.iloc[:,-1])
    ExpoParameters, ExpoCV = scipy.optimize.curve_fit(ExpoModel, X_initConcCatechol, Y_initReactionRate,
                                                      InitialParameters)
    ParameterA, ParameterB, ParameterC = ExpoParameters
    TyrMichaelisMenten_SummaryDict["V-C Expo Parameter A"].append(ParameterA)
    TyrMichaelisMenten_SummaryDict["V-C Expo Parameter B"].append(ParameterB)
    TyrMichaelisMenten_SummaryDict["V-C Expo Parameter C"].append(ParameterC)

    # Determine R Squared Value
    SquaredDiffs_Y = np.square(Y_initReactionRate - ExpoModel(X_initConcCatechol, ParameterA, ParameterB, ParameterC))
    SquaredDiffsFromMean_Y = np.square(Y_initReactionRate - np.mean(Y_initReactionRate))
    ParameterR2 = 1 - np.sum(SquaredDiffs_Y) / np.sum(SquaredDiffsFromMean_Y)
    TyrMichaelisMenten_SummaryDict["V-C R Squared Value"].append(ParameterR2)

    # Exclude the blank solutions
    ExcludedConc = []
    ExcludedConc.append(str(float(X_initConcCatechol[-1])))
    X_initConcCatechol = np.delete(X_initConcCatechol, -1)
    Y_initReactionRate = np.delete(Y_initReactionRate, -1)

    # For the first condition, Tyrosinase-only, and the second condition, Tyrosinase with p-Hydroxybenzoic acid,
    # exclude the last solution (negative reaction rate). Exclude another solution for the second condition.
    # Exclude another solution for the first condition.
    if loop_index_i == 0 or loop_index_i == 1:
        ExcludedConc.append(str(float(X_initConcCatechol[-1])))
        X_initConcCatechol = np.delete(X_initConcCatechol, -1)
        Y_initReactionRate = np.delete(Y_initReactionRate, -1)
    if loop_index_i == 0:
        ExcludedConc.append(str(float(X_initConcCatechol[-2])))
        X_initConcCatechol = np.delete(X_initConcCatechol, -2)
        Y_initReactionRate = np.delete(Y_initReactionRate, -2)
    if loop_index_i == 1:
        ExcludedConc.append(str(float(X_initConcCatechol[-1])))
        X_initConcCatechol = np.delete(X_initConcCatechol, -1)
        Y_initReactionRate = np.delete(Y_initReactionRate, -1)


    # Add remarks regarding outliers
    print(ExcludedConc)
    if len(ExcludedConc) != 0:
        if len(ExcludedConc) > 1:
            remarkString = f"For the Lineweaver-Burk or V⁻¹-C⁻¹ Plot, outliers at the following concentrations were excluded: {', '.join(ExcludedConc)} mM"
        else:
            remarkString = f"For the Lineweaver-Burk or V⁻¹-C⁻¹ Plot, outliers at the following concentrations were excluded: {ExcludedConc[0]} mM"
        TyrMichaelisMenten_SummaryDict["Remarks"].append(remarkString)

    # Calculate slope, intercept, and R Squared for linear model between 1/V and 1/C
    X_recip_initConcCatechol = np.power(X_initConcCatechol, -1)
    Y_recip_initReactionRate = np.power(Y_initReactionRate, -1)
    VCSlope_B, VCIntercept_A, VC_RValue, VC_PValue, VC_StdErrorOfSlope_BExtra, = scipy.stats.linregress(X_recip_initConcCatechol, Y_recip_initReactionRate)
    TyrMichaelisMenten_SummaryDict["V⁻¹-C⁻¹ Intercept A"].append(VCIntercept_A)
    TyrMichaelisMenten_SummaryDict["V⁻¹-C⁻¹ Slope B"].append(VCSlope_B)
    TyrMichaelisMenten_SummaryDict["V⁻¹-C⁻¹ R Squared Value"].append(VC_RValue**2)

    # Calculate the three standard error values
    Y_Predicted = VCIntercept_A + VCSlope_B*X_recip_initConcCatechol
    countN = len(X_recip_initConcCatechol)
    VC_ResStdError = sqrt(np.sum(np.square(Y_recip_initReactionRate-Y_Predicted))/(countN-2))
    meanX = np.mean(X_recip_initConcCatechol)
    VC_StdErrorOfSlope_B = VC_ResStdError/sqrt(np.sum(np.square(X_recip_initConcCatechol-meanX)))
    VC_StdErrorOfIntercept_A = VC_ResStdError*sqrt(np.sum(np.square(X_recip_initConcCatechol))/(countN*np.sum(np.square(X_recip_initConcCatechol-meanX))))
    TyrMichaelisMenten_SummaryDict["V⁻¹-C⁻¹ Residual Standard Error"].append(VC_ResStdError)
    TyrMichaelisMenten_SummaryDict["V⁻¹-C⁻¹ Std. Error of Slope"].append(VC_StdErrorOfSlope_B)
    TyrMichaelisMenten_SummaryDict["V⁻¹-C⁻¹ Std. Error of Intercept"].append(VC_StdErrorOfIntercept_A)

    # Estimate maximum reaction velocity v and Michaelis-Menten Constant Km
    MaxRxnVel_V = abs(1 / VCIntercept_A)
    MMConst_Km = VCSlope_B*MaxRxnVel_V
    Uncertainty_V = VC_StdErrorOfIntercept_A / VCIntercept_A**2
    Uncertainty_Km = MMConst_Km*sqrt((VC_StdErrorOfSlope_B/VCSlope_B)**2 + (Uncertainty_V/MaxRxnVel_V)**2)
    TyrMichaelisMenten_SummaryDict["Maximum Reaction Velocity V (mM/s)"].append(MaxRxnVel_V)
    TyrMichaelisMenten_SummaryDict["Uncertainty in V (mM/s)"].append(Uncertainty_V)
    TyrMichaelisMenten_SummaryDict["Michaelis-Menten Constant Km (mM)"].append(MMConst_Km)
    TyrMichaelisMenten_SummaryDict["Uncertainty in Km (mM)"].append(Uncertainty_Km)

    loop_index_i += 1

TyrMichaelisMenten_Summary = pd.DataFrame(TyrMichaelisMenten_SummaryDict)
print(TyrMichaelisMenten_Summary)

# Save summary of Michaelis-Menten parameters as a CSV File
TyrMichaelisMenten_Summary.to_csv(path_MichaelisMentenParams, index=False)

# Create a scatter plot between reaction rate "v" versus concentration "C" of reactant catechol
loop_index_i = 0
colors = ["blue", "orange", "yellow"]

for SummaryDF in TyrKinetics_SummaryList:
    labelConditions = SummaryDF.iat[0, 1]
    X_initConcCatechol = np.array(SummaryDF['Initial Conc. mM Catechol'])
    Y_initReactionRate = np.array(SummaryDF.iloc[:,-1])

    Xmin, Xmax = np.min(X_initConcCatechol), np.max(X_initConcCatechol)
    NumberOfPoints = 100
    X_ContinuousConc = np.linspace(Xmin, Xmax, NumberOfPoints)

    MM_SummaryRow = TyrMichaelisMenten_Summary.iloc[loop_index_i, :]
    ParameterA = MM_SummaryRow["V-C Expo Parameter A"]
    ParameterB = MM_SummaryRow["V-C Expo Parameter B"]
    ParameterC = MM_SummaryRow["V-C Expo Parameter C"]
    ParameterR2 = MM_SummaryRow["V-C R Squared Value"]

    # Plotting
    plt.scatter(X_initConcCatechol, Y_initReactionRate*(10**4), color=colors[loop_index_i], label = f"{labelConditions}, R² = {ParameterR2:.2f}")
    plt.plot(X_ContinuousConc, ExpoModel(X_ContinuousConc, ParameterA, ParameterB, ParameterC)*(10**4), '--', color='black')
    loop_index_i += 1
plt.title("Plot of Reaction Rate and Initial Concentration of Catechol")
plt.legend()
plt.xlabel("Initial Concentration of Catechol (mM)")
plt.ylabel(f"Reaction Rate at {timeCommon} s (×10⁻⁴ mM/s)")
plt.show()

# Create a scatter plot between 1/V and 1/C (Lineweaver-Burk Plot)
loop_index_i = 0
colors = ["blue", "orange", "yellow"]

for SummaryDF in TyrKinetics_SummaryList:
    labelConditions = SummaryDF.iat[0, 1]
    X_initConcCatechol = np.array(SummaryDF['Initial Conc. mM Catechol'])
    Y_initReactionRate = np.array(SummaryDF.iloc[:,-1])

    # Exclude the blank solutions
    X_initConcCatechol = np.delete(X_initConcCatechol, -1)
    Y_initReactionRate = np.delete(Y_initReactionRate, -1)

    # For the first condition, Tyrosinase-only, and the second condition, Tyrosinase with p-Hydroxybenzoic acid,
    # exclude the last solution (negative reaction rate). Exclude another solution for the second condition.
    # Exclude another solution for the first condition.
    if loop_index_i == 0 or loop_index_i == 1:
        X_initConcCatechol = np.delete(X_initConcCatechol, -1)
        Y_initReactionRate = np.delete(Y_initReactionRate, -1)
    if loop_index_i == 0:
        X_initConcCatechol = np.delete(X_initConcCatechol, -2)
        Y_initReactionRate = np.delete(Y_initReactionRate, -2)
    if loop_index_i == 1:
        X_initConcCatechol = np.delete(X_initConcCatechol, -1)
        Y_initReactionRate = np.delete(Y_initReactionRate, -1)

    # Take reciprocal of both variables
    X_recip_initConcCatechol = np.power(X_initConcCatechol, -1)
    Y_recip_initReactionRate = np.power(Y_initReactionRate, -1)

    Xmin, Xmax = np.min(X_recip_initConcCatechol), np.max(X_recip_initConcCatechol)
    NumberOfPoints = 100
    X_ContinuousConc = np.linspace(Xmin, Xmax, NumberOfPoints)

    MM_SummaryRow = TyrMichaelisMenten_Summary.iloc[loop_index_i, :]
    VCSlope_B = MM_SummaryRow["V⁻¹-C⁻¹ Slope B"]
    VCIntercept_A = MM_SummaryRow["V⁻¹-C⁻¹ Intercept A"]
    VC_R2Value = MM_SummaryRow["V⁻¹-C⁻¹ R Squared Value"]

    # Plotting
    plt.scatter(X_recip_initConcCatechol, (Y_recip_initReactionRate)/10**4, color=colors[loop_index_i], label = f"{labelConditions}, R² = {VC_R2Value:.2f}")
    plt.plot(X_ContinuousConc, (VCIntercept_A + VCSlope_B*X_ContinuousConc)/10**4, '--', color='black')
    loop_index_i += 1

plt.title("Lineweaver-Burk Plot for Mushroom Tyrosinase")
plt.legend()
plt.xlabel("Reciprocal of Initial Concentration of Catechol (mM ⁻¹)")
plt.ylabel(f"Reciprocal of Reaction Rate at {timeCommon} (×10⁴ s/mM)")
plt.show()

# ---------------------------------------------------
# Additional Statistical t-Test
# ---------------------------------------------------

def twoSamplesT(mean1, sdev1, n1, mean2, sdev2, n2, df):  # two-sided test
    tStat = abs(mean1-mean2) / sqrt((sdev1**2)/n1 + (sdev2**2)/n2)
    pValue = 2*(1-scipy.stats.t.cdf(abs(tStat), df))
    return tStat, pValue

Pairings =[(0, 1),
           (1, 2),
           (2, 0)]
SampleSizes = [3, 3, 5] # Considers outliers that were excluded
TyrT_TestSummaryDict = {"Condition 1":[],
                        "Condition 2":[],
                        "t Value for Slopes":[],
                        "p Value for Slopes":[],
                        "Significantly different slopes?":[],
                        "t Value for Intercepts":[],
                        "p Value for Intercepts":[],
                        "Significantly different intercepts?":[]}

for indexPairA, indexPairB in Pairings:
    RowA = TyrMichaelisMenten_Summary.iloc[indexPairA, :]
    RowB = TyrMichaelisMenten_Summary.iloc[indexPairB, :]
    TyrT_TestSummaryDict["Condition 1"].append(RowA["Conditions"])
    TyrT_TestSummaryDict["Condition 2"].append(RowB["Conditions"])

    Slope_MeanA = RowA["V⁻¹-C⁻¹ Slope B"]
    Slope_MeanB = RowB["V⁻¹-C⁻¹ Slope B"]
    Slope_SDevA = RowA["V⁻¹-C⁻¹ Std. Error of Slope"]
    Slope_SDevB = RowB["V⁻¹-C⁻¹ Std. Error of Slope"]
    Intercept_MeanA = RowA["V⁻¹-C⁻¹ Intercept A"]
    Intercept_MeanB = RowB["V⁻¹-C⁻¹ Intercept A"]
    Intercept_SDevA = RowA["V⁻¹-C⁻¹ Std. Error of Intercept"]
    Intercept_SDevB = RowB["V⁻¹-C⁻¹ Std. Error of Intercept"]
    Count_A = SampleSizes[indexPairA]
    Count_B = SampleSizes[indexPairB]
    DF = Count_A + Count_B - 2

    Slope_t, Slope_p = twoSamplesT(Slope_MeanA, Slope_SDevA, Count_A, Slope_MeanB, Slope_SDevB, Count_B, DF)
    Intercept_t, Intercept_p = twoSamplesT(Intercept_MeanA, Intercept_SDevA, Count_A, Intercept_MeanB, Intercept_SDevB, Count_B, DF)
    SigDiffSlope = Slope_p < 0.05
    SigDiffIntercept = Intercept_p < 0.05

    TyrT_TestSummaryDict["t Value for Slopes"].append(Slope_t)
    TyrT_TestSummaryDict["p Value for Slopes"].append(Slope_p)
    TyrT_TestSummaryDict["Significantly different slopes?"].append("Yes" if SigDiffSlope else "No")
    TyrT_TestSummaryDict["t Value for Intercepts"].append(Intercept_t)
    TyrT_TestSummaryDict["p Value for Intercepts"].append(Intercept_p)
    TyrT_TestSummaryDict["Significantly different intercepts?"].append("Yes" if SigDiffIntercept else "No")
TyrT_TestSummary = pd.DataFrame(TyrT_TestSummaryDict)
print(TyrT_TestSummary)
TyrT_TestSummary.to_csv(path_TTest_for_LineweaverBurk)
#  - Significantly Different Slope and/or Intercept? (from Tyrosinase-only case)
#  - Type of Inhibition Involved

