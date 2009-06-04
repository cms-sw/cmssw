import os, re

#Sort the candles to make sure MinBias is executed before QCD_80_120, otherwise DIGI PILEUP would not find its MinBias root files

#Define ALL the physics processes available to the Performance Suite user (for most uses only 2 or 5 of these at a time will be used):
Candles=["MinBias"            ,
         "HiggsZZ4LM200"      ,         
         "SingleElectronE1000",
         "SingleMuMinusPt10"  ,
         "SinglePiMinusE1000" ,
         "TTbar"              ,
         "QCD_80_120",
         #Example addition of a new candle:
         "QCD_3000_3500"
         ]
#List containing the root filenames hardcoded in cmsDriver.py
#In the future we could eliminate this dependency by forcing the filename convention in the suite since we use the --fileout cmsDriver.py option now.
CandFname={
    Candles[0]:"MINBIAS_",
    Candles[1]:"HZZLLLL_200",    
    Candles[2]:"E_1000",
    Candles[3]:"MU-_pt10",
    Candles[4]:"PI-_1000",
    Candles[5]:"TTBAR_" ,
    Candles[6]:"QCD_80_120",
    #Example addition of a new candle:
    Candles[7]:"QCD_3000_3500"
    }

#List with all the "individual" steps understood by the Performance Suite
Step = ["GEN,SIM",
        "DIGI",
        "L1",              
        "DIGI2RAW",
        "HLT",
        "RAW2DIGI,RECO",
        #Add also all PILEUP steps
        "DIGI_PILEUP",
        "L1_PILEUP",
        "DIGI2RAW_PILEUP",
        "HLT_PILEUP",
        "RAW2DIGI,RECO_PILEUP"
        ]

#List of Production steps (to be used by the publishing script to find reports:
ProductionSteps = ["GEN,SIM,DIGI,L1,DIGI2RAW,HLT",
                   "GEN,SIM,DIGI,L1,DIGI2RAW",
                   "RAW2DIGI,RECO", #This is already included in Step! So remember to eliminate duplicates if doing the union of the two!
                   #Add also all PILEUP steps
                   "GEN,SIM,DIGI,L1,DIGI2RAW,HLT_PILEUP",
                   "GEN,SIM,DIGI,L1,DIGI2RAW_PILEUP",
                   "RAW2DIGI,RECO_PILEUP", #This is already included in Step!
                   "GEN,FASTSIM", #Adding FASTSIM workflow
                   "HLT"          #Adding HLT alone workflow
                   ]
#A dictionary with the reverse look-up for the candle given the root base filename 
revCFname = {
    "MINBIAS_"    : Candles[0],
    "HZZLLLL_200" : Candles[1],    
    "E_1000"      : Candles[1],
    "MU-_pt10"    : Candles[2],
    "PI-_1000"    : Candles[3],
    "TTBAR_"      : Candles[4], 
    "QCD_80_120"  : Candles[6],
    #Example addition of a new candle:
    "QCD_3000_3500": Candles[7]
    }

CandDesc=["Minimum Bias",
          "Higgs -> ZZ -> 4 leptons",
          "Electron",
          "Muon",
          "Pion",
          "TTBar",
          "QCD Jets 80-120 GeV",
          #Example addition of a new candle:
          "QCD Jets 3000-3500 GeV"
          ]

# Need a little hash to match the candle with the ROOT name used by cmsDriver.py.

FileName = {}
# Hash to switch from keyword to .cfi use of cmsDriver.py:
KeywordToCfi = {}
configs   = ['MinBias.cfi',               
             'H200ZZ4L.cfi',
             'SingleElectronE1000.cfi',
             'SingleMuPt10.cfi',
             'SinglePiE1000.cfi',
             'TTbar_Tauola.cfi',               
             'QCD_Pt_80_120.cfi',
             #Example addition of a new candle:
             'QCD_Pt_3000_3500.cfi'
             ]
    
filenames = [CandFname[Candles[0]],        
             CandFname[Candles[1]],
             CandFname[Candles[2]],
             CandFname[Candles[3]],
             CandFname[Candles[4]],
             CandFname[Candles[5]],
             CandFname[Candles[6]],
             #Example addition of a new candle:
             CandFname[Candles[7]]
             ]
for x in range(len(Candles)):

    KeywordToCfi[Candles[x]] = configs[x]
    FileName[Candles[x]]     = filenames[x]

#Allowed event contents (this list is used at the moment only in cmsRelvalreportInput.py to make sure any unprofiled step uses the FEVTDEBUGHLT eventcontent. Other uses can be devised later (adding FEVTDEBUG and FEVTDEBUGHLT for example)
EventContents=['RAWSIM',
               'RECOSIM'
               ]

#PILE-UP Settings:

#Set here the cmsDriver.py --pileup option:
cmsDriverPileUpOption='LowLumiPileUp'

#Set the customise fragments path for the various steps:
#Note that currently the default customise fragment for steps not defined in this dictionary is the DIGI one below.
#Each step could have its own, by adding it in this dictionary
#When cmsRelvalreport.py is invoked with --pileup in its --cmsdriver option, then the DIGI-PILEUP customise will be
#used for all steps.
CustomiseFragment = {
         'GEN,SIM': 'Validation/Performance/TimeMemoryG4Info.py',
         'DIGI': 'Validation/Performance/TimeMemoryInfo.py',
         'DIGI-PILEUP':'Validation/Performance/MixingModule.py'
         }

def getVerFromLog(previous):
    prevlog = os.path.join(previous,"cmsPerfSuite.log")
    if os.path.exists(prevlog):
        for line in open(prevlog):
            if "Test Release based on:" in line:
                verreg = re.compile("^.*Test Release based on: (.*)$")
                match = verreg.search(line)
                if match:
                    return match.groups()[0]
        
    return "Unknown_prev_release"
