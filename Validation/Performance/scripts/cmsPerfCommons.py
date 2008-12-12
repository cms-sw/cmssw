import os, re

#Sort the candles to make sure MinBias is executed before QCD_80_120, otherwise DIGI PILEUP would not find its MinBias root files

Candles=["MinBias"            ,
         "HiggsZZ4LM200"      ,         
         "SingleElectronE1000",
         "SingleMuMinusPt10"  ,
         "SinglePiMinusE1000" ,
         "TTbar"              ,
         "QCD_80_120"         
         ]
#Introducing different sets of "default" candles for different tests (runs of cmsPerfSuite.py):
#CandlesToRun={'Default':[Candles[0],
#                         Candles[1],
#                         Candles[2],
#                         Candles[3],
#                         Candles[4],
#                         Candles[5],
#                         Candles[6]
#                         ],
#             'Simulation':[Candles[0],
#                           Candles[2],
#                           Candles[3],
#                           Candles[4],
#                           Candles[5]
#                           ],
#             'Step1':[Candles[0],
#                      Candles[5],
#                      ],
#             'Step2':[Candles[0],
#                      Candles[5],
#                      ]
#    }
#SimulationCandles=["MinBias"            ,
#                   "SingleElectronE1000",
#                   "SingleMuMinusPt10"  ,
#                   "SinglePiMinusE1000" ,
#                   "TTbar"
#                   ]
#Step1Candles=["MinBias",
#              "TTbar"
#              ]
#Step2Candles=["MinBias",
#              "TTbar"
#              ]
#
CandFname={
    Candles[0]:"MINBIAS_",
    Candles[1]:"HZZLLLL_200",    
    Candles[2]:"E_1000",
    Candles[3]:"MU-_pt10",
    Candles[4]:"PI-_1000",
    Candles[5]:"TTBAR_" ,
    Candles[6]:"QCD_80_120"
    }

#List with all the steps
Step = ["GEN,SIM",
        "DIGI",
        "L1",              
        "DIGI2RAW",
        "HLT",
        "RAW2DIGI,RECO",
        "DIGI_PILEUP",
        "L1_PILEUP",
        "DIGI2RAW_PILEUP",
        "HLT_PILEUP",
        "RAW2DIGI_PILEUP",
        "RECO_PILEUP"
        ]

#Strings with the definition of cmsDriver.py Step1 and Step2
#To be kept up to date
#Step1 = 'GEN,SIM,DIGI,L1,DIGI2RAW,HLT'
#Step2 = 'RAW2DIGI,RECO'

revCFname = {
    "MINBIAS_"    : Candles[0],
    "HZZLLLL_200" : Candles[1],    
    "E_1000"      : Candles[1],
    "MU-_pt10"    : Candles[2],
    "PI-_1000"    : Candles[3],
    "TTBAR_"      : Candles[4], 
    "QCD_80_120"  : Candles[6]
    }

CandDesc=["Minimum Bias",
          "Higgs -> ZZ -> 4 leptons",
          "Electron",
          "Muon",
          "Pion",
          "TTBar",
          "QCD Jets"
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
             'TTbar.cfi',               
             'QCD_Pt_80_120.cfi'
             ]
    
filenames = [CandFname[Candles[0]],        
             CandFname[Candles[1]],
             CandFname[Candles[2]],
             CandFname[Candles[3]],
             CandFname[Candles[4]],
             CandFname[Candles[5]],
             CandFname[Candles[6]]
             ]
for x in range(len(Candles)):

    KeywordToCfi[Candles[x]] = configs[x]
    FileName[Candles[x]]     = filenames[x]

#Adding IgProf, Callgrind, MemCheck dictionaries:

#Adding also the TimeSize one for now, even if it may never be used...
#RunTimeSize={'Default':{Candles[0]:True,
#                        Candles[1]:True,
#                        Candles[2]:True,
#                        Candles[3]:True,
#                        Candles[4]:True,
#                        Candles[5]:True,
#                        Candles[6]:True
#                        },
#             'Simulation':{Candles[0]:True,
#                           Candles[2]:True,
#                           Candles[3]:True,
#                           Candles[4]:True,
#                           Candles[5]:True
#                           },
#             'Step1':{Candles[0]:True,
#                      Candles[5]:True
#                      },
#             'Step2':{Candles[0]:True,
#                      Candles[5]:True
#                      }
#             }
#
#RunIgProf={'Default':{Candles[0]:True,
#                        Candles[1]:True,
#                        Candles[2]:True,
#                        Candles[3]:True,
#                        Candles[4]:True,
#                        Candles[5]:True,
#                        Candles[6]:True
#                        },
#             'Simulation':{Candles[0]:True,
#                           Candles[2]:True,
#                           Candles[3]:True,
#                           Candles[4]:True,
#                           Candles[5]:True
#                           },
#             'Step1':{Candles[0]:True,
#                      Candles[5]:True
#                      },
#             'Step2':{Candles[0]:True,
#                      Candles[5]:True
#                      }
#    }
#
#RunCallgrind={
#    }
#
#RunMemCheck={
#    }

#PILE-UP Settings:

#Minimum number of TimeSizeEvents for Pile Up to run
#Basically, this number if used to prevent DIGI PILE-UP from running when there would be too few MinBias events
#Maybe this should be replaced by some code that would smartly produce the MinBias events needed if they are too few
#in the MinBias_TimeSize directory (check all possible MinBias dirs, _MemCheck and Callgrind too), or they do not exist

MIN_REQ_TS_EVENTS = 20

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

#Adding a dictionary to tell on which candle to run digi pile-up:

#RunDigiPileUp={'Default':{Candles[0]:True,
#                        Candles[1]:True,
#                        Candles[2]:True,
#                        Candles[3]:True,
#                        Candles[4]:True,
#                        Candles[5]:True,
#                        Candles[6]:True
#                        },
#             'Simulation':{Candles[0]:True,
#                           Candles[2]:True,
#                           Candles[3]:True,
#                           Candles[4]:True,
#                           Candles[5]:True
#                           },
#             'Step1':{Candles[0]:True,
#                      Candles[5]:True
#                      },
#             'Step2':{Candles[0]:True,
#                      Candles[5]:True
#                      }
#    }

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
