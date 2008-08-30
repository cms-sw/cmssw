import os, re

MIN_REQ_TS_EVENTS = 50
#Sort the candles to make sure MinBias is executed before QCD_80_120, otherwise DIGI PILEUP would not find its MinBias root files
Candles=["MinBias"            ,
         "HiggsZZ4LM200"      ,         
         "SingleElectronE1000",
         "SingleMuMinusPt10"  ,
         "SinglePiMinusE1000" ,
         "TTbar"              ,
         "QCD_80_120"         
]

CandFname={
    Candles[0]:"MINBIAS_",
    Candles[1]:"HZZLLLL_200",    
    Candles[2]:"E_1000",
    Candles[3]:"MU-_pt10",
    Candles[4]:"PI-_1000",
    Candles[5]:"TTBAR_",
    Candles[6]:"QCD_80_120"
    }

CandDesc=["Minimum Bias",
          "Higgs Boson",
          "Electron",
          "Muon",
          "Pion",
          "TTBar",
          "QCD Jets"]

# Need a little hash to match the candle with the ROOT name used by cmsDriver.py.

FileName = {}
# Hash to switch from keyword to .cfi use of cmsDriver.py:
KeywordToCfi = {}
for x in range(len(Candles)):
    
    configs   = ['MinBias.cfi',               
                 'H200ZZ4L.cfi',
                 'SingleElectronE1000.cfi',
                 'SingleMuPt10.cfi',
                 'SinglePiE1000.cfi',
                 'TTbar.cfi',               
                 'QCD_Pt_80_120.cfi']
    
    filenames = [CandFname[Candles[0]],        
                 CandFname[Candles[1]],
                 CandFname[Candles[2]],
                 CandFname[Candles[3]],
                 CandFname[Candles[4]],
                 CandFname[Candles[5]],
                 CandFname[Candles[6]]
                 ]


    KeywordToCfi[Candles[x]] = configs[x]
    FileName[Candles[x]]     = filenames[x]


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
