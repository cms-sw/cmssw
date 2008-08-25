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

CandDesc=["Minimum Bias",
          "Higgs Boson",
          "Electron",
          "Muon",
          "Pion",
          "TTBar",
          "QCD Jets"]
