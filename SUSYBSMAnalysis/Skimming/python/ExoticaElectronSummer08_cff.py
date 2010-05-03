import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
exoticaEHLT = hltHighLevel
#Define the HLT path to be used. 
exoticaEHLT.HLTPaths =['HLT_Ele10_SW_L1R']

#Define the HLT quality cut 

hltEFilter = cms.EDFilter("HLT1Electron",
     inputTag = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
     MaxEta = cms.double(5.0),
     MinN = cms.int32(1),
     MinPt=cms.double(15.0)
)
                               
#Define the Reco quality cut
exoticaERecoQalityCut = cms.EDFilter("EtaPtMinPixelMatchGsfElectronSelector",
    src = cms.InputTag("pixelMatchGsfElectrons"),
    etaMin = cms.double(-5.0),
    etaMax = cms.double(5.0),
    ptMin   = cms.double(15.),                    
    filter = cms.bool(True)                                   
                                      
)

#Define group sequence, using HLT bits + either HLT/Reco quality cut. 
exoticaEHLTQualitySeq = cms.Sequence(
    exoticaEHLT+hltEFilter
)
exoticaERecoQualitySeq = cms.Sequence(
    exoticaEHLT+
    exoticaERecoQalityCut

)

