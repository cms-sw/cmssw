import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
exoticaMuHLT = hltHighLevel
#Define the HLT path to be used. 
exoticaMuHLT.HLTPaths =['HLT_Mu3']

#Define the HLT quality cut 
hltMuonFilter = cms.EDFilter("HLT1Muon",
     inputTag = cms.InputTag("hltL3MuonCandidates"),# HLT cuts pt on"hltL3MuonCandidates"
     MaxEta = cms.double(5.0),
     MinN = cms.int32(1),
     MinPt=cms.double(20.0)
 )
                               
#Define the Reco quality cut
exoticaMuRecoQalityCut = cms.EDFilter("MuonRefSelector",#CSA07 uses PtMinMuonCountFilter
    src = cms.InputTag("muons"),
    cut = cms.string('pt > 15.0'),
    filter = cms.bool(True)            
                                      
)

#Define group sequence, using HLT bits + either HLT/Reco quality cut. 
exoticaMuHLTQualitySeq = cms.Sequence(
    exoticaMuHLT+
    hltMuonFilter
)
exoticaMuRecoQualitySeq = cms.Sequence(
    exoticaMuHLT+
    exoticaMuRecoQalityCut
)

