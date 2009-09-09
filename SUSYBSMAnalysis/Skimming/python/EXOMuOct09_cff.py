import FWCore.ParameterSet.Config as cms

#Define the Reco quality cut
exoticaMuRecoQalityCut = cms.EDFilter("MuonRefSelector",
	src = cms.InputTag("muons"),
    cut = cms.string('pt > 10.0'),
    filter = cms.bool(True)            
                                      
)

#Define group sequence, using Reco quality cut. 

exoticaMuSeq = cms.Sequence(
    exoticaMuRecoQalityCut
)

