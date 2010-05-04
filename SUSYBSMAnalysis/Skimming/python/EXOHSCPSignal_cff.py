import FWCore.ParameterSet.Config as cms


exoticaRecoHSCPDedxFilter = cms.EDFilter("HSCPFilter",
										 inputTrackCollection = cms.InputTag("generalTracks"),									 
										 inputDedxCollection =  cms.InputTag("dedxHarmonic2"),									 
										 trkPtMin = cms.double(20),
										 dedxMin =cms.double(4.),
										 ndedxHits = cms.int32(5),
										 etaMin= cms.double(-2.4),
										 etaMax= cms.double(2.4),
										 chi2nMax = cms.double(10),
										 d0Max = cms.double(10),
										 dzMax = cms.double(20)
                                      
)

exoticaRecoHSCPMuonFilter = cms.EDFilter("MuonRefSelector",
    src = cms.InputTag("muons"),
    cut = cms.string('pt > 30.0'),
    filter = cms.bool(True)            
                                      
)
exoticaHSCPDedxSeq = cms.Sequence(exoticaRecoHSCPDedxFilter)

exoticaHSCPMuonSeq = cms.Sequence(exoticaRecoHSCPMuonFilter)




