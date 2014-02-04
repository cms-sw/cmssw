import FWCore.ParameterSet.Config as cms

# Module to create simulated ME0 Pre Reco digis.
simMuonME0Digis = cms.EDProducer("ME0DigiPreRecoProducer",
    inputCollection = cms.string('g4SimHitsMuonME0Hits'),
    digiPreRecoModelString = cms.string('PreRecoGaussian'),
    timeResolution = cms.double(0.001), # in ns 
    phiResolution = cms.double(0.05), # in cm average resoltion along local x in case of no correlation
    etaResolution = cms.double(1.),# in cm average resoltion along local y in case of no correlation
    useCorrelation  = cms.bool(False),
    useEtaProjectiveGEO  = cms.bool(False),
#    digiPreRecoModelString = cms.string('PreRecoNoSmear'),
                                 
)
