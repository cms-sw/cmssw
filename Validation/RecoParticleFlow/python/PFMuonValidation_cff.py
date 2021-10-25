import FWCore.ParameterSet.Config as cms

from DQMOffline.Muon.muonPFAnalyzer_cfi import muonPFsequence

muonPFsequenceMC = muonPFsequence.clone(
    inputTagMuonReco = "muons",
    inputTagGenParticles = "genParticles",
    inputTagVertex = "offlinePrimaryVertices",
    inputTagBeamSpot = "offlineBeamSpot",
    runOnMC = True,
    folder = 'ParticleFlow/PFMuonValidation/',
    recoGenDeltaR = 0.1,
    relCombIsoCut = 0.15,
    highPtThreshold = 200.
)

pfMuonValidationSequence = cms.Sequence( muonPFsequenceMC )
