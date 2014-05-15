import FWCore.ParameterSet.Config as cms


from DQMOffline.Muon.muonPFAnalyzer_cfi import muonPFsequence




muonPFsequenceMC = muonPFsequence.clone()
muonPFsequenceMC.inputTagMuonReco = cms.InputTag("muons")
muonPFsequenceMC.inputTagGenParticles = cms.InputTag("genParticles")
muonPFsequenceMC.inputTagVertex = cms.InputTag("offlinePrimaryVertices")
muonPFsequenceMC.inputTagBeamSpot = cms.InputTag("offlineBeamSpot")
muonPFsequenceMC.runOnMC = cms.bool(True)
muonPFsequenceMC.folder = cms.string("ParticleFlow/PFMuonValidation/")
muonPFsequenceMC.recoGenDeltaR = cms.double(0.1)
muonPFsequenceMC.relCombIsoCut = cms.double(0.15)
muonPFsequenceMC.highPtThreshold = cms.double(200.)


pfMuonValidationSequence = cms.Sequence( muonPFsequenceMC )

