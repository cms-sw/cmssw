import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.PFElectronDQMAnalyzer_cfi import pfElectronDQMAnalyzer

pfAllElectrons = cms.EDFilter("PdgIdPFCandidateSelector",
                                      pdgId = cms.vint32(11, -11),
                                      src = cms.InputTag("particleFlow")
                                      )

gensource = cms.EDProducer("GenParticlePruner",
                           src = cms.InputTag("genParticles"),
                           select = cms.vstring('drop *',
                                                'keep pdgId = 11',
                                                'keep pdgId = -11'
                                                )
                           )


pfElectronValidation1 = pfElectronDQMAnalyzer.clone()
pfElectronValidation1.BenchmarkLabel  = cms.string('PFElectronValidation/CompWithGenElectron')
pfElectronValidationSequence = cms.Sequence( gensource + pfElectronValidation1 )


# NoTracking
pfElectronValidation2 = pfElectronDQMAnalyzer.clone()
pfElectronValidation2.InputCollection = cms.InputTag('pfAllElectrons','','PFlowDQMnoTracking')
pfElectronValidation2.MatchCollection = cms.InputTag('gensource','','PFlowDQMnoTracking')
pfElectronValidation2.BenchmarkLabel  = cms.string('PFElectronValidation/CompWithGenElectron')
pfElectronValidationSequence_NoTracking = cms.Sequence( pfElectronValidation2 )
