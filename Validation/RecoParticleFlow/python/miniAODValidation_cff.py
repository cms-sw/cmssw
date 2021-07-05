import FWCore.ParameterSet.Config as cms

# genParticles
genParticles1 = cms.EDProducer("GenParticlePruner",
                               src = cms.InputTag("genParticles"),
                               select = cms.vstring('drop *',
                               # for miniAOD matching
                               'keep status == 1')
)

from DQMOffline.PFTau.PFElectronDQMAnalyzer_cfi import pfElectronDQMAnalyzer

genParticlesValidation = pfElectronDQMAnalyzer.clone(
    BenchmarkLabel  = 'packedGenParticlesValidation/CompWithGenParticles',
    InputCollection = 'packedGenParticles',
    MatchCollection = 'genParticles1'
)

miniAODValidationSequence = cms.Sequence(
    genParticlesValidation
)
