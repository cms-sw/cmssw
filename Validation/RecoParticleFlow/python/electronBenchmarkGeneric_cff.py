import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfAllElectrons_cfi import pfAllElectrons
from Validation.RecoParticleFlow.pfElectronBenchmarkGeneric_cfi import pfElectronBenchmarkGeneric

# setting the sources

pfsource = 'pfAllElectrons'
gensource = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop * ",
    "keep pdgId = cms.vint32(11,-11)"
    )
    )

pfElectronBenchmarkGeneric.InputRecoLabel = cms.InputTag(pfsource)
pfElectronBenchmarkGeneric.InputTruthLabel = cms.InputTag(gensource)

electronBenchmarkGeneric = cms.Sequence(
    pfElectronBenchmarkGeneric
    )
