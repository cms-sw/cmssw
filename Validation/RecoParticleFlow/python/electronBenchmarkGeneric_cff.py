import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllElectrons_cfi import pfAllElectrons

from Validation.RecoParticleFlow.pfElectronBenchmarkGeneric_cfi import pfElectronBenchmarkGeneric

# setting the sources

gensource = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop * ",
    "keep pdgId = {e-}",
    "keep pdgId = {e+}"
    )
)

pfElectronBenchmarkGeneric.InputRecoLabel = cms.InputTag("pfAllElectrons")
pfElectronBenchmarkGeneric.InputTruthLabel = cms.InputTag("gensource")

electronBenchmarkGeneric = cms.Sequence(
    pfAllElectrons +
    gensource + 
    pfElectronBenchmarkGeneric
)
