import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from CommonTools.ParticleFlow.ParticleSelectors.pfAllElectrons_cfi import pfAllElectrons

from Validation.RecoParticleFlow.pfElectronBenchmarkGeneric_cfi import pfElectronBenchmarkGeneric

from  CommonTools.ParticleFlow.pfNoPileUp_cff import *
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
pfAllElectrons.src = cms.InputTag("pfNoPileUp")

electronBenchmarkGeneric = cms.Sequence(
    pfNoPileUpSequence+
    pfAllElectrons +
    gensource + 
    pfElectronBenchmarkGeneric
)
