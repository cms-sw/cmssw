import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Geometry_cff import *
from Validation.RecoParticleFlow.pfJetBenchmarkGeneric_cfi import pfJetBenchmarkGeneric
from Validation.RecoParticleFlow.caloJetBenchmarkGeneric_cfi import caloJetBenchmarkGeneric
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import * 
from RecoJets.Configuration.CaloTowersES_cfi import * 
from RecoJets.JetProducers.iterativeCone5CaloJets_cff import *
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# should do a cloning
genParticlesForJets.ignoreParticleIDs.append(14)
genParticlesForJets.ignoreParticleIDs.append(12)
genParticlesForJets.ignoreParticleIDs.append(16)
genParticlesForJets.excludeResonances = False

jetBenchmarkGeneric = cms.Sequence( 
    genJetParticles*
    recoGenJets*
    pfJetBenchmarkGeneric +
    iterativeCone5CaloJets +
    caloJetBenchmarkGeneric
    )
