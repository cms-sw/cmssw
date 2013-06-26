import FWCore.ParameterSet.Config as cms

from Validation.RecoParticleFlow.pfJetBenchmarkGeneric_cfi import pfJetBenchmarkGeneric
from Validation.RecoParticleFlow.caloJetBenchmarkGeneric_cfi import caloJetBenchmarkGeneric
from Validation.RecoParticleFlow.goodGenJets_cfi import *

from RecoJets.Configuration.CaloTowersES_cfi import * 
from RecoJets.JetProducers.ic5CaloJets_cfi import *
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

jetBenchmarkGeneric = cms.Sequence( 
    goodGenJets *
    pfJetBenchmarkGeneric +
    iterativeCone5CaloJets +
    caloJetBenchmarkGeneric
    )
