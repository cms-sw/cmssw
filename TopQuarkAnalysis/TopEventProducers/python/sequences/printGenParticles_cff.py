import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

printGenParticles = cms.EDAnalyzer("ParticleListDrawer",
  src = cms.InputTag("genParticles"),
  maxEventsToPrint = cms.untracked.int32(-1),
  useMessageLogger = cms.untracked.bool(True)
)

printInitSubset = cms.EDAnalyzer("ParticleListDrawer",
  src = cms.InputTag("initSubset"),
  maxEventsToPrint = cms.untracked.int32(-1),
  useMessageLogger = cms.untracked.bool(True)
)

printDecaySubset = cms.EDAnalyzer("ParticleListDrawer",
  src = cms.InputTag("decaySubset"),
  maxEventsToPrint = cms.untracked.int32(-1),
  useMessageLogger = cms.untracked.bool(True)
)

printGenParticlesAndSubsets = cms.Sequence(printGenParticles *
                                           printInitSubset *
                                           printDecaySubset
                                           )
