import FWCore.ParameterSet.Config as cms

#from Configuration.Generator.PythiaUEZ2Settings_cfi import *
from Configuration.GenProduction.DYToMuMu_M_20_TuneZ2_7TeV_pythia6_cff import generator

tfFilter = cms.EDFilter("DYGenFilter",
  code = cms.untracked.int32(13)
)


ProductionFilterSequence = cms.Sequence(generator*tfFilter)







