import FWCore.ParameterSet.Config as cms

#from Configuration.Generator.PythiaUEZ2Settings_cfi import *
#from Configuration.GenProduction.DYToMuMu_M_20_TuneZ2_7TeV_pythia6_cff import generator
from Configuration.GenProduction.DYToTauTau_M_20_TuneZ2_7TeV_pythia6_tauola_cff import generator

tfFilter = cms.EDFilter("DYGenFilter",
  code = cms.untracked.int32(15)
)


#process.generator.ExternalDecays.Tauola.InputCards.mdtau = cms.int32(115)
#process.generator.ExternalDecays.Tauola.InputCards.mdtau = cms.int32(116)

ProductionFilterSequence = cms.Sequence(generator*tfFilter)







