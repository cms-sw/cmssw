import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from Validation.MuonHits.muonSimHitMatcherPSet import muonSimHitMatcherPSet
from Validation.MuonGEMHits.MuonGEMCommonParameters_cfi import GEMValidationCommonParameters

gemSimHitValidation = DQMEDAnalyzer('GEMSimHitValidation',
    GEMValidationCommonParameters,
    gemSimHit = muonSimHitMatcherPSet.gemSimHit,
    detailPlot = cms.bool(True),
    TOFRange = cms.untracked.vdouble(18, 22, # GEM11
                                     26, 30), # GE21
)

gemSimValidation = cms.Sequence(gemSimHitValidation)
