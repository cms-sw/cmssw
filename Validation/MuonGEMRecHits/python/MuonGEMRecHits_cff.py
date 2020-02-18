import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from Validation.MuonHits.muonSimHitMatcherPSet import muonSimHitMatcherPSet
from Validation.MuonGEMHits.MuonGEMCommonParameters_cfi import GEMValidationCommonParameters
from Validation.MuonGEMRecHits.muonGEMRecHitPSet import gemRecHit

gemRecHitsValidation = DQMEDAnalyzer('GEMRecHitValidation',
    GEMValidationCommonParameters,
    gemSimHit = muonSimHitMatcherPSet.gemSimHit,
    gemRecHit = gemRecHit,
    detailPlot = cms.bool(True),
)

gemLocalRecoValidation = cms.Sequence(gemRecHitsValidation)
