import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from Validation.MuonHits.muonSimHitMatcherPSet import muonSimHitMatcherPSet
from Validation.MuonGEMDigis.muonGEMDigiPSet import muonGEMDigiPSet
from Validation.MuonGEMRecHits.muonGEMRecHitPSet import gemRecHit

gemRecHitsValidation = DQMEDAnalyzer('GEMRecHitValidation',
    simTrack = muonSimHitMatcherPSet.simTrack,
    simVertex = muonSimHitMatcherPSet.simVertex,
    gemSimHit = muonSimHitMatcherPSet.gemSimHit,
    gemRecHit = gemRecHit,
    # st1, st2 of xbin, st1, st2 of ybin
    nBinGlobalZR = cms.untracked.vdouble(200,200,150,250),
    # st1 xmin xmax, st2 xmin xmax, st1 ymin ymax, st2 ymin ymax
    RangeGlobalZR = cms.untracked.vdouble(564,574,792,802,110,290,120,390),
    nBinGlobalXY = cms.untracked.int32(720),
    detailPlot = cms.bool(False),
)

gemLocalRecoValidation = cms.Sequence(gemRecHitsValidation)
