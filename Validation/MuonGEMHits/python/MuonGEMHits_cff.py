import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from Validation.MuonHits.muonSimHitMatcherPSet import muonSimHitMatcherPSet

gemSimHitValidation = DQMEDAnalyzer('GEMSimHitValidation',
    gemSimHit = muonSimHitMatcherPSet.gemSimHit,
    simTrack = muonSimHitMatcherPSet.simTrack,
    simVertex = muonSimHitMatcherPSet.simVertex,
    # st1, st2 of xbin, st1, st2 of ybin
    nBinGlobalZR = cms.untracked.vdouble(200,200,150,250),
    # st1 xmin xmax, st2 xmin xmax, st1 ymin ymax, st2 ymin ymax
    RangeGlobalZR = cms.untracked.vdouble(564,574,792,802,110,290,120,390),
    nBinGlobalXY = cms.untracked.int32(720),
    detailPlot = cms.bool(False),
)

gemSimValidation = cms.Sequence(gemSimHitValidation)
