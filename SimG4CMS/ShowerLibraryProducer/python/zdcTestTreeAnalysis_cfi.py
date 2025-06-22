import FWCore.ParameterSet.Config as cms
from SimG4Core.Configuration.SimG4Core_cff import *

g4SimHits.Watchers = cms.VPSet(cms.PSet(
    ZdcTestTreeAnalysis = cms.PSet(
        Verbosity = cms.int32(0),
    ),
    type = cms.string('ZdcTestTreeAnalysis')
    ))
