import FWCore.ParameterSet.Config as cms

siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
        file = 
cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseII/Tilted/EmptyPixelSkimmedGeometry.txt')
        )
es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

from SLHCUpgradeSimulations.Configuration.loadInnerTrackerConditionFromDB import *
SiPhase2DBConfig = SiPhase2DBConfiguration()
SiPhase2DBConfig.vGeometry = "14"
SiPhase2DBConfig.vLA       = "0" # uH = 0.106 everywhere
SiPhase2DBConfig.vLAwidth  = "0" # empty payload
SiPhase2DBConfig.vSimLA    = "0" # uH = 0.106 everywhere
appendConditions(SiPhase2DBConfig)
es_prefer_ITconditions = cms.ESPrefer("PoolDBESSource","loadPhase2InneTrackerConditions")

