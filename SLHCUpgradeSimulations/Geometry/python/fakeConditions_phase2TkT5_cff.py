import FWCore.ParameterSet.Config as cms

siPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
        file = 
cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseII/Tilted/EmptyPixelSkimmedGeometry.txt')
        )
es_prefer_fake_gain = cms.ESPrefer("SiPixelFakeGainOfflineESSource","siPixelFakeGainOfflineESSource")

from SLHCUpgradeSimulations.Configuration.loadInnerTrackerConditionFromDB import *
DBConfig = DBConfiguration()
DBConfig.vGeometry = "5"
DBConfig.vLA       = "0"
DBConfig.vLAwidth  = "0"
DBConfig.vSimLA    = "0"
DBConfig.printConfig()

appendConditions(DBConfig)
es_prefer_ITconditions = cms.ESPrefer("PoolDBESSource","loadPhase2InneTrackerConditions")
