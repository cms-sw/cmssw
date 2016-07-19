import FWCore.ParameterSet.Config as cms
from SimTracker.SiPhase2Digitizer.Phase2TrackerMonitorDigi_cfi import *


pixDigiMon = digiMon.clone()
pixDigiMon.PixelPlotFillingFlag = cms.bool(True)
pixDigiMon.TopFolderName = cms.string("Ph2TkPixelDigi")

otDigiMon = digiMon.clone()
otDigiMon.PixelPlotFillingFlag = cms.bool(False)
otDigiMon.TopFolderName = cms.string("Ph2TkDigi")
