import FWCore.ParameterSet.Config as cms
from SimTracker.SiPhase2Digitizer.Phase2TrackerValidateDigi_cfi import *


pixDigiValid = digiValid.clone()
pixDigiValid.PixelPlotFillingFlag = cms.bool(True)
pixDigiValid.TopFolderName = cms.string("Ph2TkPixelDigi")

otDigiValid = digiValid.clone()
otDigiValid.PixelPlotFillingFlag = cms.bool(False)
otDigiValid.TopFolderName = cms.string("Ph2TkDigi")

