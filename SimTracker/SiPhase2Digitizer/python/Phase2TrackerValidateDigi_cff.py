import FWCore.ParameterSet.Config as cms
from SimTracker.SiPhase2Digitizer.Phase2TrackerValidateDigi_cfi import *


pixDigiValid = digiValid.clone()
pixDigiValid.PixelPlotFillingFlag = cms.bool(True)
pixDigiValid.TopFolderName = cms.string("Ph2TkPixelDigi")
pixDigiValid.XYPositionMapH = cms.PSet(
    Nxbins = cms.int32(340),
    xmin   = cms.double(-170.),
    xmax   = cms.double(170.),
    Nybins = cms.int32(340),
    ymin   = cms.double(-170.),
    ymax   = cms.double(170.))
pixDigiValid.RZPositionMapH = cms.PSet(
    Nxbins = cms.int32(3000),
    xmin   = cms.double(-3000.0),
    xmax   = cms.double(3000.),
    Nybins = cms.int32(280),
    ymin   = cms.double(0.),
    ymax   = cms.double(280.))

otDigiValid = digiValid.clone()
otDigiValid.PixelPlotFillingFlag = cms.bool(False)
otDigiValid.TopFolderName = cms.string("Ph2TkDigi")

