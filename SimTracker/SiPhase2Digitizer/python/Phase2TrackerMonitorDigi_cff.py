import FWCore.ParameterSet.Config as cms
from SimTracker.SiPhase2Digitizer.Phase2TrackerMonitorDigi_cfi import *


pixDigiMon = digiMon.clone()
pixDigiMon.PixelPlotFillingFlag = cms.bool(True)
pixDigiMon.TopFolderName = cms.string("Ph2TkPixelDigi")
pixDigiMon.PositionOfDigisH = cms.PSet(
    Nxbins = cms.int32(1350),
    xmin   = cms.double(0.5),
    xmax   = cms.double(1350.5),
    Nybins = cms.int32(450),
    ymin   = cms.double(0.5),
    ymax   = cms.double(450.5))
pixDigiMon.XYPositionMapH = cms.PSet(
    Nxbins = cms.int32(340),
    xmin   = cms.double(-170.),
    xmax   = cms.double(170.),
    Nybins = cms.int32(340),
    ymin   = cms.double(-170.),
    ymax   = cms.double(170.))
pixDigiMon.RZPositionMapH = cms.PSet(
    Nxbins = cms.int32(3000),
    xmin   = cms.double(-3000.0),
    xmax   = cms.double(3000.),
    Nybins = cms.int32(280),
    ymin   = cms.double(0.),
    ymax   = cms.double(280.))

otDigiMon = digiMon.clone()
otDigiMon.PixelPlotFillingFlag = cms.bool(False)
otDigiMon.TopFolderName = cms.string("Ph2TkDigi")
