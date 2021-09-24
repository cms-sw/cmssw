import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process('PrintGeometry',Run3)
process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')

#from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
#process = cms.Process('PrintGeometry',Phase2C11)
#process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.G4cerr=dict()

from SimG4Core.PrintGeomInfo.g4PrintGeomSummary_cfi import *

process = printGeomSummary(process)
