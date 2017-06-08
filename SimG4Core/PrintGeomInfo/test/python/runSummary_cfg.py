import FWCore.ParameterSet.Config as cms

process = cms.Process("G4PrintGeometry")

#process.load('Configuration.Geometry.GeometryIdeal_cff')
#process.load('Configuration.Geometry.GeometryExtended_cff')
#process.load('Configuration.Geometry.GeometryExtended2015_cff')
process.load('Configuration.Geometry.GeometryExtended2017_cff')
#process.load('Configuration.Geometry.GeometryExtended2019_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D12_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D13_cff')

from SimG4Core.PrintGeomInfo.g4PrintGeomSummary_cfi import *

process = printGeomSummary(process)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('G4cerr', 'G4cout'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        G4cerr = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        G4cout = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
    )
)
