import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalTBAnalysis")
process.load("Geometry.EcalTestBeam.cmsEcalIdealTBGeometryXML_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.source = cms.Source("PoolSource",
    maxEvents = cms.untracked.int32(-1),
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('file:ecal_TB_simout.root')
)

process.ecalHitsValidation = cms.EDFilter("EcalSimHitsTask",
    moduleLabelTk = cms.untracked.string('g4SimHits'),
    moduleLabelVtx = cms.untracked.string('g4SimHits'),
    outputFile = cms.untracked.string('EcalSimHitsTBValidation.root'),
    verbose = cms.untracked.bool(True),
    moduleLabelMC = cms.untracked.string('source')
)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.ecalDigisValidation = DQMEDAnalyzer('EcalDigisValidation',
    moduleLabelTk = cms.untracked.string('g4SimHits'),
    moduleLabelVtx = cms.untracked.string('g4SimHits'),
    outputFile = cms.untracked.string('EcalDigisTBValidation.root'),
    verbose = cms.untracked.bool(True),
    moduleLabelMC = cms.untracked.string('source')
)

process.ecalBarrelDigisValidation = DQMEDAnalyzer('EcalBarrelDigisValidation',
    verbose = cms.untracked.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger")

process.DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

process.p1 = cms.Path(process.ecalHitsValidation*process.ecalDigisValidation*process.ecalBarrelDigisValidation)

