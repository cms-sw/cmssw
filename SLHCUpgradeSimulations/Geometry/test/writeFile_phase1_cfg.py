import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
#process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.Services_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("SLHCUpgradeSimulations.Geometry.Phase1_R39F16_smpx_cmsSimIdealGeometryXML_cff")
#process.load('SLHCUpgradeSimulations.Geometry.Phase1_R34F16_cmsSimIdealGeometryXML_cff')
process.load('SLHCUpgradeSimulations.Geometry.Phase1_R30F12_cmsSimIdealGeometryXML_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'DESIGN53_V3::All'
#process.GlobalTag.globaltag = 'MC_42_V10::All'
process.TrackerDigiGeometryESModule.applyAlignment = False
print process.TrackerGeometricDetESModule.fromDDD
print process.TrackerDigiGeometryESModule.fromDDD,process.TrackerDigiGeometryESModule.applyAlignment

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.load("FWCore/MessageService/MessageLogger_cfi")
process.MessageLogger.destinations = cms.untracked.vstring("logfile")
process.MessageLogger.logfile = cms.untracked.PSet(threshold = cms.untracked.string('INFO'))

process.Timing = cms.Service("Timing")

process.prodstrip = cms.EDAnalyzer("SiStripDetInfoFileWriter",
    FilePath = cms.untracked.string('SiStripDetInfo_phase1.dat'),
)

process.prodpixel = cms.EDAnalyzer("SiPixelDetInfoFileWriter",
    FilePath = cms.untracked.string('PixelSkimmedGeometry_phase1.txt'),
    WriteROCInfo = cms.untracked.bool(True)
)

process.asciiPrint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.prodstrip)
process.p2 = cms.Path(process.prodpixel)
process.ep = cms.EndPath(process.asciiPrint)


