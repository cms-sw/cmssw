import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")

process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')


process.trackerGeometry.applyAlignment = cms.bool(False)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')


process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.load("FWCore/MessageService/MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.logfile = cms.untracked.PSet(threshold = cms.untracked.string('INFO'))

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


