import FWCore.ParameterSet.Config as cms

process = cms.Process("TPDBAn")
process.load("CondCore.CondDB.CondDB_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(135445)
#    firstRun = cms.untracked.uint32(135175)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.prefer("GlobalTag")
process.GlobalTag.globaltag = '112X_dataRun3_HLT_v3'


process.tpDBAnalyzer = cms.EDAnalyzer("EcalTPCondAnalyzer")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EcalTPCondAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(100000000)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('tpDBAnalyzer')
)

process.p = cms.Path(process.tpDBAnalyzer)
