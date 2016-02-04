import FWCore.ParameterSet.Config as cms

process = cms.Process("TPDBAn")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

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

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.prefer("GlobalTag")
process.GlobalTag.globaltag = 'GR10_P_V5::All'


process.tpDBAnalyzer = cms.EDAnalyzer("EcalTPCondAnalyzer")

process.p = cms.Path(process.tpDBAnalyzer)
