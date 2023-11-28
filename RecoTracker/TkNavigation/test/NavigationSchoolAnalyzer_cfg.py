import FWCore.ParameterSet.Config as cms

# set the geometry and the GlobalTag

process = cms.Process("NavigationSchoolAnalyzer")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')
process.load("RecoTracker.TkNavigation.NavigationSchoolESProducer_cff")

#process.MessageLogger = cms.Service("MessageLogger",
#    destinations = cms.untracked.vstring('detailedInfo')
#)

#process.Tracer = cms.Service("Tracer",
#    indentation = cms.untracked.string('$$')
#)


#This has to be modified in order to read the tracker + MTD structure
process.TrackerRecoGeometryESProducer = cms.ESProducer("TrackerRecoGeometryESProducer")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.navigationSchoolAnalyzer = cms.EDAnalyzer("NavigationSchoolAnalyzer",
#    navigationSchoolName = cms.string('BeamHaloNavigationSchool')
#    navigationSchoolName = cms.string('CosmicNavigationSchool')
    navigationSchoolName = cms.string('SimpleNavigationSchool')
)

process.muonNavigationTest = cms.EDAnalyzer("MuonNavigationTest")

process.p = cms.Path(process.navigationSchoolAnalyzer+process.muonNavigationTest)
