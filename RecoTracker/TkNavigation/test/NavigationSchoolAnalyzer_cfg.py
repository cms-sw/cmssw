import FWCore.ParameterSet.Config as cms

process = cms.Process("NavigationSchoolAnalyze")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'STARTUP_V4::All'
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoTracker.TkNavigation.NavigationSchoolESProducer_cff")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo')
)

process.Tracer = cms.Service("Tracer",
    indentation = cms.untracked.string('$$')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.navigationSchoolAnalyzer = cms.EDAnalyzer("NavigationSchoolAnalyzer",
    navigationSchoolName = cms.string('BeamHaloNavigationSchool')
#    navigationSchoolName = cms.string('SimpleNavigationSchool')
)

process.p = cms.Path(process.navigationSchoolAnalyzer)


