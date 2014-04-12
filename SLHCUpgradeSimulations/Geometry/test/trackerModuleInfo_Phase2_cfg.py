import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'DESIGN42_V17::All'
# Pick your geometry
#process.load("SLHCUpgradeSimulations.Geometry.hybrid_cmsIdealGeometryXML_cff")
#process.load('SLHCUpgradeSimulations.Geometry.Longbarrel_cmsSimIdealGeometryXML_cff')
process.load("SLHCUpgradeSimulations.Geometry.Phase1_R34F16_cmsSimIdealGeometryXML_cff")
#process.load('Configuration.StandardSequences.GeometryExtended_cff')

process.TrackerGeometricDetExtraESModule = cms.ESProducer( "TrackerGeometricDetExtraESModule",
                                                           fromDDD = cms.bool( True )
                                                           )

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource")

process.source = cms.Source("EmptySource")

process.o1 = cms.OutputModule("AsciiOutputModule")
process.outpath = cms.EndPath(process.o1)

process.prod = cms.EDAnalyzer("ModuleInfo_Phase2",
   fromDDD = cms.bool(True)
)
process.p = cms.Path(process.prod)
