import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

# Number of events to be processed
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#process.load("FastSimulation.Configuration.CommonInputsFake_cff") For 226
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'DESIGN42_V10::All'
# Pick your geometry, Comment them all out for current geometry
#process.load("SLHCUpgradeSimulations.Geometry.hybrid_cmsIdealGeometryXML_cff")
#process.load("SLHCUpgradeSimulations.Geometry.longbarrel_cmsIdealGeometryXML_cff")
process.load("SLHCUpgradeSimulations.Geometry.Phase1_R39F16_cmsSimIdealGeometryXML_cff")


process.source = cms.Source("EmptySource")

process.o1 = cms.OutputModule("AsciiOutputModule")
process.outpath = cms.EndPath(process.o1)

process.prod = cms.EDAnalyzer("ModuleInfo_Phase2",
   fromDDD = cms.bool(True)
)
process.p = cms.Path(process.prod)
