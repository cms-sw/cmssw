import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Configuration.Geometry.GeometryExtendedReco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_mc', '')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.hcalDigitizerTest = cms.EDAnalyzer("HcalDigitizerTest")

process.p1 = cms.Path(process.hcalDigitizerTest)

