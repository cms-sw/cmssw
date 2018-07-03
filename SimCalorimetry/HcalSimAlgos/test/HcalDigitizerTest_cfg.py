import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Configuration.Geometry.GeometryExtendedReco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_mc', '')

process.source = cms.Source("EmptySource")

from CalibCalorimetry.HcalPlugins.HcalConditions_forGlobalTag_cff import es_hardcode as ref_hardcode

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.hcalDigitizerTest = cms.EDAnalyzer("HcalDigitizerTest".,
    useHBUpgrade = ref_hardcode.useHBUpgrade,
    useHEUpgrade = ref_hardcode.useHEUpgrade,
    useHFUpgrade = ref_hardcode.useHFUpgrade,
    testHFQIE10  = ref_hardcode.testHFQIE10,
    hb = ref_hardcode.hb,
    he = ref_hardcode.he,
    hf = ref_hardcode.hf,
    ho = ref_hardcode.ho,
    hbUpgrade = ref_hardcode.hbUpgrade,
    heUpgrade = ref_hardcode.heUpgrade,
    hfUpgrade = ref_hardcode.hfUpgrade,
)

process.p1 = cms.Path(process.hcalDigitizerTest)

