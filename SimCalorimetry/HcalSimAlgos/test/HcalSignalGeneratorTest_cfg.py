import FWCore.ParameterSet.Config as cms

process = cms.Process("Mu2")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
       #fileNames = cms.untracked.vstring (   '/store/relval/CMSSW_3_0_0_pre7/RelValSinglePiPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0002/24CB7386-89E9-DD11-9F0E-00304875A593.root')
    fileNames = cms.untracked.vstring ('file:/uscms_data/d1/mikeh/QCD_Pt_50_80_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_NoNoise_30X.root' )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.eventContent = cms.EDAnalyzer("EventContentAnalyzer")
process.DQMStore = cms.Service("DQMStore")
process.GlobalTag.globaltag = "IDEAL_30X::All"
process.dump = cms.EDAnalyzer("HcalDigiDump")
process.load("Configuration.StandardSequences.Digi_cff")
process.load("Configuration.StandardSequences.Services_cff")

process.hcalSignal = cms.EDAnalyzer("HcalSignalGeneratorTest",
    HBHEdigiCollectionPile  = cms.InputTag("simHcalUnsuppressedDigis"),
    HOdigiCollectionPile    = cms.InputTag("simHcalUnsuppressedDigis"),
    HFdigiCollectionPile    = cms.InputTag("simHcalUnsuppressedDigis"),
    ZDCdigiCollectionPile   = cms.InputTag("ZDCdigiCollection")
)


process.p1 = cms.Path(process.mix+process.simHcalUnsuppressedDigis+process.hcalSignal)

