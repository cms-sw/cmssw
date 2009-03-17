import FWCore.ParameterSet.Config as cms


process = cms.Process("PRODMIX")
process.load("SimGeneral.DataMixingModule.mixOne_data_on_sim_cfi")

process.load("Configuration.EventContent.EventContent_cff")

#process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.connect = "frontier://FrontierInt/CMS_COND_30X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_30X::All"
#process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRAFT_ALL_V4::All"
process.prefer("GlobalTag")

# Magnetic field: force mag field to be 0 tesla
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#Geometry
process.load("Configuration.StandardSequences.GeometryPilot2_cff")

# Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

#process.MessageLogger = cms.Service("MessageLogger",
#     destinations = cms.untracked.vstring('cout'),
#     cout = cms.untracked.PSet( threshold = cms.untracked.string("DEBUG") ),
#     debugModules = cms.untracked.vstring('DataMixingModule')
#)


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mix = cms.untracked.uint32(12345)
    )
)

#process.Tracer = cms.Service("Tracer",
#    indention = cms.untracked.string('$$')
#)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/uscms_data/d1/mikeh/QCD_Pt_50_80_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_NoNoise_30X.root')
#        fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/m/mikeh/cms/promptreco.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('file:MixTest.root')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

#process.p = cms.Path(process.mix+process.dump)
process.p = cms.Path(process.mix)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p,process.outpath)



