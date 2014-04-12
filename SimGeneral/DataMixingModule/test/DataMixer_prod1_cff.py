import FWCore.ParameterSet.Config as cms


process = cms.Process("PRODMIX")


process.load('Configuration/StandardSequences/Services_cff')

process.load("SimGeneral.DataMixingModule.mixOne_data_on_sim_cfi")

process.load("Configuration.EventContent.EventContent_cff")

#---------------
#add this to re-create the CrossingFrame for Hcal input
#
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
#
#substitute so we only do Hcal:
#
MyMixCaloHits = cms.PSet(input = cms.VInputTag(cms.InputTag("g4SimHits","HcalHits"),
               cms.InputTag("g4SimHits","HcalTB06BeamHits"), cms.InputTag("g4SimHits","ZDCHITS")),
      type = cms.string('PCaloHit'),
      subdets = cms.vstring('HcalHits','HcalTB06BeamHits','ZDCHITS')
)
process.mix.mixObjects = cms.PSet(mixCH = cms.PSet(MyMixCaloHits))
#---------------
#
#modify Digi sequences to regenerate Trigger primitives for calorimeter
#
process.load('Configuration/StandardSequences/DigiDM_cff')
#
#---------------
#modify inputs to L1Simulator so it looks at our new Digis
#
process.load('Configuration/StandardSequences/SimL1EmulatorDM_cff')
#
#process.simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigisDM'
#process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag("mixData","MuonCSCComparatorDigisDM")
#process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag("mixData","MuonCSCWireDigisDM")
#
#process.simRpcTriggerDigis.label = 'muonRPCDigisDM'
#
# Calorimeter should have been provided by remaking the TriggerPrimitives with new Digis...
#
#---------------
# need to do DigiToRaw to complete simulation step
#
process.load('Configuration/StandardSequences/DigiToRawDM_cff')
#
#process.siPixelRawData.InputLabel = cms.InputTag("siPixelDigisDM")
#process.SiStripDigiToRaw.InputModuleLabel = cms.string('mixData')
#process.SiStripDigiToRaw.InputDigiLabel = cms.string('siStripDigisDM')
#
#process.ecalPacker.Label = 'mixData'
#process.ecalPacker.InstanceEB = 'EBDigiCollectionDM'
#process.ecalPacker.InstanceEE = 'EEDigiCollectionDM'
#process.esDigiToRaw.InstanceES = cms.string('ESDigiCollectionDM')
#process.esDigiToRaw.Label = cms.string('mixData')
#
#process.hcalRawData.HBHE = cms.untracked.InputTag("mixData")          
#process.hcalRawData.HF = cms.untracked.InputTag("mixData")           
#process.hcalRawData.HO = cms.untracked.InputTag("mixData")            
#process.hcalRawData.ZDC = cms.untracked.InputTag("mixData")
#
#process.cscpacker.wireDigiTag = cms.InputTag("mixData","MuonCSCWireDigiDM")
#process.cscpacker.stripDigiTag = cms.InputTag("mixData","MuonCSCStripDigiDM")          
#process.cscpacker.comparatorDigiTag = cms.InputTag("mixData","MuonCSCComparatorDigiDM")
#process.dtpacker.digiColl =  cms.InputTag('mixData')
#process.DigiToRaw.rpcpacker.
#RPCs assume there is only one RPCDigiCollection - bad!!!
#
#HAVE TO DO SOMETHING ABOUT THESE IN DIGI STEP
#ecalPacker.labelEBSRFlags = "simEcalDigis:ebSrFlags" #?????
#ecalPacker.labelEESRFlags = "simEcalDigis:eeSrFlags" #?????
#ES: ecalPacker.labelESsRFlags = "simEcalDigis:eeSrFlags" #?????
#
#
#
#
#---------------
# do HLT while we're here
#
process.load('HLTrigger/Configuration/HLT_1E31_cff')
#
# no redirection necessary, since L1 is already simulated
#---------------

#process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

#process.GlobalTag.connect = "frontier://FrontierInt/CMS_COND_30X_GLOBALTAG"

#process.GlobalTag.globaltag = "CRAFT_31X::All"
process.GlobalTag.globaltag = "IDEAL_31X::All"
#process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRAFT_ALL_V4::All"
#process.prefer("GlobalTag")

# Magnetic field: force mag field to be 0 tesla
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#Geometry
process.load("Configuration.StandardSequences.GeometryPilot2_cff")

# Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
     destinations = cms.untracked.vstring('cout'),
     cout = cms.untracked.PSet( threshold = cms.untracked.string("DEBUG") ),
     debugModules = cms.untracked.vstring('DataMixingModule')
)


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        mixData = cms.untracked.uint32(12345)
    )
)

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/F076816C-504D-DE11-A394-001D09F253C0.root')
#fileNames = cms.untracked.vstring('file:/uscms/home/mikeh/work/CMSSW_3_1_0_pre8/src/Configuration/PyReleaseValidation/data/QCD_Pt_80_120_cfi_GEN_SIM_DIGI.root')
#fileNames = cms.untracked.vstring('file:/uscms/home/mikeh/work/CMSSW_3_1_0_pre7/src/SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root')
#fileNames = cms.untracked.vstring('file:/uscms/home/mikeh/work/CMSSW_3_1_0_pre7/src/myreco_D_RAW2DIGI_RECO.root')
#    fileNames = cms.untracked.vstring('file:/uscms_data/d1/mikeh/QCD_Pt_50_80_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_NoNoise_30X.root')
#        fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/m/mikeh/cms/promptreco.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('file:MixTest.root')
)


#
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.pdump = cms.Path(process.dump)

process.pMix = cms.Path(process.mix+process.mixData)
process.outpath = cms.EndPath(process.FEVT)


#process.schedule = cms.Schedule(process.pMix,process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.HLTSchedule,process.pdump,process.outpath)

#this one works without HLT:
#process.schedule = cms.Schedule(process.pMix,process.digitisation_step,process.pdump,process.L1simulation_step,process.digi2raw_step,process.outpath)
process.schedule = cms.Schedule(process.pMix,process.digitisation_step,process.pdump,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.outpath])
        





