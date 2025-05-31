# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: PAT -s VALIDATION:@miniAODValidation,DQM:@miniAODDQM --runUnscheduled --mc --era Run2_2017 --scenario pp --conditions auto:phase1_2017_realistic --eventcontent DQM --datatier DQMIO --filein /store/relval/CMSSW_10_6_1/RelValZTT_13UP17/MINIAODSIM/PUpmx25ns_106X_mc2017_realistic_v6_ul17hlt_premix_rsb-v1/20000/07F0AD9A-A1F3-8847-B95A-F4208E2EEE9F.root --geometry DB:Extended --python_filename=config.py -n 1000 --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2023_cff import Run3_2023

process = cms.Process('DQM',Run3_2023)

# import of standard configurations
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Validation.RecoTau.RecoTauValidation_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('RecoJets.Configuration.RecoGenJets_cff')
process.load('RecoJets.Configuration.GenJetParticles_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1), #100
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

#https://github.com/cms-sw/cmssw/issues/43793#issuecomment-1912388830
#process.options = cms.untracked.PSet(
#    TryToContinue = cms.untracked.vstring('ProductNotFound')
#)

#process_name='QCD'
#process_name='ZTT'
#process_name='ZEE'
#process_name='ZMM'
process_name='JETHT'
#process_name='DoubleElectron'
#process_name='DoubleMuon'

process_dict ={
    'ZMM':[
        #'/store/relval/CMSSW_13_0_11/RelValZMM_14/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV208-v1/2580000/c1f4789f-0143-4c1c-8c5b-4737ece99548.root',
        '/store/relval/CMSSW_13_0_11/RelValZMM_14/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV208-v2/2580000/6fb27712-4897-4b5c-8eea-4d04f9f88fca.root',
        '/store/relval/CMSSW_13_0_11/RelValZMM_14/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV208-v2/2580000/ddb0422f-d759-40ec-888c-82dfddc8d2d3.root',
    ],
    'ZEE':[
        #'/store/relval/CMSSW_13_0_11/RelValZEE_14/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV208-v1/2580000/98dd45f2-b2a6-4037-9d12-850dee82abff.root',
        '/store/relval/CMSSW_13_0_11/RelValZEE_14/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV208-v2/2580000/ba2f4c0e-a874-42cb-8750-5bce8a6a86f8.root',
        '/store/relval/CMSSW_13_0_11/RelValZEE_14/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV208-v2/2580000/f3e2bb4d-5988-42fa-800c-06556f1f02bd.root',
    ],
    'ZTT':[
        #ttbar_relval_23_v1
        #"/store/relval/CMSSW_13_0_11/RelValTTbar_14TeV/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV204-v1/2590000/69c3b054-bd07-4178-9481-14f04f79ca95.root",
        #ttbar_relval_23_v2
        #"/store/relval/CMSSW_13_0_11/RelValTTbar_14TeV/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV204-v2/2590000/02c835ae-8352-4759-b908-0aca15feb36d.root",
        #"/store/relval/CMSSW_13_0_11/RelValTTbar_14TeV/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV204-v2/2590000/3f2b92d6-3fb1-456b-92db-e1c65ad0d29e.root",
        #"/store/relval/CMSSW_13_0_11/RelValTTbar_14TeV/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV204-v2/2590000/9f2d1acd-f33a-4199-b800-92076e327deb.root",
        #"/store/relval/CMSSW_13_0_11/RelValTTbar_14TeV/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV204-v2/2590000/d6e551c5-462a-4466-9689-77e665bd25df.root",
        #"/store/relval/CMSSW_13_0_11/RelValTTbar_14TeV/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV204-v2/2590000/e24e6546-95b7-4125-90b3-d33b56ffad25.root",
        '/store/relval/CMSSW_13_0_11/RelValZTT_14/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV208-v1/2580000/433751c7-b9a1-4036-b1c8-668e679db182.root',
        '/store/relval/CMSSW_13_0_11/RelValZTT_14/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV208-v1/2580000/c048525f-000d-4fb5-9694-78d73e9e15f5.root',
        ##'/store/relval/CMSSW_13_0_11/RelValZTT_14/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV208-v2/2580000/2ced95d2-ebcc-4456-abaa-736aade50b0f.root',
        ##'/store/relval/CMSSW_13_0_11/RelValZTT_14/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV208-v2/2580000/39aac844-e272-4df6-b354-2e07de7039eb.root',
        ##'/store/relval/CMSSW_13_0_11/RelValZTT_14/MINIAODSIM/PU_130X_mcRun3_2023_realistic_relvals2023D_v1_RV208-v2/2580000/713106b9-62a3-40ea-a541-a2bad98e1171.root',
    ],
    'QCD':[
        '/store/relval/CMSSW_11_0_0_pre10/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/110X_mcRun2_asymptotic_v2-v1/10000/91461CFA-8CEF-8C4E-864D-FFC1760FAC67.root'
    ],
    'JETHT':[
        "/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/775/00000/277b3a14-ab60-49c6-bcb1-81eedf109f5d.root",
        "/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/775/00000/7ceb4e30-fb58-4de1-823d-0178aca37c46.root",
        "/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/775/00000/9387a49c-c7f5-4109-9be1-4412b1dcb92f.root",
        "/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/775/00000/96bb89e2-2cba-4fd3-9141-89796e992f4c.root",
        "/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/775/00000/e7a4213b-a29b-40ca-8cce-f60bb318caae.root",
        #"/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/776/00000/24a6b480-1c8d-44f8-b164-6c1aca5747a9.root",
        #"/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/776/00000/5d43bdf0-616a-42c6-9453-5450733c9760.root",
        #"/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/776/00000/cffa86eb-c969-4115-beaa-d45ffb3ff872.root",
        #"/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/776/00000/da5b4f4c-1e45-41de-8021-8f62f4277bcf.root",
        #"/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/776/00000/ff03336f-2497-4992-8e85-927366556650.root",
        #"/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/790/00000/7e8d23ef-950b-41d6-8c6b-1806ee265c03.root",
    ],
    'DoubleElectron':[
        "/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/775/00000/9387a49c-c7f5-4109-9be1-4412b1dcb92f.root",
    ],
    'DoubleMuon':[
        "/store/data/Run2023D/Tau/MINIAOD/PromptReco-v2/000/370/775/00000/9387a49c-c7f5-4109-9be1-4412b1dcb92f.root",
    ],
}

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(*process_dict[process_name]),
    secondaryFileNames = cms.untracked.vstring()
)
# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('PAT nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('PAT_VALIDATION_DQM_'+process_name+'.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '126X_mcRun3_2023_forPU65_v1', '')

# Path and EndPath definitions
#process.prevalidation_step = cms.Path(process.prevalidationMiniAOD)
#process.validation_step = cms.EndPath(process.validationMiniAOD)
#process.validation_step = cms.EndPath(process.tauValidationSequenceMiniAOD)
process.validation_step = cms.EndPath(process.tauValidationSequenceMiniAODonMC) if process_name in ["ZTT", "ZMM", "ZEE", "QCD"] else cms.EndPath(process.tauValidationSequenceMiniAODonDATA)
#process.validation_step = cms.EndPath(process.tauValidationMiniAODRealData)
#process.validation_step = cms.EndPath(process.tauValidationMiniAODZMM)
#process.dqmoffline_step = cms.EndPath(process.DQMOfflineMiniAOD)
#process.dqmofflineOnPAT_step = cms.EndPath(process.PostDQMOfflineMiniAOD)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# End of customisation functions
#do not add changes to your config after this point (unless you know what you are doing)
#from FWCore.ParameterSet.Utilities import convertToUnscheduled # test: will be deprecated soon
#process=convertToUnscheduled(process) # test: will be deprecated soon


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
