# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: PAT -s VALIDATION:@miniAODValidation,DQM:@miniAODDQM --runUnscheduled --mc --era Run2_2017 --scenario pp --conditions auto:phase1_2017_realistic --eventcontent DQM --datatier DQMIO --filein /store/relval/CMSSW_10_6_1/RelValZTT_13UP17/MINIAODSIM/PUpmx25ns_106X_mc2017_realistic_v6_ul17hlt_premix_rsb-v1/20000/07F0AD9A-A1F3-8847-B95A-F4208E2EEE9F.root --geometry DB:Extended --python_filename=config.py -n 1000 --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017

process = cms.Process('DQM',Run2_2017)

# import of standard configurations
#process.load('Configuration.StandardSequences.Services_cff')
#process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
#process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
#process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Validation.RecoTau.RecoTauValidation_cff')
process.load('DQMServices.Core.DQMStoreNonLegacy_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#process.load("Configuration.Generator.ZTT_Tauola_All_hadronic_cfi")
#process.load("Configuration.Generator.QCDForPF_cfi")

#process.load('Validation.RecoJets.JetValidation_cff')
process.load('RecoJets.Configuration.RecoGenJets_cff')
process.load('RecoJets.Configuration.GenJetParticles_cff')
#process.load('Validation.RecoMET.METRelValForDQM_cff')
#process.load('Validation.RecoParticleFlow.miniAODValidation_cff')
#process.load('Validation.EventGenerator.BasicGenValidation_cff')
#process.load('Validation.RecoTau.dataTypes.ValidateTausOnQCD_cff')

#process.load('Configuration.StandardSequences.Validation_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(8500),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

process.options = cms.untracked.PSet(
	SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# Input source
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('/store/relval/CMSSW_10_6_1/RelValZTT_13UP17/MINIAODSIM/PUpmx25ns_106X_mc2017_realistic_v6_ul17hlt_premix_rsb-v1/20000/07F0AD9A-A1F3-8847-B95A-F4208E2EEE9F.root'),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_11_0_0_pre10/RelValZMM_13/MINIAODSIM/110X_mcRun2_asymptotic_v2-v1/10000/2085AB00-3B9C-664F-A31A-32852CE6F284.root'),
    secondaryFileNames = cms.untracked.vstring()
)
# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('PAT nevts:8500'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('PAT_VALIDATION_DQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

# Path and EndPath definitions
#process.prevalidation_step = cms.Path(process.prevalidationMiniAOD)
#process.validation_step = cms.EndPath(process.validationMiniAOD)
#process.validation_step = cms.EndPath(process.tauValidationSequenceMiniAOD)
#process.validation_step = cms.EndPath(process.tauValidationMiniAODRealData)
process.validation_step = cms.EndPath(process.tauValidationMiniAODZMM)
#process.dqmoffline_step = cms.EndPath(process.DQMOfflineMiniAOD)
#process.dqmofflineOnPAT_step = cms.EndPath(process.PostDQMOfflineMiniAOD)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# End of customisation functions
#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
