# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step4 --conditions auto:run2_mc_HIon --scenario HeavyIons -s HARVESTING:validationHarvesting+dqmHarvesting --filetype DQM --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_HI --mc --magField 38T_PostLS1 -n 100 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.HarvestingHeavyIons_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.load('DQMOffline.EGamma.electronOfflineClientSequence_cff')    # for electron DQM in HI
process.load("Validation.RecoEgamma.electronPostValidationSequence_cff")  # for electron Validation in HI

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring('file:step3_RAW2DIGI_L1Reco_RECO_VALIDATION_DQM_inDQM.root')
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step4 nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_HIon', '')

# Path and EndPath definitions
process.edmtome_step = cms.Path(process.EDMtoME)
process.validationHarvestingHI = cms.Path(process.postValidationHI)
process.dqmHarvestingPOG = cms.Path(process.DQMOfflineHeavyIons_SecondStep_PrePOG)
process.dqmsave_step = cms.Path(process.DQMSaver)
process.electronOfflineClientSequence_step = cms.Path(process.electronOfflineClientSequence)   # for DQM
process.electronPostValidationSequence_step = cms.Path(process.electronPostValidationSequence)  # for Validation

# Schedule definition
#process.schedule = cms.Schedule(process.edmtome_step,process.validationHarvesting,process.dqmHarvesting,process.dqmsave_step)

process.schedule = cms.Schedule(process.electronPostValidationSequence_step,process.electronOfflineClientSequence_step,process.edmtome_step,process.validationHarvesting,process.dqmHarvesting,process.dqmsave_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1_HI 

#call to customisation function customisePostLS1_HI imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1_HI(process)

# End of customisation functions

