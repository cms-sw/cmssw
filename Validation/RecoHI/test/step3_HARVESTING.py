# Auto generated configuration file
# using: 
# Revision: 1.173
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step3 -s HARVESTING:validationHarvesting --harvesting AtRunEnd --conditions MC_37Y_V0::All --filein file:step2_RAW2DIGI_RECO_VALIDATION.root --scenario HeavyIons --mc --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    annotation = cms.untracked.string('step3 nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step2_RAW2DIGI_RECO_VALIDATION.root'),
    processingMode = cms.untracked.string('RunsAndLumis')
)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'MC_37Y_V0::All'

# Path and EndPath definitions
process.edmtome_step = cms.Path(process.EDMtoME)
process.validationHarvesting = cms.Path(process.recoMuonPostProcessors
                                        +process.postProcessorTrack)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(process.edmtome_step,
                                process.validationHarvesting,
                                process.dqmsave_step)
