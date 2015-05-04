# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --conditions auto:run2_mc -s RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM --datatier GEN-SIM-RECO,DQMIO -n 10 --magField 38T_PostLS1 --eventcontent RECOSIM,DQM --io RECOUP15.io --python RECOUP15.py --no_exec --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO")

# import of standard configurations
process.load("Configuration.StandardSequences.Services_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
# Load geometry from files so that we pick up the new Reco Material
# process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_PostLS1_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.L1Reco_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("CommonTools.ParticleFlow.EITopPAG_cff")
process.load("Configuration.StandardSequences.Validation_cff")
process.load("DQMOffline.Configuration.DQMOfflineMC_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    "/store/relval/CMSSW_7_4_0_pre8/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_74_V7-v1/00000/2AD75824-F7BC-E411-A98F-0025905A48B2.root",
    "/store/relval/CMSSW_7_4_0_pre8/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_74_V7-v1/00000/34DAB86B-2EBD-E411-8A68-0025905A60FE.root",
    "/store/relval/CMSSW_7_4_0_pre8/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_74_V7-v1/00000/38A75F26-F7BC-E411-B1C7-0025905A60EE.root",
    "/store/relval/CMSSW_7_4_0_pre8/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_74_V7-v1/00000/3E2FFB1A-F7BC-E411-AE8A-0025905B8606.root",
    "/store/relval/CMSSW_7_4_0_pre8/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_74_V7-v1/00000/DCB5C918-F7BC-E411-AD78-0025905A60F8.root"
    ),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string("step3 nevts:10"),
    name = cms.untracked.string("Applications"),
    version = cms.untracked.string("$Revision: 1.19 $")
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string("GEN-SIM-RECO"),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string("file:step3.root"),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string("DQMIO"),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string("file:step3_inDQM.root"),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.playback =  True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:run2_mc", '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.eventinterpretaion_step = cms.Path(process.EIsequence)
process.prevalidation_step = cms.Path(process.prevalidation)
process.dqmoffline_step = cms.Path(process.DQMOffline)
process.validation_step = cms.EndPath(process.validation)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.eventinterpretaion_step,process.prevalidation_step,process.validation_step,process.dqmoffline_step,process.RECOSIMoutput_step,process.DQMoutput_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

