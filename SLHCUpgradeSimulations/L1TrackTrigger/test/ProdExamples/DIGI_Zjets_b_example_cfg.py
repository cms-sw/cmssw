# Auto generated configuration file
# using: 
# Revision: 1.20 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step1 --filein dbs:/Neutrino_Pt2to20_gun/TTI2023Upg14-DES23_62_V1-v1/GEN-SIM --fileout file:L1T-2023TTIUpg14D-00002.root --pileup_input dbs:/MinBias_TuneZ2star_14TeV-pythia6/TTI2023Upg14-DES23_62_V1-v1/GEN-SIM --mc --eventcontent FEVTDEBUGHLT --pileup AVE_140_BX_25ns --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023TTI,Configuration/DataProcessing/Utils.addMonitoring --datatier GEN-SIM-DIGI-RAW --conditions PH2_1K_FB_V3::All --step DIGI:pdigi_valid,L1,L1TrackTrigger,DIGI2RAW,RECO:pixeltrackerlocalreco --magField 38T_PostLS1 --geometry Extended2023TTI,Extended2023TTIReco --python_filename L1T-2023TTIUpg14D-00002_1_cfg.py --no_exec -n 48
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

index=77
puindex=77 + 110

npileup=140

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

nevents = 125

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(nevents)
)


# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('/store/group/dpg_trigger/comm_trigger/L1TrackTrigger/620_SLHC12/Extended2023TTI/Zjets_b/SIM/Pythia_Ztobbbar_SIM.root') ,
    skipEvents=cms.untracked.uint32( 9500 )
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('step1 nevts:48'),
    name = cms.untracked.string('Applications')
)

# Output definition

theFileName =   'file:Zjets_b_E2023TTI_PU'+str(npileup)+'_'+str(index)+'.root'

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    fileName = cms.untracked.string( theFileName ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW')
    )
)

# Additional output definition

# Other statements
process.mix.input.nbPileupEvents.averageNumber = cms.double(npileup)
process.mix.bunchspace = cms.int32(25)
process.mix.minBunch = cms.int32(-12)
process.mix.maxBunch = cms.int32(3)
minBiasFileNames = cms.untracked.vstring(
   "/store/mc/TTI2023Upg14/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v1/00000/28D7125E-BFD1-E311-BE4A-848F69FD294F.root","/store/mc/TTI2023Upg14/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v1/00000/28D7254D-55D0-E311-87EE-00266CFCC1B4.root","/store/mc/TTI2023Upg14/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v1/00000/28E749DE-93D1-E311-A2AD-002618943930.root","/store/mc/TTI2023Upg14/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/DES23_62_V1-v1/00000/28F52E2F-4CD0-E311-B1A9-008CFA064704.root",
)

process.mix.digitizers = cms.PSet(process.theDigitizersValid)

process.mix.input.fileNames = minBiasFileNames
process.mix.input.seed = cms.int32(21345+1000*puindex)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.L1TrackTrigger_step = cms.Path(process.L1TrackTrigger)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.reconstruction_step = cms.Path(process.pixeltrackerlocalreco)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.L1TrackTrigger_step,process.digi2raw_step,process.reconstruction_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023TTI 

#call to customisation function cust_2023TTI imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023TTI(process)

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring 

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)

# End of customisation functions
