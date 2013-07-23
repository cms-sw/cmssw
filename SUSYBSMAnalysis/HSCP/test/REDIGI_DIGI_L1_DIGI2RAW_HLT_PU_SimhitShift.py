# Auto generated configuration file
# using: 
# Revision: 1.303.2.7 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: REDIGI --step DIGI,L1,DIGI2RAW,HLT:GRun --conditions START42_V11::All --pileup E7TeV_FlatDist10_2011EarlyData_50ns_PoissonOOT --datamix NODATAMIXER --customise Configuration/StandardSequences/DigiToRecoNoPU.customise --eventcontent RAWSIM --datatier GEN-SIM-RAW --no_exec -n 3
import FWCore.ParameterSet.Config as cms

process = cms.Process('HLTSIMHITSHIFTER')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_E7TeV_FlatDist10_2011EarlyData_50ns_PoissonOOT')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       'file:~/lpcphys/HSCPRawData/Gluino900.root'
	   )
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('REDIGI nevts:3'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('HSCPSimHitShiftHLT.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RAW')
    )
)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'START42_V11::All'
process.g4SimHits = cms.EDProducer("SimHitShifter",
			ShiftFileName = cms.untracked.string('2011_2Map.txt')
								   )

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.mix+process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.endjob_step,process.RAWSIMoutput_step])

# customisation of the process.

# Automatic addition of the customisation function from Configuration.StandardSequences.DigiToRecoNoPU
from Configuration.StandardSequences.DigiToRecoNoPU import customise 

#call to customisation function customise imported from Configuration.StandardSequences.DigiToRecoNoPU
process = customise(process)

# End of customisation functions
