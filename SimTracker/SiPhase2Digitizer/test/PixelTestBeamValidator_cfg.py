import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
from Configuration.ProcessModifiers.PixelCPEGeneric_cff import PixelCPEGeneric

process = cms.Process('digiTest',Phase2C9,PixelCPEGeneric)

import FWCore.ParameterSet.VarParsing as VarParsing 
options = VarParsing.VarParsing ("analysis") 
options.register("skipEvents",default=0)
# Not needed in the config, but needed because used as input option by htcondor_jobs
options.register("firstEvent",default=1)
options.maxEvents = -1
options.skipEvents = 0
options.firstEvent = 1

options.parseArguments() 


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2026D54Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 5000

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T19', '')
# list of files

process.source = cms.Source("PoolSource",
    fileNames =  cms.untracked.vstring(options.inputFiles),
    skipEvents = cms.untracked.uint32(options.skipEvents),
    )

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step1 nevts:1'),
    name = cms.untracked.string('Applications')
)
# Output definition

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string(options.outputFile),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)
process.load('DQM.SiTrackerPhase2.Phase2TrackerMonitorDigi_cff')
process.load('Validation.SiTrackerPhase2V.Phase2TrackerValidateDigi_cff')
process.load('SimTracker.SiPhase2Digitizer.PixelTestBeamValidation_cff')

# Include the same value was used in the digitizer 
process.pixelcells.ElectronsPerADC = cms.double(1700)
# Tracks are processed only for entering angles to the sensor of -2.5,2.5 degrees.
# Uncomment lines below to allow all angles (values in radians)
#process.pixelcells.TracksEntryAngleX = cms.untracked.vdouble(-1.5,1.5)
#process.pixelcells.TracksEntryAngleY = cms.untracked.vdouble(-1.5,1.5)

#process.digiana_seq = cms.Sequence(process.pixDigiMon * process.otDigiMon * process.pixDigiValid * process.otDigiValid)
process.digiana_seq = cms.Sequence(process.pixDigiMon*process.pixDigiValid*process.pixelcells)

process.load('DQMServices.Components.DQMEventInfo_cfi')
process.dqmEnv.subSystemFolder = cms.untracked.string('Ph2TkDigi')

process.dqm_comm = cms.Sequence(process.dqmEnv)

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

#process.digi_step = cms.Sequence(process.siPixelRawData*process.siPixelDigis)
process.p = cms.Path(process.digiana_seq * process.dqm_comm )
