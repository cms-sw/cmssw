# Auto generated configuration file
# using: 
# Revision: 1.14 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --beamspot Gauss --conditions auto:upgrade2019 -n -1 -s DIGI,L1,DIGI2RAW --datatier GEN-SIM-DIGI-RAW --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019 --geometry Extended2019 --magField 38T_PostLS1 --no_exec --filein file:out_file_name_selected.root --fileout file:step_digi_l1_digi2raw_OutSven.root --eventcontent FEVTDEBUGHLT
import FWCore.ParameterSet.Config as cms

process = cms.Process('DIGI2RAW')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')

process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2019_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
'root://eoscms//eos/cms/store/user/mileva/gemTest/singleMuPt1000_gen_sim_merged620slhc/gensimMuPt1000Merged.root'
)
)

process.options = cms.untracked.PSet()
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.RandomNumberGeneratorService.simMuonGEMDigis = cms.PSet(
    initialSeed = cms.untracked.uint32(1234567),
    engineName = cms.untracked.string('HepJamesRandom')    
)
process.mix.mixObjects.mixSH.crossingFrames.append('MuonGEMHits')
process.mix.mixObjects.mixSH.input.append(cms.InputTag("g4SimHits","MuonGEMHits"))
process.mix.mixObjects.mixSH.subdets.append('MuonGEMHits')
process.load('SimMuon.GEMDigitizer.muonGEMDigis_cfi')
process.load('SimMuon.GEMDigitizer.muonGEMCSCPadDigis_cfi')
process.muonDigi += process.simMuonGEMDigis
process.muonDigi += process.simMuonGEMCSCPadDigis

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.14 $'),
    annotation = cms.untracked.string('step2 nevts:-1'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    fileName = cms.untracked.string('file:outputDigi.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW')
    )
)

process.FEVTDEBUGHLToutput.outputCommands += cms.untracked.vstring('keep *_simMuonGEMDigis_*_*')
process.FEVTDEBUGHLToutput.outputCommands += cms.untracked.vstring('keep *_simMuonGEMCSCPadDigis_*_*')

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2019 

#call to customisation function cust_2019 imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2019(process)

# End of customisation functions
