# Auto generated configuration file
# using: 
# Revision: 1.265 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: Neutron_cfi -s GEN,SIM,DIGI,RECO --conditions auto:startup --eventcontent FEVTDEBUG --datatier GEN-SIM-RECO --processName GENSIMDIGIRECO310X --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('DIGIRECO310X')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('SimG4CMS.Calo.GeometryAPD_cff')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('SimG4CMS.Calo.DigiAPD_cff')
process.load('SimG4CMS.Calo.RecoAPD_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:simevent_APD_Epoamxy.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('Neutron_cfi nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('Neutron_DIGI_RECO.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RECO')
    )
)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'START310_V4::All'

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi)
process.reconstruction_step = cms.Path(process.localreco)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.reconstruction_step,process.endjob_step,process.FEVTDEBUGoutput_step)

process.simEcalUnsuppressedDigis.apdAddToBarrel = True
process.simEcalUnsuppressedDigis.apdSeparateDigi = False
