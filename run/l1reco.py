# Auto generated configuration file
# using: 
# Revision: 1.20 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --mc --eventcontent AODSIM --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --datatier AODSIM --conditions POSTLS162_V1::All --step RAW2DIGI,L1Reco,RECO --magField 38T_PostLS1 --geometry Extended2015 --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('file:data/crab_mix_noPU_step2.root')
#    fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/m/mulhearn/filecontent/CMSSW_6_2_X_2013-10-25-0200/src/run/gaelle_pass2/data/crab_mix_l1cust_nopu_step2.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.demo = cms.EDAnalyzer(
    'DustyDemo',
    muonSource = cms.InputTag("l1extraParticles"),
    nonIsolatedEmSource = cms.InputTag("l1extraParticles","NonIsolated"),
    etMissSource = cms.InputTag("l1extraParticles","MET"),
    htMissSource = cms.InputTag("l1extraParticles","MHT"),
    forwardJetSource = cms.InputTag("l1extraParticles","Forward"),
    centralJetSource = cms.InputTag("l1extraParticles","Central"),
    tauJetSource = cms.InputTag("l1extraParticles","Tau"),
    hfRingsSource = cms.InputTag("l1extraParticles"),
    particleMapSource = cms.InputTag("l1extraParticleMap"),
    isolatedEmSource = cms.InputTag("l1extraParticles","Isolated")
    )

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string('hl1reco.root')
    )

#process.AODSIMoutput = cms.OutputModule("PoolOutputModule",
#    compressionLevel = cms.untracked.int32(4),
#    compressionAlgorithm = cms.untracked.string('LZMA'),
#    eventAutoFlushCompressedSize = cms.untracked.int32(15728640),
#    outputCommands = process.AODSIMEventContent.outputCommands,
#    fileName = cms.untracked.string('hl1reco.root'),
#    dataset = cms.untracked.PSet(
#        filterName = cms.untracked.string(''),
#        dataTier = cms.untracked.string('AODSIM')
#    )
#)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'POSTLS162_V2::All', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.analysis_step = cms.Path(process.demo)
#process.reconstruction_step = cms.Path(process.reconstruction)
#process.endjob_step = cms.EndPath(process.endOfProcess)
#process.AODSIMoutput_step = cms.EndPath(process.AODSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.analysis_step)#,process.reconstruction_step,process.endjob_step,process.AODSIMoutput_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1 
 
#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

# End of customisation functions
