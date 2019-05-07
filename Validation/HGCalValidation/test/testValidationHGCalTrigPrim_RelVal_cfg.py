import FWCore.ParameterSet.Config as cms 
from Configuration.ProcessModifiers.convertHGCalDigisSim_cff import convertHGCalDigisSim

from Configuration.Eras.Era_Phase2_cff import Phase2
process = cms.Process('DIGI',Phase2,convertHGCalDigisSim)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D17_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.DQMStore = cms.Service("DQMStore")
# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff")
# import DQMStore service
process.load('DQMOffline.Configuration.DQMOffline_cff')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(15)
)

# Input source
process.source = cms.Source("PoolSource",
       fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/j/jsauvan/public/HGCAL/TestingRelVal/CMSSW_9_3_7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW/93X_upgrade2023_realistic_v5_2023D17noPU-v2/2661406C-972C-E811-9754-0025905A60DE.root'),
       inputCommands=cms.untracked.vstring(
           'keep *',
           'drop l1tEMTFHit2016Extras_simEmtfDigis_CSC_HLT',
           'drop l1tEMTFHit2016Extras_simEmtfDigis_RPC_HLT',
           'drop l1tEMTFHit2016s_simEmtfDigis__HLT',
           'drop l1tEMTFTrack2016Extras_simEmtfDigis__HLT',
           'drop l1tEMTFTrack2016s_simEmtfDigis__HLT',
           )
       )

# Production Info
#process.configurationMetadata = cms.untracked.PSet(
#    version = cms.untracked.string('$Revision: 1.20 $'),
#    annotation = cms.untracked.string('SingleElectronPt10_cfi nevts:10'),
#    name = cms.untracked.string('Applications')
#)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# load HGCAL TPG simulation
process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')
process.hgcl1tpg_step = cms.Path(process.hgcalTriggerPrimitives)
# Change to V7 trigger geometry for older samples
#  from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_ZoltanSplit_V7
#  process = custom_geometry_ZoltanSplit_V7(process)

# load validation
process.load('Validation.HGCalValidation.hgcalValidationTPG_cff')
process.hgcalValidationTPG_step = cms.Path(process.runHGCALValidationTPG)

process.dqmSaver.workflow = '/validation/' + 'HGCAL' + '/TPG'
process.dqmsave_step = cms.Path(process.dqmSaver)

# Schedule definition
process.schedule = cms.Schedule(process.hgcl1tpg_step, process.hgcalValidationTPG_step, process.dqmsave_step)

# Add early deletion of temporary data products to reduce peak memory need
#from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
#process = customiseEarlyDelete(process)
# End adding early deletion

