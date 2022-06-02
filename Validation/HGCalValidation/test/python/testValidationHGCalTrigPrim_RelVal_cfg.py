import FWCore.ParameterSet.Config as cms 
#from Configuration.ProcessModifiers.convertHGCalDigisSim_cff import convertHGCalDigisSim

#from Configuration.Eras.Era_Phase2_cff import Phase2
#process = cms.Process('DIGI',Phase2,convertHGCalDigisSim)

# -*- coding: utf-8 -*-

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9

process = cms.Process('USER',Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
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

#import FWCore.ParameterSet.Config as cms
#process.MessageLogger = cms.Service("MessageLogger",
#        destinations = cms.untracked.vstring(                           #1
#                'myOutputFile'                                          #2
#        ),
#        myOutputFile = cms.untracked.PSet(                              #3
#                threshold = cms.untracked.string( 'WARNING' )          #4
#        ),
#)

import FWCore.ParameterSet.Config as cms
process.MessageLogger = cms.Service("MessageLogger",
                     destinations       =  cms.untracked.vstring('debugmessages'),
                     categories         = cms.untracked.vstring('interestingToMe'),
                     debugModules  = cms.untracked.vstring('*'),

                     debugmessages          = cms.untracked.PSet(
                                                threshold =  cms.untracked.string('DEBUG'),
                                                INFO       =  cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                                DEBUG   = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                                interestingToMe = cms.untracked.PSet(limit = cms.untracked.int32(10000000))
                                                    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(15)
)

# Input source
process.source = cms.Source("PoolSource",
       fileNames = cms.untracked.vstring('/store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/DoublePhoton_FlatPt-1To100/FEVT/PU200_111X_mcRun4_realistic_T15_v1_ext1-v2/1210000/F2E5E947-0CB4-D245-A943-17F2F05709D3.root'),
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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

# load HGCAL TPG simulation
process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')
process.hgcl1tpg_step = cms.Path(process.hgcalTriggerPrimitives)

# load validation
process.load('Validation.HGCalValidation.hgcalValidationTPG_cff')
process.hgcalValidationTPG_step = cms.Path(process.runHGCALValidationTPG)

#process.dqmSaver.workflow = '/validation/' + 'HGCAL' + '/TPG'
#process.dqmsave_step = cms.Path(process.dqmSaver)

# NEW added
process.onlineSaver = cms.EDAnalyzer("DQMFileSaverOnline",
    producer = cms.untracked.string('DQM'),
    path = cms.untracked.string('./'),
    tag = cms.untracked.string('new'),
)

process.o = cms.EndPath(process.onlineSaver)
process.schedule = cms.Schedule(process.hgcl1tpg_step, process.hgcalValidationTPG_step, process.o)
# END NEW added 

# Schedule definition
#process.schedule = cms.Schedule(process.hgcl1tpg_step, process.hgcalValidationTPG_step, process.dqmsave_step)

# Add early deletion of temporary data products to reduce peak memory need
#from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
#process = customiseEarlyDelete(process)
# End adding early deletion

