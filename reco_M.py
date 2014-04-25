# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --datatier GEN-SIM-RECO,DQMROOT --conditions auto:startup -s RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM --eventcontent RECOSIM,DQM -n 100 --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms

process = cms.Process('RERECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('CommonTools.ParticleFlow.EITopPAG_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
#    '/store/relval/CMSSW_7_1_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PU25ns_POSTLS171_V1-v2/00000/1A198FA1-B3BC-E311-963F-02163E00CD6B.root',
#    '/store/relval/CMSSW_7_1_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PU25ns_POSTLS171_V1-v2/00000/7468E9C0-A9BC-E311-8160-02163E00E74E.root',
#    '/store/relval/CMSSW_7_1_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PU25ns_POSTLS171_V1-v2/00000/94BCF852-43BD-E311-9307-0025904B57DA.root',
#    '/store/relval/CMSSW_7_1_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PU25ns_POSTLS171_V1-v2/00000/A681BD21-CFBC-E311-B0B0-02163E00EA9A.root',
#    '/store/relval/CMSSW_7_1_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PU25ns_POSTLS171_V1-v2/00000/E4E9454E-7DBD-E311-9D2B-02163E00EA65.root',
#    '/store/relval/CMSSW_7_1_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PU25ns_POSTLS171_V1-v2/00000/EC1D5CC1-A9BC-E311-B320-02163E00A196.root'
#'/store/relval/CMSSW_7_1_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/065CA478-17B6-E311-83AE-0025905A611C.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/08492532-1BB6-E311-90F7-0025905938AA.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/0ACD0B5C-E3B6-E311-B784-003048679076.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/242FEF51-0CB6-E311-8DD3-0025905938B4.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/247306DB-0EB6-E311-8073-0025905A611C.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/28945BC0-22B6-E311-A3CF-0026189438A0.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/86A79188-2BB6-E311-AC16-003048678B7C.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/8A45DEC0-18B6-E311-B9BB-0025905A60B6.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/B4BA04DD-EAB6-E311-B255-0025905A6136.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/C03E8E1C-15B6-E311-A7FE-0025905A48F2.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/ECE6D91F-15B6-E311-80B2-0025905A6082.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValTTbar_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/3E806F9A-4BB6-E311-A4D2-002618943935.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValTTbar_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/66797485-44B6-E311-9924-002618943939.root',
#'/store/relval/CMSSW_7_1_0_pre5/RelValTTbar_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/B4F97AB1-25B6-E311-A16B-003048FFD760.root'

'/store/relval/CMSSW_7_1_0_pre5/RelValZTT_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/8E8B2309-E6B6-E311-A8C6-002618943896.root'
    )
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step3 nevts:100'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = cms.untracked.vstring("keep *_*_*_*",
                                           "drop *_particleFlowRecHit*_*_RECO"),
    fileName = cms.untracked.string('file:reco.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RECO')
    )
)


process.load("RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi")
process.load("RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi")
process.load("RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHEHO_cfi")
process.load("RecoParticleFlow.PFTracking.particleFlowTrack_cff")
process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cff")

process.hcalSeeds = cms.EDProducer('PFSeedSelector',
                                      src = cms.InputTag('particleFlowRecHitHBHEHO')
)                                      

process.arbor = cms.EDProducer('PFArborLinker',
                                      src = cms.InputTag('hcalSeeds')
)                                      

# Path and EndPath definitions
process.reconstruction_step = cms.Path(process.pfTrack+process.particleFlowCluster+process.particleFlowRecHitHF+process.particleFlowClusterHF+process.particleFlowRecHitHBHEHO+process.hcalSeeds+process.particleFlowSimParticle+process.arbor)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)


# Schedule definition
process.schedule = cms.Schedule(process.reconstruction_step,process.RECOSIMoutput_step)


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')
