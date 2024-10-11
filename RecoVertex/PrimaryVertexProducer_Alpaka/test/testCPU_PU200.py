import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('PV',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '140X_mcRun3_2023_realistic_v3')

# Input files
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_14_0_0/RelValTTbarToDilepton_14TeV/GEN-SIM-RECO/PU_140X_mcRun4_realistic_v1_STD_2026D98_PU-v1/2580000/2af100a0-34ce-4b20-8e3e-399fc84d79ea.root',
    ),
    secondaryFileNames = cms.untracked.vstring(),
)

# Number of events to run
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1),
)

# Production metadata
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('PV nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition
process.FEVToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RECO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('testCPU_PU0.root'), # output file name
    outputCommands = cms.untracked.vstring('drop *','keep *_*_*_PV', 'keep *_genPUProtons_*_*'),# I.e., just drop everything and keep things in this module
    splitLevel = cms.untracked.int32(0)
)

# Endpath and output
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVToutput_step = cms.EndPath(process.FEVToutput)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)


################################
## Now the plugins themselves ##
################################


from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices

process.offlinePrimaryVertices = offlinePrimaryVertices
process.offlinePrimaryVertices.TkClusParameters.TkDAClusParameters.runInBlocks = cms.bool(True)
process.offlinePrimaryVertices.TkClusParameters.TkDAClusParameters.block_size = cms.uint32(512)
process.offlinePrimaryVertices.TkClusParameters.TkDAClusParameters.overlap_frac = cms.double(0.5)
process.offlinePrimaryVertices.vertexCollections = cms.VPSet(
       [cms.PSet(label=cms.string(""),
           algorithm=cms.string("WeightedMeanFitter"),
           chi2cutoff = cms.double(2.5),
           minNdof=cms.double(0.0),
           useBeamConstraint = cms.bool(False),
           maxDistanceToBeam = cms.double(1.0)
        ),
        cms.PSet(label=cms.string("WithBS"),
            algorithm = cms.string('WeightedMeanFitter'),
            minNdof=cms.double(0.0),
            chi2cutoff = cms.double(2.5),
            useBeamConstraint = cms.bool(True),
            maxDistanceToBeam = cms.double(1.0)
        )])


process.options.wantSummary = True

###################################
## Last, organize paths and exec ##
###################################

process.vertexing_task = cms.EndPath(process.offlinePrimaryVertices)
process.schedule = cms.Schedule(process.vertexing_task)
process.schedule.extend([process.endjob_step,process.FEVToutput_step])
