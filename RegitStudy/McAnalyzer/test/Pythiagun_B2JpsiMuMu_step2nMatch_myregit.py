# Auto generated configuration file
# using: 
# Revision: 1.381.2.28 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step2 --filein file:pythiagun_bJpsi_gen.root --fileout file:pythiagun_bJpsi_step2.root --mc --eventcontent RECODEBUG --datatier GEN-SIM-RECODEBUG --conditions STARTHI53_V27::All --step RAW2DIGI,L1Reco,RECO --scenario HeavyIons --python_filename Pythiagun_B2JpsiMuMu_step2.py --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('file:pythiagun_jpsi_gen.root')
)

process.options = cms.untracked.PSet(

)

##################################################################################
# # Some Services
# process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
#                                         ignoreTotal=cms.untracked.int32(0),
#                                         oncePerEventMode = cms.untracked.bool(False)
#                                         )
# process.Timing = cms.Service("Timing")
# process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
# process.MessageLogger.warnings = cms.untracked.PSet(
#       threshold = cms.untracked.string('WARNING')
#       )

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.28 $'),
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

# process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
#     splitLevel = cms.untracked.int32(0),
#     eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
#     outputCommands = process.FEVTDEBUGEventContent.outputCommands,
#     fileName = cms.untracked.string('file:pythiagun_bJpsi_step2_regit.root'),
#     dataset = cms.untracked.PSet(
#         filterName = cms.untracked.string(''),
#         dataTier = cms.untracked.string('GEN-SIM-RECODEBUG')
#     )
# )

#################################################################################"standAloneMuons","UpdatedAtVtx"
#--------------- matching:
import SimMuon.MCTruth.MuonAssociatorByHits_cfi
process.staMuonAssociatorByHits = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
process.staMuonAssociatorByHits.tracksTag                     = cms.InputTag("standAloneMuons","UpdatedAtVtx") #
process.staMuonAssociatorByHits.UseTracker                    = False
process.staMuonAssociatorByHits.UseMuon                       = True
process.staMuonAssociatorByHits.PurityCut_muon                = 0.75
process.staMuonAssociatorByHits.EfficiencyCut_muon            = 0.
process.staMuonAssociatorByHits.includeZeroHitMuons           = False

process.glbMuonAssociatorByHits = process.staMuonAssociatorByHits.clone()
process.glbMuonAssociatorByHits.tracksTag                     = cms.InputTag('globalMuons')
process.glbMuonAssociatorByHits.UseTracker                    = True
process.glbMuonAssociatorByHits.PurityCut_track               = 0.75
process.glbMuonAssociatorByHits.EfficiencyCut_track           = 0.
process.glbMuonAssociatorByHits.acceptOneStubMatchings        = False

process.reGlbMuonAssociatorByHits = process.glbMuonAssociatorByHits.clone()
process.reGlbMuonAssociatorByHits.tracksTag                     = cms.InputTag('reglobalMuons')

process.genTrkAssociatorByHits = process.glbMuonAssociatorByHits.clone()
process.genTrkAssociatorByHits.UseMuon                        = False
process.genTrkAssociatorByHits.tracksTag                      = cms.InputTag('hiGeneralTracks')

process.genAndRegitTrkAssociatorByHits = process.genTrkAssociatorByHits.clone()
process.genAndRegitTrkAssociatorByHits.tracksTag                      = cms.InputTag('hiGeneralAndRegitMuTracks')

###################################################################################
process.mcmatchanalysis = cms.EDAnalyzer("McMatchTrackAnalyzer",
                                         doHLT              = cms.bool(False),
                                         doHiEmbedding      = cms.bool(False),
                                         doParticleGun      = cms.bool(True),
                                         doReco2Sim         = cms.bool(True),
                                         doSim2Reco         = cms.bool(True),
                                         matchPair          = cms.bool(True),
                                         matchSingle        = cms.bool(True),
                                         pdgPair            = cms.int32(443),
                                         pdgSingle          = cms.int32(13),
                                         type1Tracks        = cms.untracked.InputTag("globalMuons"),
                                         type1MapTag        = cms.untracked.InputTag("glbMuonAssociatorByHits"),
                                         type2Tracks        = cms.untracked.InputTag("reglobalMuons"),
                                         type2MapTag        = cms.untracked.InputTag("reGlbMuonAssociatorByHits"),
                                         simTracks          = cms.untracked.InputTag("mix","MergedTrackTruth"),
                                         verticesTag        = cms.untracked.InputTag("hiSelectedVertex"),
                                         tagForHitTpMatching= cms.untracked.InputTag("simHitTPAssocProducer")
                                         )

process.mcmatchanalysis_generaltracks = process.mcmatchanalysis.clone()
process.mcmatchanalysis_generaltracks.type1Tracks = "hiGeneralTracks"
process.mcmatchanalysis_generaltracks.type1MapTag = "genTrkAssociatorByHits"
process.mcmatchanalysis_generaltracks.type2Tracks = "hiGeneralAndRegitMuTracks"
process.mcmatchanalysis_generaltracks.type2MapTag = "genAndRegitTrkAssociatorByHits"

#--------------------
process.TFileService = cms.Service("TFileService", 
      fileName = cms.string('ntuples.root')
      )

#-------------------
# simHit - trackingparticle map
process.load('SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi')

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:starthi_HIon', '')
# process.GlobalTag = GlobalTag(process.GlobalTag, 'START72_V1::All', '')

# Path and EndPath definitions
process.mumatch_step   = cms.Path(process.glbMuonAssociatorByHits)
      # +process.reGlbMuonAssociatorByHits
      # +process.genTrkAssociatorByHits)
      # +process.genAndRegitTrkAssociatorByHits)
# process.p8             = cms.Path(process.mcmatchanalysis+process.mcmatchanalysis_generaltracks)
process.p8             = cms.Path(process.simHitTPAssocProducer+process.mcmatchanalysis)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstructionHeavyIons)
process.endjob_step = cms.EndPath(process.endOfProcess)
# process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
# process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.mumatch_step,process.p8,process.endjob_step,process.FEVTDEBUGoutput_step)
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.mumatch_step,process.p8,process.endjob_step)

