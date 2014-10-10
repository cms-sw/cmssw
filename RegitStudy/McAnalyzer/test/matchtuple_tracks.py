import FWCore.ParameterSet.Config as cms

process = cms.Process("USER")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# process.GlobalTag = 'STARTHI43_V27::All'
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'STARTHI53_V27::All', '')

# process.load('RecoHI.HiCentralityAlgos.CentralityBin_cfi')

##################################################################################
# setup 'standard'  options
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

# Input source
process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring("file:/tmp/echapon/pythiagun_bJpsi_step2_regit.root"),
      noEventSort = cms.untracked.bool(True),
      duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
      skipEvents=cms.untracked.uint32(0),
      inputCommands = cms.untracked.vstring('keep *',
         #   'drop *_muons*_*_RECO',
         #   'drop *_globalMuons_*_RECO',
         #   'drop *_*Tracks_*_RECO',
         #   'drop *_*Vertex_*_RECO'
         )
      )

process.output = cms.OutputModule("PoolOutputModule",
                                  splitLevel = cms.untracked.int32(0),
                                  outputCommands = cms.untracked.vstring('keep *_*_*_*',
                                                                        # 'keep *_remuons_*_*',
                                                                        # 'keep *_reglobalMuons_*_*',
                                                                        # 'keep *_hiGeneralTracks_*_*',
                                                                        # 'keep *_hiSelectedTracks_*_*',
                                                                        # 'keep *_hiGeneralAndRegitMuTracks_*_*'
                                                                         ),
                                 # fileName = cms.untracked.string('/tmp/camelia/regit_bjpsigun_3globalRegit.root')
                                 fileName = cms.untracked.string('file:edm.root')
                                  )


process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(default = cms.untracked.PSet(limit = cms.untracked.int32(0) ## kill all messages in the log
                                                                                           )
                                                              ),
                                    destinations = cms.untracked.vstring('cout')
                                    )

##################################################################################
# Some Services
process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
                                        ignoreTotal=cms.untracked.int32(0),
                                        oncePerEventMode = cms.untracked.bool(False)
                                        )
process.Timing = cms.Service("Timing")
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

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
                                       #  type2Tracks        = cms.untracked.InputTag("standAloneMuons","UpdatedAtVtx"),
                                        # type2MapTag        = cms.untracked.InputTag("staMuonAssociatorByHits"),
                                         simTracks          = cms.untracked.InputTag("mergedtruth","MergedTrackTruth"),
                                         verticesTag        = cms.untracked.InputTag("hiSelectedVertex")
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

# Schedule definition
process.mumatch_step   = cms.Path(process.glbMuonAssociatorByHits
      +process.reGlbMuonAssociatorByHits
      +process.genTrkAssociatorByHits
      +process.genAndRegitTrkAssociatorByHits)
process.p8             = cms.Path(process.mcmatchanalysis+process.mcmatchanalysis_generaltracks)
#process.staMuonAssociatorByHits+

process.out_step       = cms.EndPath(process.output)
process.endjob_step    = cms.Path(process.endOfProcess)

#
process.schedule       = cms.Schedule(process.mumatch_step,process.p8)
process.schedule.extend([process.endjob_step,process.out_step])

# from CmsHi.Analysis2010.CommonFunctions_cff import *
# overrideCentrality(process)
process.HeavyIonGlobalParameters = cms.PSet(
    centralityVariable = cms.string("HFhits"),
    centralitySrc = cms.InputTag("hiCentrality"),
    nonDefaultGlauberModel = cms.string("Hydjet_Bass")
)  

