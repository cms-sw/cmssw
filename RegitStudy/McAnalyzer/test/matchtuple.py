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
      fileNames = cms.untracked.vstring("file:pythiagun_jpsi_embd_step2.root"),
      # fileNames = cms.untracked.vstring("/store/user/echapon/pythiagun_bJpsi_Pt020_STARTHI53_V27_gen_20140504/pythiagun_bJpsi_Pt020_STARTHI53_V27_step2_regit_20140504/a1f0eb4c124f02403cb577f655ffcec3/pythiagun_bJpsi_step2_regit_100_1_YnB.root"),
      # fileNames = cms.untracked.vstring("/store/user/echapon/pythiagun_bJpsi_Pt020_STARTHI53_V27_gen_20140504/pythiagun_bJpsi_Pt020_STARTHI53_V27_step2_pp_20140504/e6f1f3dc82ecfb4d8356cb8482fd3a9a/pythiagun_bJpsi_step2_pp_100_1_PSX.root"),
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

#import SimMuon.MCTruth.MuonAssociatorByHits_cfi
process.glbMuonAssociatorByHits = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
process.glbMuonAssociatorByHits.tracksTag                     = cms.InputTag('globalMuons')
process.glbMuonAssociatorByHits.UseTracker                    = True
process.glbMuonAssociatorByHits.UseMuon                       = True
process.glbMuonAssociatorByHits.PurityCut_track               = 0.75
process.glbMuonAssociatorByHits.EfficiencyCut_track           = 0.
process.glbMuonAssociatorByHits.PurityCut_muon                = 0.75
process.glbMuonAssociatorByHits.EfficiencyCut_muon            = 0.
process.glbMuonAssociatorByHits.includeZeroHitMuons           = False
process.glbMuonAssociatorByHits.acceptOneStubMatchings        = False

#import SimMuon.MCTruth.MuonAssociatorByHits_cfi
process.reGlbMuonAssociatorByHits = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
process.reGlbMuonAssociatorByHits.tracksTag                     = cms.InputTag('reglobalMuons')
process.reGlbMuonAssociatorByHits.UseTracker                    = True
process.reGlbMuonAssociatorByHits.UseMuon                       = True
process.reGlbMuonAssociatorByHits.PurityCut_track               = 0.75
process.reGlbMuonAssociatorByHits.EfficiencyCut_track           = 0.
process.reGlbMuonAssociatorByHits.PurityCut_muon                = 0.75
process.reGlbMuonAssociatorByHits.EfficiencyCut_muon            = 0.
process.reGlbMuonAssociatorByHits.includeZeroHitMuons           = False
process.reGlbMuonAssociatorByHits.acceptOneStubMatchings        = False

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

#--------------------
process.TFileService = cms.Service("TFileService", 
      fileName = cms.string('ntuples.root')
      # fileName = cms.string('/tmp/camelia/matcher_zmumuflat_start.root')
      #fileName = cms.string('/tmp/camelia/bjpsi36_gun500_onTopOfRegular_tune1SmallCilSmallRegLastIter_tune2.root')
      #fileName = cms.string('/tmp/camelia/jpsipt03_gun2000_onTopOfRegular_tune1SmallCilSmallRegLastIter_tune2.root')
      #  fileName = cms.string('/tmp/camelia/upspt05_gun500_tryDiffSet.root')
      )

# Schedule definition
# do regit

# process.load("RecoHI.HiMuonAlgos.HiReRecoMuon_cff")
# process.raw2digi       = cms.Path(process.RawToDigi)

# process.load("RecoHI.HiTracking.hiIterTracking_cff")
# process.trackerRecHits = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits)
# process.hiTrackReco   = cms.Path(process.heavyIonTracking*process.hiIterTracking)
#iteerative tracking

# process.regit          = cms.Path(process.reMuonRecoPbPb)#TrackRecoPbPb)
process.mumatch_step   = cms.Path(process.glbMuonAssociatorByHits+process.reGlbMuonAssociatorByHits)
process.p8             = cms.Path(process.mcmatchanalysis)
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

