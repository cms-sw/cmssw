import FWCore.ParameterSet.Config as cms

# Track selector
from Validation.RecoMuon.NewSelectors_cff import *

# TrackAssociation
from SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi import *
import SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi

trackAssociatorByHits = SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone()

onlineTrackAssociatorByHits = SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone()
onlineTrackAssociatorByHits.ThreeHitTracksAreSpecial = False

# select probeTracks from generalTracks
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
probeTracks = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
#probeTracks.quality = cms.vstring('highPurity')
#probeTracks.quality = cms.vstring('loose'),      # default
probeTracks.tip = 3.5
probeTracks.lip = 30.
probeTracks.ptMin = 4.0
probeTracks.minRapidity = -2.4
probeTracks.maxRapidity = 2.4
probeTracks_seq = cms.Sequence( probeTracks )

#
# quickTrackAssociatorByHits on probeTracks used as monitor wrt MuonAssociatorByHits
#
tpToTkmuTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByHits'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
#    label_tr = cms.InputTag('generalTracks')
    label_tr = cms.InputTag('probeTracks')
)

#
# Configuration for Seed track extractor
#
import SimMuon.MCTruth.SeedToTrackProducer_cfi
seedsOfSTAmuons = SimMuon.MCTruth.SeedToTrackProducer_cfi.SeedToTrackProducer.clone()
seedsOfSTAmuons.L2seedsCollection = cms.InputTag("ancientMuonSeed")
seedsOfSTAmuons_seq = cms.Sequence( seedsOfSTAmuons )

seedsOfDisplacedSTAmuons = SimMuon.MCTruth.SeedToTrackProducer_cfi.SeedToTrackProducer.clone()
seedsOfDisplacedSTAmuons.L2seedsCollection = cms.InputTag("displacedMuonSeeds")
seedsOfDisplacedSTAmuons_seq = cms.Sequence( seedsOfDisplacedSTAmuons )

#
# MuonAssociatorByHits used for all track collections
#
import SimMuon.MCTruth.NewMuonAssociatorByHits_cfi
MABH = SimMuon.MCTruth.NewMuonAssociatorByHits_cfi.NewMuonAssociatorByHits.clone()
# DEFAULTS ###################################
#    EfficiencyCut_track = cms.double(0.),
#    PurityCut_track = cms.double(0.),
#    EfficiencyCut_muon = cms.double(0.),
#    PurityCut_muon = cms.double(0.),
#    includeZeroHitMuons = cms.bool(True),
#    acceptOneStubMatchings = cms.bool(False),
##############################################
MABH.EfficiencyCut_track = 0.5
MABH.PurityCut_track = 0.75
#MABH.EfficiencyCut_muon = 0.5
MABH.EfficiencyCut_muon = 0.     # for high pt muons this is a better choice
MABH.PurityCut_muon = 0.75
MABH.includeZeroHitMuons = False
#
MABHhlt = MABH.clone()
MABHhlt.DTrechitTag = 'hltDt1DRecHits'
MABHhlt.ignoreMissingTrackCollection = True
################################################

NEWtpToTkMuonAssociation = MABH.clone()
#tpToTkMuonAssociation.tracksTag = 'generalTracks'
NEWtpToTkMuonAssociation.tracksTag ='probeTracks'
NEWtpToTkMuonAssociation.UseTracker = True
NEWtpToTkMuonAssociation.UseMuon = False

NEWtpToStaSeedAssociation = MABH.clone()
NEWtpToStaSeedAssociation.tracksTag = 'seedsOfSTAmuons'
NEWtpToStaSeedAssociation.UseTracker = False
NEWtpToStaSeedAssociation.UseMuon = True

NEWtpToStaMuonAssociation = MABH.clone()
NEWtpToStaMuonAssociation.tracksTag = 'standAloneMuons'
NEWtpToStaMuonAssociation.UseTracker = False
NEWtpToStaMuonAssociation.UseMuon = True

NEWtpToStaUpdMuonAssociation = MABH.clone()
NEWtpToStaUpdMuonAssociation.tracksTag = 'standAloneMuons:UpdatedAtVtx'
NEWtpToStaUpdMuonAssociation.UseTracker = False
NEWtpToStaUpdMuonAssociation.UseMuon = True

NEWtpToGlbMuonAssociation = MABH.clone()
NEWtpToGlbMuonAssociation.tracksTag = 'globalMuons'
NEWtpToGlbMuonAssociation.UseTracker = True
NEWtpToGlbMuonAssociation.UseMuon = True

NEWtpToStaRefitMuonAssociation = MABH.clone()
NEWtpToStaRefitMuonAssociation.tracksTag = 'refittedStandAloneMuons'
NEWtpToStaRefitMuonAssociation.UseTracker = False
NEWtpToStaRefitMuonAssociation.UseMuon = True

NEWtpToStaRefitUpdMuonAssociation = MABH.clone()
NEWtpToStaRefitUpdMuonAssociation.tracksTag = 'refittedStandAloneMuons:UpdatedAtVtx'
NEWtpToStaRefitUpdMuonAssociation.UseTracker = False
NEWtpToStaRefitUpdMuonAssociation.UseMuon = True

NEWtpToDisplacedTrkMuonAssociation = MABH.clone()
NEWtpToDisplacedTrkMuonAssociation.tracksTag = 'displacedTracks'
NEWtpToDisplacedTrkMuonAssociation.UseTracker = True
NEWtpToDisplacedTrkMuonAssociation.UseMuon = False

NEWtpToDisplacedStaSeedAssociation = MABH.clone()
NEWtpToDisplacedStaSeedAssociation.tracksTag = 'seedsOfDisplacedSTAmuons'
NEWtpToDisplacedStaSeedAssociation.UseTracker = False
NEWtpToDisplacedStaSeedAssociation.UseMuon = True

NEWtpToDisplacedStaMuonAssociation = MABH.clone()
NEWtpToDisplacedStaMuonAssociation.tracksTag = 'displacedStandAloneMuons'
NEWtpToDisplacedStaMuonAssociation.UseTracker = False
NEWtpToDisplacedStaMuonAssociation.UseMuon = True

NEWtpToDisplacedGlbMuonAssociation = MABH.clone()
NEWtpToDisplacedGlbMuonAssociation.tracksTag = 'displacedGlobalMuons'
NEWtpToDisplacedGlbMuonAssociation.UseTracker = True
NEWtpToDisplacedGlbMuonAssociation.UseMuon = True

NEWtpToTevFirstMuonAssociation = MABH.clone()
NEWtpToTevFirstMuonAssociation.tracksTag = 'tevMuons:firstHit'
NEWtpToTevFirstMuonAssociation.UseTracker = True
NEWtpToTevFirstMuonAssociation.UseMuon = True

NEWtpToTevPickyMuonAssociation = MABH.clone()
NEWtpToTevPickyMuonAssociation.tracksTag = 'tevMuons:picky'
NEWtpToTevPickyMuonAssociation.UseTracker = True
NEWtpToTevPickyMuonAssociation.UseMuon = True

NEWtpToTevDytMuonAssociation = MABH.clone()
NEWtpToTevDytMuonAssociation.tracksTag = 'tevMuons:dyt'
NEWtpToTevDytMuonAssociation.UseTracker = True
NEWtpToTevDytMuonAssociation.UseMuon = True

NEWtpToL3TkMuonAssociation = MABHhlt.clone()
NEWtpToL3TkMuonAssociation.tracksTag = 'hltL3TkTracksFromL2'
NEWtpToL3TkMuonAssociation.UseTracker = True
NEWtpToL3TkMuonAssociation.UseMuon = False

NEWtpToL2MuonAssociation = MABHhlt.clone()
NEWtpToL2MuonAssociation.tracksTag = 'hltL2Muons'
NEWtpToL2MuonAssociation.UseTracker = False
NEWtpToL2MuonAssociation.UseMuon = True

NEWtpToL2UpdMuonAssociation = MABHhlt.clone()
NEWtpToL2UpdMuonAssociation.tracksTag = 'hltL2Muons:UpdatedAtVtx'
NEWtpToL2UpdMuonAssociation.UseTracker = False
NEWtpToL2UpdMuonAssociation.UseMuon = True

NEWtpToL3MuonAssociation = MABHhlt.clone()
NEWtpToL3MuonAssociation.tracksTag = 'hltL3Muons'
NEWtpToL3MuonAssociation.UseTracker = True
NEWtpToL3MuonAssociation.UseMuon = True

#
# COSMICS reco
#
# 2-legs cosmics reco: simhits can be twice the reconstructed ones in any single leg
# (Quality cut have to be set at 0.25, purity cuts can stay at default value 0.75)
# T.B.D. upper and lower leg should be analyzed separately 
#
NEWtpToTkCosmicSelMuonAssociation = MABH.clone()
NEWtpToTkCosmicSelMuonAssociation.tracksTag = 'ctfWithMaterialTracksP5LHCNavigation'
NEWtpToTkCosmicSelMuonAssociation.UseTracker = True
NEWtpToTkCosmicSelMuonAssociation.UseMuon = False
NEWtpToTkCosmicSelMuonAssociation.EfficiencyCut_track = 0.25

NEWtpToStaCosmicSelMuonAssociation = MABH.clone()
NEWtpToStaCosmicSelMuonAssociation.tracksTag = 'cosmicMuons'
NEWtpToStaCosmicSelMuonAssociation.UseTracker = False
NEWtpToStaCosmicSelMuonAssociation.UseMuon = True
NEWtpToStaCosmicSelMuonAssociation.EfficiencyCut_muon = 0.25

NEWtpToGlbCosmicSelMuonAssociation = MABH.clone()
NEWtpToGlbCosmicSelMuonAssociation.tracksTag = 'globalCosmicMuons'
NEWtpToGlbCosmicSelMuonAssociation.UseTracker = True
NEWtpToGlbCosmicSelMuonAssociation.UseMuon = True
NEWtpToGlbCosmicSelMuonAssociation.EfficiencyCut_track = 0.25
NEWtpToGlbCosmicSelMuonAssociation.EfficiencyCut_muon = 0.25

# 1-leg cosmics reco
NEWtpToTkCosmic1LegSelMuonAssociation = MABH.clone()
NEWtpToTkCosmic1LegSelMuonAssociation.tracksTag = 'ctfWithMaterialTracksP5'
NEWtpToTkCosmic1LegSelMuonAssociation.UseTracker = True
NEWtpToTkCosmic1LegSelMuonAssociation.UseMuon = False

NEWtpToStaCosmic1LegSelMuonAssociation = MABH.clone()
NEWtpToStaCosmic1LegSelMuonAssociation.tracksTag = 'cosmicMuons1Leg'
NEWtpToStaCosmic1LegSelMuonAssociation.UseTracker = False
NEWtpToStaCosmic1LegSelMuonAssociation.UseMuon = True

NEWtpToGlbCosmic1LegSelMuonAssociation = MABH.clone()
NEWtpToGlbCosmic1LegSelMuonAssociation.tracksTag = 'globalCosmicMuons1Leg'
NEWtpToGlbCosmic1LegSelMuonAssociation.UseTracker = True
NEWtpToGlbCosmic1LegSelMuonAssociation.UseMuon = True

#
# The full-sim association sequences
#

NewMuonAssociation_seq = cms.Sequence(
    probeTracks_seq+NEWtpToTkMuonAssociation
    +trackAssociatorByHits+tpToTkmuTrackAssociation
    +seedsOfSTAmuons_seq+NEWtpToStaSeedAssociation+NEWtpToStaMuonAssociation+NEWtpToStaUpdMuonAssociation
    +NEWtpToGlbMuonAssociation
    )

NewMuonAssociationTEV_seq = cms.Sequence(
    NEWtpToTevFirstMuonAssociation+NEWtpToTevPickyMuonAssociation+NEWtpToTevDytMuonAssociation
    )

NewMuonAssociationDisplaced_seq = cms.Sequence(
    seedsOfDisplacedSTAmuons_seq+NEWtpToDisplacedStaSeedAssociation+NEWtpToDisplacedStaMuonAssociation
    +NEWtpToDisplacedTrkMuonAssociation+NEWtpToDisplacedGlbMuonAssociation
    )

NewMuonAssociationRefit_seq = cms.Sequence(
    NEWtpToStaRefitMuonAssociation+NEWtpToStaRefitUpdMuonAssociation
    )

NewMuonAssociationCosmic_seq = cms.Sequence(
    NEWtpToTkCosmicSelMuonAssociation+ NEWtpToTkCosmic1LegSelMuonAssociation
    +NEWtpToStaCosmicSelMuonAssociation+NEWtpToStaCosmic1LegSelMuonAssociation
    +NEWtpToGlbCosmicSelMuonAssociation+NEWtpToGlbCosmic1LegSelMuonAssociation
    )

NewMuonAssociationHLT_seq = cms.Sequence(
    NEWtpToL2MuonAssociation+NEWtpToL2UpdMuonAssociation+NEWtpToL3TkMuonAssociation+NEWtpToL3MuonAssociation
    )


# fastsim has no hlt specific dt hit collection
from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    _DTrechitTag = SimMuon.MCTruth.NewMuonAssociatorByHits_cfi.NewMuonAssociatorByHits.DTrechitTag
    NEWtpToL3TkMuonAssociation.DTrechitTag = _DTrechitTag
    NEWtpToL2MuonAssociation.DTrechitTag = _DTrechitTag
    NEWtpToL2UpdMuonAssociation.DTrechitTag = _DTrechitTag
    NEWtpToL3MuonAssociation.DTrechitTag = _DTrechitTag
