import FWCore.ParameterSet.Config as cms

# TrackingParticle selectors
from Validation.RecoMuon.selectors_cff import *
# reco::Track selectors 
from Validation.RecoMuon.track_selectors_cff import *

# quickTrackAssociatorByHits on probeTracks used as monitor wrt MuonAssociatorByHits
import SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi
trackAssociatorByHits = SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone()

from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import trackingParticleRecoTrackAsssociation as _trackingParticleRecoTrackAsssociation
tpToTkmuTrackAssociation = _trackingParticleRecoTrackAsssociation.clone(
    associator = 'trackAssociatorByHits',
#    label_tr = 'generalTracks',
    label_tr = 'probeTracks'
)

#
# MuonAssociatorByHits used for all track collections
#
import SimMuon.MCTruth.MuonAssociatorByHits_cfi
MABH = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone(
# DEFAULTS ###################################
#    EfficiencyCut_track = 0.,
#    PurityCut_track = 0.,
#    EfficiencyCut_muon = 0.,
#    PurityCut_muon = 0.,
#    includeZeroHitMuons = True,
#    acceptOneStubMatchings = False,
#    rejectBadGlobal = True,
#    tpTag = "mix","MergedTrackTruth",
#    tpRefVector = False
##############################################
    EfficiencyCut_track = 0.5,
    PurityCut_track = 0.75,
    EfficiencyCut_muon = 0.5,
    PurityCut_muon = 0.75,
    includeZeroHitMuons = False,
    tpTag = ("TPmu"),
    tpRefVector = True
##############################################
)
tpToTkMuonAssociation = MABH.clone(
    #tracksTag = 'generalTracks',
    tracksTag ='probeTracks',
    UseTracker = True,
    UseMuon = False,
    tpTag = ("TPtrack")
)
tpToStaSeedAssociation = MABH.clone(
    tracksTag = 'seedsOfSTAmuons',
    UseTracker = False,
    UseMuon = True,
    EfficiencyCut_muon = 0.
)
tpToStaMuonAssociation = MABH.clone(
    tracksTag = 'standAloneMuons',
    UseTracker = False,
    UseMuon = True
)
tpToStaUpdMuonAssociation = MABH.clone(
    tracksTag = 'standAloneMuons:UpdatedAtVtx',
    UseTracker = False,
    UseMuon = True
)
tpToGlbMuonAssociation = MABH.clone(
    tracksTag = 'globalMuons',
    UseTracker = True,
    UseMuon = True
)
tpToStaRefitMuonAssociation = MABH.clone(
    tracksTag = 'refittedStandAloneMuons',
    UseTracker = False,
    UseMuon = True
)
tpToStaRefitUpdMuonAssociation = MABH.clone(
    tracksTag = 'refittedStandAloneMuons:UpdatedAtVtx',
    UseTracker = False,
    UseMuon = True
)
tpToDisplacedTrkMuonAssociation = MABH.clone(
    tracksTag = 'displacedTracks',
    UseTracker = True,
    UseMuon = False,
    tpTag = ("TPtrack")
)
tpToDisplacedStaSeedAssociation = MABH.clone(
    tracksTag = 'seedsOfDisplacedSTAmuons',
    UseTracker = False,
    UseMuon = True,
    EfficiencyCut_muon = 0.
)
tpToDisplacedStaMuonAssociation = MABH.clone(
    tracksTag = 'displacedStandAloneMuons',
    UseTracker = False,
    UseMuon = True
)
tpToDisplacedGlbMuonAssociation = MABH.clone(
    tracksTag = 'displacedGlobalMuons',
    UseTracker = True,
    UseMuon = True
)
tpToTevFirstMuonAssociation = MABH.clone(
    tracksTag = 'tevMuons:firstHit',
    UseTracker = True,
    UseMuon = True,
    EfficiencyCut_muon = 0.
)
tpToTevPickyMuonAssociation = MABH.clone(
    tracksTag = 'tevMuons:picky',
    UseTracker = True,
    UseMuon = True,
    EfficiencyCut_muon = 0.
)
tpToTevDytMuonAssociation = MABH.clone(
    tracksTag = 'tevMuons:dyt',
    UseTracker = True,
    UseMuon = True,
    EfficiencyCut_muon = 0.,
    rejectBadGlobal = False
)
# tuneP (GlobalMuons with TuneP definition)
tpToTunePMuonAssociation = MABH.clone(
    tracksTag = 'tunepMuonTracks',
    UseTracker = True,
    UseMuon = True,
    EfficiencyCut_muon = 0.,
    rejectBadGlobal = False
)
# PFMuons
tpToPFMuonAssociation = MABH.clone(
    tracksTag = 'pfMuonTracks',
    UseTracker = True,
    UseMuon = True,
    tpTag = ("TPpfmu"),
    EfficiencyCut_muon = 0.,
    rejectBadGlobal = False
)
# all offline reco::Muons with loose cuts
# note in this case muons can be of any type: set UseTracker=UseMuon=true and rejectBadGlobal=false,
# then define the logic in the muon track producer, run beforehand.
tpTorecoMuonMuonAssociation = MABH.clone(
    tracksTag = 'recoMuonTracks',
    UseTracker = True,
    UseMuon = True,
    EfficiencyCut_track = 0.,
    EfficiencyCut_muon = 0.,
    # matching to a skimmed TP collection needs a purity cut to avoid pathological cases
    #PurityCut_track = 0.,
    #PurityCut_muon = 0.,
    includeZeroHitMuons = True,
    rejectBadGlobal = False
)
# ME0Muons
tpToME0MuonMuonAssociation = MABH.clone(
    tracksTag = 'extractMe0Muons',
    UseTracker = True,
    UseMuon = False
)
# GEMmuons
tpToGEMMuonMuonAssociation = MABH.clone(
    tracksTag = 'extractGemMuons',
    UseTracker = True,
    UseMuon = False
)
# === HLT muon tracks 
#
MABHhlt = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone(
# DEFAULTS ###################################
#    EfficiencyCut_track = cms.double(0.), # backup solution as UseGrouped/UseSplitting are always assumed to be true
#    EfficiencyCut_muon = cms.double(0.),  # | loose matching requests for triggering
#    includeZeroHitMuons = cms.bool(True), # |
#    acceptOneStubMatchings = cms.bool(False),
#    rejectBadGlobal = cms.bool(True),
##############################################
    PurityCut_track = 0.75,
    PurityCut_muon = 0.75,
    DTrechitTag = 'hltDt1DRecHits',
    ignoreMissingTrackCollection = True,
    tpTag = ("TPmu"),
    tpRefVector = True
)
##############################################

tpToL2MuonAssociation = MABHhlt.clone(
    tracksTag = 'hltL2Muons',
    UseTracker = False,
    UseMuon = True
)
tpToL2UpdMuonAssociation = MABHhlt.clone(
    tracksTag = 'hltL2Muons:UpdatedAtVtx',
    UseTracker = False,
    UseMuon = True
)
tpToL3OITkMuonAssociation = MABHhlt.clone(
    tracksTag = 'hltIterL3OIMuonTrackSelectionHighPurity',
    UseTracker = True,
    UseMuon = False
)
tpToL3TkMuonAssociation = MABHhlt.clone(
    tracksTag = 'hltIterL3MuonMerged',
    UseTracker = True,
    UseMuon = False
)
tpToL3FromL1TkMuonAssociation = MABHhlt.clone(
    tracksTag = 'hltIterL3MuonAndMuonFromL1Merged',
    UseTracker = True,
    UseMuon = False
)
tpToL0L3FromL1TkMuonAssociation = MABHhlt.clone(
    tracksTag = 'hltIter0IterL3FromL1MuonTrackSelectionHighPurity',
    UseTracker = True,
    UseMuon = False
)
tpToL3GlbMuonAssociation = MABHhlt.clone(
    tracksTag = 'hltIterL3GlbMuon',
    UseTracker = True,
    UseMuon = True
)
tpToL3NoIDMuonAssociation = MABHhlt.clone(
    tracksTag = 'hltIterL3MuonsNoIDTracks',
    UseTracker = True,
    UseMuon = True,
    rejectBadGlobal = False
)
tpToL3MuonAssociation = MABHhlt.clone(
    tracksTag = 'hltIterL3MuonsTracks',
    UseTracker = True,
    UseMuon = True,
    rejectBadGlobal = False
)

#
# The Phase-2 associators
#

# L2 standalone muon seeds
Phase2tpToL2SeedAssociation = MABHhlt.clone(
    tracksTag = "hltPhase2L2MuonSeedTracks",
    UseTracker = False,
    UseMuon = True
)
# L2 standalone muons 
Phase2tpToL2MuonAssociation = MABHhlt.clone(
    tracksTag = 'hltL2MuonsFromL1TkMuon',
    UseTracker = False,
    UseMuon = True
)
# L2 standalone muons updated at vertex
Phase2tpToL2MuonUpdAssociation = MABHhlt.clone(
    tracksTag = 'hltL2MuonsFromL1TkMuon:UpdatedAtVtx',
    UseTracker = False,
    UseMuon = True
)
# L3 IO inner tracks
Phase2tpToL3IOTkAssociation = MABHhlt.clone(
    tracksTag = 'hltIter2Phase2L3FromL1TkMuonMerged',
    UseTracker = True,
    UseMuon = False
)
# L3 OI inner tracks
Phase2tpToL3OITkAssociation = MABHhlt.clone(
    tracksTag = 'hltPhase2L3OIMuonTrackSelectionHighPurity',
    UseTracker = True,
    UseMuon = False
)
# L2 muons to reuse (IO first only)
Phase2tpToL2MuonToReuseAssociation = MABHhlt.clone(
    tracksTag = 'hltPhase2L3MuonFilter:L2MuToReuse',
    UseTracker = False,
    UseMuon = True
)
# L3 IO inner tracks filtered (IO first only)
Phase2tpToL3IOTkFilteredAssociation = MABHhlt.clone(
    tracksTag = 'hltPhase2L3MuonFilter:L3IOTracksFiltered',
    UseTracker = True,
    UseMuon = False
)
# L3 OI inner tracks filtered (OI first only)
Phase2tpToL3OITkFilteredAssociation = MABHhlt.clone(
    tracksTag = 'hltPhase2L3MuonFilter:L3OITracksFiltered',
    UseTracker = True,
    UseMuon = False
)
# L3 inner tracks merged
Phase2tpToL3TkMergedAssociation = MABHhlt.clone(
    tracksTag = 'hltPhase2L3MuonMerged',
    UseTracker = True,
    UseMuon = False
)
# L3 global muons
Phase2tpToL3GlbMuonMergedAssociation = MABHhlt.clone(
    tracksTag = 'hltPhase2L3GlbMuon',
    UseTracker = True,
    UseMuon = True
)
# L3 Muons no ID (tracks)
Phase2tpToL3MuonNoIdAssociation = MABHhlt.clone(
    tracksTag = 'hltPhase2L3MuonNoIdTracks',
    UseTracker = True,
    UseMuon = True,
    rejectBadGlobal = False
)
# L3 Muons ID (tracks)
Phase2tpToL3MuonIdAssociation = MABHhlt.clone(
    tracksTag = 'hltPhase2L3MuonIdTracks',
    UseTracker = True,
    UseMuon = True,
    rejectBadGlobal = False
)

#
# COSMICS reco
#

MABHcosmic = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone(
# DEFAULTS ###################################
#    acceptOneStubMatchings = False,
#    rejectBadGlobal = True,
#    tpTag = "mix:MergedTrackTruth",
#    tpRefVector = False,
###############################################
    EfficiencyCut_track = 0.5,
    PurityCut_track = 0.75,
    EfficiencyCut_muon = 0.5,
    PurityCut_muon = 0.75,
    includeZeroHitMuons = False
)
################################################
#
# 2-legs cosmics reco: simhits can be twice the reconstructed ones in any single leg
# (Quality cut have to be set at 0.25, purity cuts can stay at default value 0.75)
# T.B.D. upper and lower leg should be analyzed separately 
#
tpToTkCosmicSelMuonAssociation = MABHcosmic.clone(
    tracksTag = 'ctfWithMaterialTracksP5LHCNavigation',
    UseTracker = True,
    UseMuon = False,
    EfficiencyCut_track = 0.25
)
tpToStaCosmicSelMuonAssociation = MABHcosmic.clone(
    tracksTag = 'cosmicMuons',
    UseTracker = False,
    UseMuon = True,
    EfficiencyCut_muon = 0.25
)
tpToGlbCosmicSelMuonAssociation = MABHcosmic.clone(
    tracksTag = 'globalCosmicMuons',
    UseTracker = True,
    UseMuon = True,
    EfficiencyCut_track = 0.25,
    EfficiencyCut_muon = 0.25
)
# 1-leg cosmics reco
tpToTkCosmic1LegSelMuonAssociation = MABHcosmic.clone(
    tracksTag = 'ctfWithMaterialTracksP5',
    UseTracker = True,
    UseMuon = False
)
tpToStaCosmic1LegSelMuonAssociation = MABHcosmic.clone(
    tracksTag = 'cosmicMuons1Leg',
    UseTracker = False,
    UseMuon = True
)
tpToGlbCosmic1LegSelMuonAssociation = MABHcosmic.clone(
    tracksTag = 'globalCosmicMuons1Leg',
    UseTracker = True,
    UseMuon = True
)

#
# Offline Muon Association sequences
#

muonAssociation_seq = cms.Sequence(
    probeTracks_seq+tpToTkMuonAssociation
    +trackAssociatorByHits+tpToTkmuTrackAssociation
    +seedsOfSTAmuons_seq+tpToStaSeedAssociation+tpToStaMuonAssociation+tpToStaUpdMuonAssociation
    +tpToGlbMuonAssociation
    +pfMuonTracks_seq+tpToPFMuonAssociation
    +recoMuonTracks_seq+tpTorecoMuonMuonAssociation
)

muonAssociationTEV_seq = cms.Sequence(
    tpToTevFirstMuonAssociation+tpToTevPickyMuonAssociation+tpToTevDytMuonAssociation
    +tunepMuonTracks_seq+tpToTunePMuonAssociation
)

muonAssociationDisplaced_seq = cms.Sequence(
    seedsOfDisplacedSTAmuons_seq+tpToDisplacedStaSeedAssociation+tpToDisplacedStaMuonAssociation
    +tpToDisplacedTrkMuonAssociation+tpToDisplacedGlbMuonAssociation
)

muonAssociationRefit_seq = cms.Sequence(
    tpToStaRefitMuonAssociation+tpToStaRefitUpdMuonAssociation
)

muonAssociationCosmic_seq = cms.Sequence(
    tpToTkCosmicSelMuonAssociation+ tpToTkCosmic1LegSelMuonAssociation
    +tpToStaCosmicSelMuonAssociation+tpToStaCosmic1LegSelMuonAssociation
    +tpToGlbCosmicSelMuonAssociation+tpToGlbCosmic1LegSelMuonAssociation
)

#
# The HLT association sequence
#

muonAssociationHLT_seq = cms.Sequence(
    tpToL2MuonAssociation+tpToL2UpdMuonAssociation
    +tpToL3OITkMuonAssociation+tpToL3TkMuonAssociation+tpToL3FromL1TkMuonAssociation+tpToL0L3FromL1TkMuonAssociation
    +tpToL3GlbMuonAssociation
    +hltIterL3MuonsNoIDTracks_seq+tpToL3NoIDMuonAssociation
    +hltIterL3MuonsTracks_seq+tpToL3MuonAssociation
)

#
# The Phase 2 sequences
#

muonAssociationReduced_seq = cms.Sequence(
    probeTracks_seq+tpToTkMuonAssociation
    +tpToStaUpdMuonAssociation
    +tpToGlbMuonAssociation
    +tunepMuonTracks_seq+tpToTunePMuonAssociation
    +pfMuonTracks_seq+tpToPFMuonAssociation
    +recoMuonTracks_seq+tpTorecoMuonMuonAssociation
    +tpToDisplacedStaMuonAssociation
    +tpToDisplacedTrkMuonAssociation
    +tpToDisplacedGlbMuonAssociation
)

_muonAssociationHLT_seq = cms.Sequence(
    hltPhase2L2MuonSeedTracks+Phase2tpToL2SeedAssociation
    +Phase2tpToL2MuonAssociation+Phase2tpToL2MuonUpdAssociation
    +Phase2tpToL3IOTkAssociation+Phase2tpToL3OITkAssociation
    +Phase2tpToL3TkMergedAssociation+Phase2tpToL3GlbMuonMergedAssociation
    +hltPhase2L3MuonNoIdTracks+Phase2tpToL3MuonNoIdAssociation
    +hltPhase2L3MuonIdTracks+Phase2tpToL3MuonIdAssociation
)

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith(muonAssociationHLT_seq, _muonAssociationHLT_seq)

# Inside-Out first
_muonAssociationHLT_seq_IO_first = cms.Sequence(
    hltPhase2L2MuonSeedTracks+Phase2tpToL2SeedAssociation
    +Phase2tpToL2MuonAssociation+Phase2tpToL2MuonUpdAssociation
    +Phase2tpToL3IOTkAssociation+Phase2tpToL3OITkAssociation
    +Phase2tpToL2MuonToReuseAssociation+Phase2tpToL3IOTkFilteredAssociation
    +Phase2tpToL3TkMergedAssociation+Phase2tpToL3GlbMuonMergedAssociation
    +hltPhase2L3MuonNoIdTracks+Phase2tpToL3MuonNoIdAssociation
    +hltPhase2L3MuonIdTracks+Phase2tpToL3MuonIdAssociation
)
# Outside-In first
_muonAssociationHLT_seq_OI_first = cms.Sequence(
    hltPhase2L2MuonSeedTracks+Phase2tpToL2SeedAssociation
    +Phase2tpToL2MuonAssociation+Phase2tpToL2MuonUpdAssociation
    +Phase2tpToL3OITkAssociation+Phase2tpToL3OITkFilteredAssociation
    +Phase2tpToL3IOTkAssociation+Phase2tpToL3TkMergedAssociation
    +Phase2tpToL3GlbMuonMergedAssociation
    +hltPhase2L3MuonNoIdTracks+Phase2tpToL3MuonNoIdAssociation
    +hltPhase2L3MuonIdTracks+Phase2tpToL3MuonIdAssociation
)

from Configuration.ProcessModifiers.phase2L2AndL3Muons_cff import phase2L2AndL3Muons
phase2L2AndL3Muons.toReplaceWith(muonAssociationHLT_seq, _muonAssociationHLT_seq_IO_first)

from Configuration.ProcessModifiers.phase2L3MuonsOIFirst_cff import phase2L3MuonsOIFirst
(phase2L2AndL3Muons & phase2L3MuonsOIFirst).toReplaceWith(muonAssociationHLT_seq, _muonAssociationHLT_seq_OI_first)

# fastsim has no hlt specific dt hit collection
from Configuration.Eras.Modifier_fastSim_cff import fastSim
_DTrechitTag = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.DTrechitTag
fastSim.toModify(tpToL2MuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL2UpdMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3OITkMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3TkMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3FromL1TkMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL0L3FromL1TkMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3GlbMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3NoIDMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3MuonAssociation, DTrechitTag = _DTrechitTag)

# Phase-2 fastsim
fastSim.toModify(Phase2tpToL2SeedAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(Phase2tpToL2MuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(Phase2tpToL2MuonUpdAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(Phase2tpToL3IOTkAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(Phase2tpToL3OITkAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(Phase2tpToL2MuonToReuseAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(Phase2tpToL3IOTkFilteredAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(Phase2tpToL3OITkFilteredAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(Phase2tpToL3TkMergedAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(Phase2tpToL3GlbMuonMergedAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(Phase2tpToL3MuonNoIdAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(Phase2tpToL3MuonIdAssociation, DTrechitTag = _DTrechitTag)
