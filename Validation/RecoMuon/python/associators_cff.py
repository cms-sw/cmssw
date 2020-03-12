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
    associator = cms.InputTag('trackAssociatorByHits'),
#    label_tr = cms.InputTag('generalTracks')
    label_tr = cms.InputTag('probeTracks')
)

#
# MuonAssociatorByHits used for all track collections
#
import SimMuon.MCTruth.MuonAssociatorByHits_cfi
MABH = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
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
MABH.EfficiencyCut_muon = 0.5
MABH.PurityCut_muon = 0.75
MABH.includeZeroHitMuons = False
################################################

tpToTkMuonAssociation = MABH.clone()
#tpToTkMuonAssociation.tracksTag = 'generalTracks'
tpToTkMuonAssociation.tracksTag ='probeTracks'
tpToTkMuonAssociation.UseTracker = True
tpToTkMuonAssociation.UseMuon = False

tpToStaSeedAssociation = MABH.clone()
tpToStaSeedAssociation.tracksTag = 'seedsOfSTAmuons'
tpToStaSeedAssociation.UseTracker = False
tpToStaSeedAssociation.UseMuon = True
tpToStaSeedAssociation.EfficiencyCut_muon = 0.

tpToStaMuonAssociation = MABH.clone()
tpToStaMuonAssociation.tracksTag = 'standAloneMuons'
tpToStaMuonAssociation.UseTracker = False
tpToStaMuonAssociation.UseMuon = True

tpToStaUpdMuonAssociation = MABH.clone()
tpToStaUpdMuonAssociation.tracksTag = 'standAloneMuons:UpdatedAtVtx'
tpToStaUpdMuonAssociation.UseTracker = False
tpToStaUpdMuonAssociation.UseMuon = True

tpToGlbMuonAssociation = MABH.clone()
tpToGlbMuonAssociation.tracksTag = 'globalMuons'
tpToGlbMuonAssociation.UseTracker = True
tpToGlbMuonAssociation.UseMuon = True

tpToStaRefitMuonAssociation = MABH.clone()
tpToStaRefitMuonAssociation.tracksTag = 'refittedStandAloneMuons'
tpToStaRefitMuonAssociation.UseTracker = False
tpToStaRefitMuonAssociation.UseMuon = True

tpToStaRefitUpdMuonAssociation = MABH.clone()
tpToStaRefitUpdMuonAssociation.tracksTag = 'refittedStandAloneMuons:UpdatedAtVtx'
tpToStaRefitUpdMuonAssociation.UseTracker = False
tpToStaRefitUpdMuonAssociation.UseMuon = True

tpToDisplacedTrkMuonAssociation = MABH.clone()
tpToDisplacedTrkMuonAssociation.tracksTag = 'displacedTracks'
tpToDisplacedTrkMuonAssociation.UseTracker = True
tpToDisplacedTrkMuonAssociation.UseMuon = False

tpToDisplacedStaSeedAssociation = MABH.clone()
tpToDisplacedStaSeedAssociation.tracksTag = 'seedsOfDisplacedSTAmuons'
tpToDisplacedStaSeedAssociation.UseTracker = False
tpToDisplacedStaSeedAssociation.UseMuon = True
tpToDisplacedStaSeedAssociation.EfficiencyCut_muon = 0.

tpToDisplacedStaMuonAssociation = MABH.clone()
tpToDisplacedStaMuonAssociation.tracksTag = 'displacedStandAloneMuons'
tpToDisplacedStaMuonAssociation.UseTracker = False
tpToDisplacedStaMuonAssociation.UseMuon = True

tpToDisplacedGlbMuonAssociation = MABH.clone()
tpToDisplacedGlbMuonAssociation.tracksTag = 'displacedGlobalMuons'
tpToDisplacedGlbMuonAssociation.UseTracker = True
tpToDisplacedGlbMuonAssociation.UseMuon = True

tpToTevFirstMuonAssociation = MABH.clone()
tpToTevFirstMuonAssociation.tracksTag = 'tevMuons:firstHit'
tpToTevFirstMuonAssociation.UseTracker = True
tpToTevFirstMuonAssociation.UseMuon = True
tpToTevFirstMuonAssociation.EfficiencyCut_muon = 0.

tpToTevPickyMuonAssociation = MABH.clone()
tpToTevPickyMuonAssociation.tracksTag = 'tevMuons:picky'
tpToTevPickyMuonAssociation.UseTracker = True
tpToTevPickyMuonAssociation.UseMuon = True
tpToTevPickyMuonAssociation.EfficiencyCut_muon = 0.

tpToTevDytMuonAssociation = MABH.clone()
tpToTevDytMuonAssociation.tracksTag = 'tevMuons:dyt'
tpToTevDytMuonAssociation.UseTracker = True
tpToTevDytMuonAssociation.UseMuon = True
tpToTevDytMuonAssociation.EfficiencyCut_muon = 0.

# tuneP (GlobalMuons with TuneP definition)
tpToTunePMuonAssociation = MABH.clone()
tpToTunePMuonAssociation.tracksTag = 'tunepMuonTracks'
tpToTunePMuonAssociation.UseTracker = True
tpToTunePMuonAssociation.UseMuon = True
tpToTunePMuonAssociation.EfficiencyCut_muon = 0.

# PFMuons
tpToPFMuonAssociation = MABH.clone()
tpToPFMuonAssociation.tracksTag = 'pfMuonTracks'
tpToPFMuonAssociation.UseTracker = True
tpToPFMuonAssociation.UseMuon = True
tpToPFMuonAssociation.EfficiencyCut_muon = 0.

# all offline reco::Muons with loose cuts
tpTorecoMuonMuonAssociation = MABH.clone()
tpTorecoMuonMuonAssociation.tracksTag = 'recoMuonTracks'
tpTorecoMuonMuonAssociation.UseTracker = True
tpTorecoMuonMuonAssociation.UseMuon = True
tpTorecoMuonMuonAssociation.EfficiencyCut_track = 0.
tpTorecoMuonMuonAssociation.PurityCut_track = 0.
tpTorecoMuonMuonAssociation.EfficiencyCut_muon = 0.
tpTorecoMuonMuonAssociation.PurityCut_muon = 0.
tpTorecoMuonMuonAssociation.includeZeroHitMuons = True

# ME0Muons
tpToME0MuonMuonAssociation = MABH.clone()
tpToME0MuonMuonAssociation.tracksTag = 'extractMe0Muons'
tpToME0MuonMuonAssociation.UseTracker = True
tpToME0MuonMuonAssociation.UseMuon = False

# GEMmuons
tpToGEMMuonMuonAssociation = MABH.clone()
tpToGEMMuonMuonAssociation.tracksTag = 'extractGemMuons'
tpToGEMMuonMuonAssociation.UseTracker = True
tpToGEMMuonMuonAssociation.UseMuon = False

# === HLT muon tracks 
#
MABHhlt = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
MABHhlt.PurityCut_track = 0.75
MABHhlt.PurityCut_muon = 0.75
MABHhlt.DTrechitTag = 'hltDt1DRecHits'
MABHhlt.ignoreMissingTrackCollection = True
# N.B. for HLT these default settings are implicitly applied :
#MABHhlt.EfficiencyCut_track = 0.   # backup solution as UseGrouped/UseSplitting are always assumed to be true
#MABHhlt.EfficiencyCut_muon = 0.    # | loose matching requests for triggering
#MABHhlt.includeZeroHitMuons = True # |

tpToL2MuonAssociation = MABHhlt.clone()
tpToL2MuonAssociation.tracksTag = 'hltL2Muons'
tpToL2MuonAssociation.UseTracker = False
tpToL2MuonAssociation.UseMuon = True

tpToL2UpdMuonAssociation = MABHhlt.clone()
tpToL2UpdMuonAssociation.tracksTag = 'hltL2Muons:UpdatedAtVtx'
tpToL2UpdMuonAssociation.UseTracker = False
tpToL2UpdMuonAssociation.UseMuon = True

tpToL3OITkMuonAssociation = MABHhlt.clone()
tpToL3OITkMuonAssociation.tracksTag = 'hltIterL3OIMuonTrackSelectionHighPurity'
tpToL3OITkMuonAssociation.UseTracker = True
tpToL3OITkMuonAssociation.UseMuon = False

tpToL3TkMuonAssociation = MABHhlt.clone()
tpToL3TkMuonAssociation.tracksTag = 'hltIterL3MuonMerged'
tpToL3TkMuonAssociation.UseTracker = True
tpToL3TkMuonAssociation.UseMuon = False

tpToL3FromL1TkMuonAssociation = MABHhlt.clone()
tpToL3FromL1TkMuonAssociation.tracksTag = 'hltIterL3MuonAndMuonFromL1Merged'
tpToL3FromL1TkMuonAssociation.UseTracker = True
tpToL3FromL1TkMuonAssociation.UseMuon = False

tpToL3GlbMuonAssociation = MABHhlt.clone()
tpToL3GlbMuonAssociation.tracksTag = 'hltIterL3GlbMuon'
tpToL3GlbMuonAssociation.UseTracker = True
tpToL3GlbMuonAssociation.UseMuon = True

tpToL3NoIDMuonAssociation = MABHhlt.clone()
tpToL3NoIDMuonAssociation.tracksTag = 'hltIterL3MuonsNoIDTracks'
tpToL3NoIDMuonAssociation.UseTracker = True
tpToL3NoIDMuonAssociation.UseMuon = True

tpToL3MuonAssociation = MABHhlt.clone()
tpToL3MuonAssociation.tracksTag = 'hltIterL3MuonsTracks'
tpToL3MuonAssociation.UseTracker = True
tpToL3MuonAssociation.UseMuon = True
# ===

#
# COSMICS reco
#
# 2-legs cosmics reco: simhits can be twice the reconstructed ones in any single leg
# (Quality cut have to be set at 0.25, purity cuts can stay at default value 0.75)
# T.B.D. upper and lower leg should be analyzed separately 
#
tpToTkCosmicSelMuonAssociation = MABH.clone()
tpToTkCosmicSelMuonAssociation.tracksTag = 'ctfWithMaterialTracksP5LHCNavigation'
tpToTkCosmicSelMuonAssociation.UseTracker = True
tpToTkCosmicSelMuonAssociation.UseMuon = False
tpToTkCosmicSelMuonAssociation.EfficiencyCut_track = 0.25

tpToStaCosmicSelMuonAssociation = MABH.clone()
tpToStaCosmicSelMuonAssociation.tracksTag = 'cosmicMuons'
tpToStaCosmicSelMuonAssociation.UseTracker = False
tpToStaCosmicSelMuonAssociation.UseMuon = True
tpToStaCosmicSelMuonAssociation.EfficiencyCut_muon = 0.25

tpToGlbCosmicSelMuonAssociation = MABH.clone()
tpToGlbCosmicSelMuonAssociation.tracksTag = 'globalCosmicMuons'
tpToGlbCosmicSelMuonAssociation.UseTracker = True
tpToGlbCosmicSelMuonAssociation.UseMuon = True
tpToGlbCosmicSelMuonAssociation.EfficiencyCut_track = 0.25
tpToGlbCosmicSelMuonAssociation.EfficiencyCut_muon = 0.25

# 1-leg cosmics reco
tpToTkCosmic1LegSelMuonAssociation = MABH.clone()
tpToTkCosmic1LegSelMuonAssociation.tracksTag = 'ctfWithMaterialTracksP5'
tpToTkCosmic1LegSelMuonAssociation.UseTracker = True
tpToTkCosmic1LegSelMuonAssociation.UseMuon = False

tpToStaCosmic1LegSelMuonAssociation = MABH.clone()
tpToStaCosmic1LegSelMuonAssociation.tracksTag = 'cosmicMuons1Leg'
tpToStaCosmic1LegSelMuonAssociation.UseTracker = False
tpToStaCosmic1LegSelMuonAssociation.UseMuon = True

tpToGlbCosmic1LegSelMuonAssociation = MABH.clone()
tpToGlbCosmic1LegSelMuonAssociation.tracksTag = 'globalCosmicMuons1Leg'
tpToGlbCosmic1LegSelMuonAssociation.UseTracker = True
tpToGlbCosmic1LegSelMuonAssociation.UseMuon = True

#
# The full-sim association sequences
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

muonAssociationHLT_seq = cms.Sequence(
    tpToL2MuonAssociation+tpToL2UpdMuonAssociation
    +tpToL3OITkMuonAssociation+tpToL3TkMuonAssociation+tpToL3FromL1TkMuonAssociation
    +tpToL3GlbMuonAssociation
    +hltIterL3MuonsNoIDTracks_seq+tpToL3NoIDMuonAssociation
    +hltIterL3MuonsTracks_seq+tpToL3MuonAssociation
    )


# fastsim has no hlt specific dt hit collection
from Configuration.Eras.Modifier_fastSim_cff import fastSim
_DTrechitTag = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.DTrechitTag
fastSim.toModify(tpToL2MuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL2UpdMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3OITkMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3TkMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3FromL1TkMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3GlbMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3NoIDMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3MuonAssociation, DTrechitTag = _DTrechitTag)
