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
# COSMICS reco
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
    +tpToL3OITkMuonAssociation+tpToL3TkMuonAssociation+tpToL3FromL1TkMuonAssociation+tpToL0L3FromL1TkMuonAssociation
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
fastSim.toModify(tpToL0L3FromL1TkMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3GlbMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3NoIDMuonAssociation, DTrechitTag = _DTrechitTag)
fastSim.toModify(tpToL3MuonAssociation, DTrechitTag = _DTrechitTag)
