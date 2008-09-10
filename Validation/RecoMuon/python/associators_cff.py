import FWCore.ParameterSet.Config as cms

#Track selector
from Validation.RecoMuon.selectors_cff import *

#TrackAssociation
from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByPosition_cfi import *

#TrackAssociation by DeltaR
TrackAssociatorByPosDeltaR = cms.ESProducer("TrackAssociatorByPositionESProducer",
    # QminCut not used
    QminCut = cms.double(120.0),
    MinIfNoMatch = cms.bool(False),
    ComponentName = cms.string('TrackAssociatorByPosDeltaR'),
    propagator = cms.string('SteppingHelixPropagatorAny'),
    # minimum distance from the origin to find a hit 
    # from a simulated particle and match it to reconstructed track
    positionMinimumDistance = cms.double(0.0),
    # use the delta eta-phi estimator on the position 
    # at a plane in the muon system    
    method = cms.string('posdr'),
    QCut = cms.double(0.5)
)

tpToTkmuTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    associator = cms.string('TrackAssociatorByHits'),
#    label_tp = cms.InputTag("mergedtruth","MergedTrackTruth"),
    label_tp = cms.InputTag("muonTP"),
    label_tr = cms.InputTag("generalTracks")
)

tpToStaTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    associator = cms.string('TrackAssociatorByPosDeltaR'),
#    label_tp = cms.InputTag("mergedtruth","MergedTrackTruth"),
    label_tp = cms.InputTag("muonTP"),
    label_tr = cms.InputTag("standAloneMuons","UpdatedAtVtx")
#    label_tr = cms.InputTag("muonSta")
)

tpToGlbTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    associator = cms.string('TrackAssociatorByPosDeltaR'),
#    label_tp = cms.InputTag("mergedtruth","MergedTrackTruth"),
    label_tp = cms.InputTag("muonTP"),
    label_tr = cms.InputTag("globalMuons")
#    label_tr = cms.InputTag("muonGlb")
)


#MuonAssociation
import SimMuon.MCTruth.MuonAssociatorByHits_cfi
# Tracker muon association
tpToTkMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
# Standalone muon association
tpToStaMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
# Global muon association
tpToGlbMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToTkMuonAssociation.tpTag = 'muonTP'
tpToTkMuonAssociation.tracksTag = 'generalTracks'
tpToTkMuonAssociation.SimToReco_useTracker = True
tpToTkMuonAssociation.SimToReco_useMuon = False
tpToTkMuonAssociation.EfficiencyCut_track = 0.5
tpToTkMuonAssociation.PurityCut_track = 0.75

tpToStaMuonAssociation.tpTag = 'muonTP'
tpToStaMuonAssociation.tracksTag = 'standAloneMuons:UpdatedAtVtx'
tpToStaMuonAssociation.SimToReco_useTracker = False
tpToStaMuonAssociation.SimToReco_useMuon = True
tpToStaMuonAssociation.EfficiencyCut_muon = 0.5
tpToStaMuonAssociation.PurityCut_muon = 0.5

tpToGlbMuonAssociation.tpTag = 'muonTP'
tpToGlbMuonAssociation.tracksTag = 'globalMuons'
tpToGlbMuonAssociation.SimToReco_useTracker = True
tpToGlbMuonAssociation.SimToReco_useMuon = True
tpToGlbMuonAssociation.EfficiencyCut_muon = 0.5
tpToGlbMuonAssociation.PurityCut_muon = 0.5
tpToGlbMuonAssociation.EfficiencyCut_track = 0.5
tpToGlbMuonAssociation.PurityCut_track = 0.75

from SimGeneral.MixingModule.mixNoPU_cfi import *

muonAssociation_seq = cms.Sequence(mix*(tpToTkMuonAssociation+tpToStaMuonAssociation+tpToGlbMuonAssociation)+(tpToTkmuTrackAssociation+tpToStaTrackAssociation+tpToGlbTrackAssociation))
