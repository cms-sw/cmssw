import FWCore.ParameterSet.Config as cms

#Track selector
from Validation.RecoMuon.selectors_cff import *

#TrackAssociation
from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByPosition_cfi import *

#TrackAssociation by DeltaR
TrackAssociatorByPosDeltaR = cms.ESProducer('TrackAssociatorByPositionESProducer',
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

tpToTkmuTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByHits'),
    label_tp = cms.InputTag('muonTP'),
    label_tr = cms.InputTag('generalTracks')
)

tpToStaTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByPosDeltaR'),
    label_tp = cms.InputTag('muonTP'),
    label_tr = cms.InputTag('standAloneMuons','UpdatedAtVtx')
#    label_tr = cms.InputTag('muonSta')
)

tpToGlbTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByPosDeltaR'),
    label_tp = cms.InputTag('muonTP'),
    label_tr = cms.InputTag('globalMuons')
#    label_tr = cms.InputTag('muonGlb')
)

tpToL2TrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByPosDeltaR'),
    label_tp = cms.InputTag('muonTP'),
    label_tr = cms.InputTag('hltL2Muons','UpdatedAtVtx')
)

tpToL3TrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    associator = cms.string('TrackAssociatorByPosDeltaR'),
    label_tp = cms.InputTag('muonTP'),
    label_tr = cms.InputTag('hltL3Muons')
)

#MuonAssociation
import SimMuon.MCTruth.MuonAssociatorByHits_cfi

tpToTkMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL2MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL3MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

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

tpToL2MuonAssociation.tpTag = 'muonTP'
tpToL2MuonAssociation.tracksTag = 'hltL2Muons:UpdatedAtVtx'

tpToL3MuonAssociation.tpTag = 'muonTP'
tpToL3MuonAssociation.tracksTag = 'hltL3Muons'

muonAssociation_seq = cms.Sequence((tpToTkMuonAssociation+tpToStaMuonAssociation+tpToGlbMuonAssociation)
                                  +(tpToTkmuTrackAssociation+tpToStaTrackAssociation+tpToGlbTrackAssociation))

muonAssociationHLT_seq = cms.Sequence((tpToL2MuonAssociation+tpToL3MuonAssociation)
                                     +(tpToL2TrackAssociation+tpToL3TrackAssociation))
