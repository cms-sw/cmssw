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
    ComponentName = cms.string('TrackAssociatorByDeltaR'),
    propagator = cms.string('SteppingHelixPropagatorAny'),
    # minimum distance from the origin to find a hit 
    # from a simulated particle and match it to reconstructed track
    positionMinimumDistance = cms.double(0.0),
    # use the delta eta-phi estimator on the position 
    # at a plane in the muon system    
    method = cms.string('momdr'),
    QCut = cms.double(0.5),
    ConsiderAllSimHits = cms.bool(True)
)

tpToTkmuTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByHits'),
    label_tp = cms.InputTag('mergedtruth', 'MergedTrackTruth'),
    label_tr = cms.InputTag('generalTracks')
)

tpToStaTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mergedtruth', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneMuons','UpdatedAtVtx')
#    label_tr = cms.InputTag('muonSta')
)

tpToGlbTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mergedtruth', 'MergedTrackTruth'),
    label_tr = cms.InputTag('globalMuons')
#    label_tr = cms.InputTag('muonGlb')
)

tpToL2TrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mergedtruth', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL2Muons','UpdatedAtVtx')
)

tpToL3TrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mergedtruth', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons')
)

tpToL3TkTrackTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByHits'),
    label_tp = cms.InputTag('mergedtruth','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3TkTracksFromL2','')
)

tpToL3L2TrackTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByHits'),
    label_tp = cms.InputTag('mergedtruth','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons:L2Seeded')
)


#MuonAssociation
import SimMuon.MCTruth.MuonAssociatorByHits_cfi

tpToTkMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL3TkMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL2MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL3MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToTkMuonAssociation.tpTag = 'mergedtruth:MergedTrackTruth'
tpToTkMuonAssociation.tracksTag = 'generalTracks'
tpToTkMuonAssociation.UseTracker = True
tpToTkMuonAssociation.UseMuon = False
tpToTkMuonAssociation.EfficiencyCut_track = 0.5
tpToTkMuonAssociation.PurityCut_track = 0.75

tpToStaMuonAssociation.tpTag = 'mergedtruth:MergedTrackTruth'
tpToStaMuonAssociation.tracksTag = 'standAloneMuons:UpdatedAtVtx'
tpToStaMuonAssociation.UseTracker = False
tpToStaMuonAssociation.UseMuon = True
tpToStaMuonAssociation.EfficiencyCut_muon = 0.5
tpToStaMuonAssociation.PurityCut_muon = 0.5

tpToGlbMuonAssociation.tpTag = 'mergedtruth:MergedTrackTruth'
tpToGlbMuonAssociation.tracksTag = 'globalMuons'
tpToGlbMuonAssociation.UseTracker = True
tpToGlbMuonAssociation.UseMuon = True
tpToGlbMuonAssociation.EfficiencyCut_muon = 0.5
tpToGlbMuonAssociation.PurityCut_muon = 0.5
tpToGlbMuonAssociation.EfficiencyCut_track = 0.5
tpToGlbMuonAssociation.PurityCut_track = 0.75

tpToL3TkMuonAssociation.tpTag = 'mergedtruth:MergedTrackTruth'
tpToL3TkMuonAssociation.tracksTag = 'hltL3TkTracksFromL2'
tpToL3TkMuonAssociation.DTrechitTag = 'hltDt1DRecHits'
tpToL3TkMuonAssociation.UseTracker = True
tpToL3TkMuonAssociation.UseMuon = False
tpToL3TkMuonAssociation.EfficiencyCut_track = 0.5
tpToL3TkMuonAssociation.PurityCut_track = 0.75
tpToL3TkMuonAssociation.ignoreMissingTrackCollection = True

tpToL2MuonAssociation.tpTag = 'mergedtruth:MergedTrackTruth'
tpToL2MuonAssociation.tracksTag = 'hltL2Muons:UpdatedAtVtx'
tpToL2MuonAssociation.DTrechitTag = 'hltDt1DRecHits'
tpToL2MuonAssociation.UseTracker = False
tpToL2MuonAssociation.UseMuon = True
tpToL2MuonAssociation.EfficiencyCut_muon = 0.5
tpToL2MuonAssociation.PurityCut_muon = 0.5
tpToL2MuonAssociation.ignoreMissingTrackCollection = True

tpToL3MuonAssociation.tpTag = 'mergedtruth:MergedTrackTruth'
tpToL3MuonAssociation.tracksTag = 'hltL3Muons'
tpToL3MuonAssociation.DTrechitTag = 'hltDt1DRecHits'
tpToL3MuonAssociation.UseTracker = True
tpToL3MuonAssociation.UseMuon = True
tpToL3MuonAssociation.EfficiencyCut_muon = 0.5
tpToL3MuonAssociation.PurityCut_muon = 0.5
tpToL3MuonAssociation.EfficiencyCut_track = 0.5
tpToL3MuonAssociation.PurityCut_track = 0.75
tpToL3MuonAssociation.ignoreMissingTrackCollection = True

muonAssociation_seq = cms.Sequence((tpToTkMuonAssociation+tpToStaMuonAssociation+tpToGlbMuonAssociation)
                                  +(tpToTkmuTrackAssociation+tpToStaTrackAssociation+tpToGlbTrackAssociation))

muonAssociationHLT_seq = cms.Sequence(
    (tpToL2MuonAssociation
     +tpToL3MuonAssociation
     +tpToL3TkMuonAssociation)
    +(
    tpToL2TrackAssociation
    +tpToL3TrackAssociation
    +tpToL3TkTrackTrackAssociation
    )
    )
