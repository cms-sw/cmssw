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

#
# Associators for Full Sim + Reco:
#

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


#
# Associators for Fast Sim
#

tpToTkmuTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByHits'),
    label_tp = cms.InputTag('mergedtruth', 'MergedTrackTruth'),
    label_tr = cms.InputTag('generalTracks')
)

tpToStaTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mergedtruth', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneMuons','UpdatedAtVtx')
)

tpToGlbTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mergedtruth', 'MergedTrackTruth'),
    label_tr = cms.InputTag('globalMuons')
)

tpToL2TrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mergedtruth', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL2Muons','UpdatedAtVtx')
)

tpToL3TrackAssociationFS = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mergedtruth', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons')
)

tpToL3TkTrackTrackAssociationFS = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByHits'),
    label_tp = cms.InputTag('mergedtruth','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3TkTracksFromL2','')
)

tpToL3L2TrackTrackAssociationFS = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('TrackAssociatorByHits'),
    label_tp = cms.InputTag('mergedtruth','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons:L2Seeded')
)


#MuonAssociation
import SimMuon.MCTruth.MuonAssociatorByHits_cfi

tpToTkMuonAssociationFS   = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaMuonAssociationFS  = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbMuonAssociationFS  = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL3TkMuonAssociationFS = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL2MuonAssociationFS   = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL3MuonAssociationFS   = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToTkMuonAssociationFS.tpTag = 'mergedtruth:MergedTrackTruth'
tpToTkMuonAssociationFS.tracksTag = 'generalTracks'
tpToTkMuonAssociationFS.UseTracker = True
tpToTkMuonAssociationFS.UseMuon = False
tpToTkMuonAssociationFS.EfficiencyCut_track = 0.5
tpToTkMuonAssociationFS.PurityCut_track = 0.75
tpToTkMuonAssociationFS.simtracksXFTag = "mix:famosSimHitsMuonSimTracks"
tpToTkMuonAssociationFS.DTsimhitsXFTag  = "mix:MuonSimHitsMuonDTHits"
tpToTkMuonAssociationFS.CSCsimHitsXFTag = "mix:MuonSimHitsMuonCSCHits"
tpToTkMuonAssociationFS.RPCsimhitsXFTag = "mix:MuonSimHitsMuonRPCHits"
tpToTkMuonAssociationFS.ROUList = ['famosSimHitsTrackerHits']

tpToStaMuonAssociationFS.tpTag = 'mergedtruth:MergedTrackTruth'
tpToStaMuonAssociationFS.tracksTag = 'standAloneMuons:UpdatedAtVtx'
tpToStaMuonAssociationFS.UseTracker = False
tpToStaMuonAssociationFS.UseMuon = True
tpToStaMuonAssociationFS.EfficiencyCut_muon = 0.5
tpToStaMuonAssociationFS.PurityCut_muon = 0.5
tpToStaMuonAssociationFS.simtracksXFTag = "mix:famosSimHitsMuonSimTracks"
tpToStaMuonAssociationFS.DTsimhitsXFTag  = "mix:MuonSimHitsMuonDTHits"
tpToStaMuonAssociationFS.CSCsimHitsXFTag = "mix:MuonSimHitsMuonCSCHits"
tpToStaMuonAssociationFS.RPCsimhitsXFTag = "mix:MuonSimHitsMuonRPCHits"
tpToStaMuonAssociationFS.ROUList = ['famosSimHitsTrackerHits']

tpToGlbMuonAssociationFS.tpTag = 'mergedtruth:MergedTrackTruth'
tpToGlbMuonAssociationFS.tracksTag = 'globalMuons'
tpToGlbMuonAssociationFS.UseTracker = True
tpToGlbMuonAssociationFS.UseMuon = True
tpToGlbMuonAssociationFS.EfficiencyCut_muon = 0.5
tpToGlbMuonAssociationFS.PurityCut_muon = 0.5
tpToGlbMuonAssociationFS.EfficiencyCut_track = 0.5
tpToGlbMuonAssociationFS.PurityCut_track = 0.75
tpToGlbMuonAssociationFS.simtracksXFTag = "mix:famosSimHitsMuonSimTracks"
tpToGlbMuonAssociationFS.DTsimhitsXFTag  = "mix:MuonSimHitsMuonDTHits"
tpToGlbMuonAssociationFS.CSCsimHitsXFTag = "mix:MuonSimHitsMuonCSCHits"
tpToGlbMuonAssociationFS.RPCsimhitsXFTag = "mix:MuonSimHitsMuonRPCHits"
tpToGlbMuonAssociationFS.ROUList = ['famosSimHitsTrackerHits']

tpToL3TkMuonAssociationFS.tpTag = 'mergedtruth:MergedTrackTruth'
tpToL3TkMuonAssociationFS.tracksTag = 'hltL3TkTracksFromL2'
tpToL3TkMuonAssociationFS.UseTracker = True
tpToL3TkMuonAssociationFS.UseMuon = False
tpToL3TkMuonAssociationFS.EfficiencyCut_track = 0.5
tpToL3TkMuonAssociationFS.PurityCut_track = 0.75
tpToL3TkMuonAssociationFS.simtracksXFTag = "mix:famosSimHitsMuonSimTracks"
tpToL3TkMuonAssociationFS.DTsimhitsXFTag  = "mix:MuonSimHitsMuonDTHits"
tpToL3TkMuonAssociationFS.CSCsimHitsXFTag = "mix:MuonSimHitsMuonCSCHits"
tpToL3TkMuonAssociationFS.RPCsimhitsXFTag = "mix:MuonSimHitsMuonRPCHits"
tpToL3TkMuonAssociationFS.ROUList = ['famosSimHitsTrackerHits']
tpToL3TkMuonAssociationFS.ignoreMissingTrackCollection = True

tpToL2MuonAssociationFS.tpTag = 'mergedtruth:MergedTrackTruth'
tpToL2MuonAssociationFS.tracksTag = 'hltL2Muons:UpdatedAtVtx'
tpToL2MuonAssociationFS.UseTracker = False
tpToL2MuonAssociationFS.UseMuon = True
tpToL2MuonAssociationFS.EfficiencyCut_muon = 0.5
tpToL2MuonAssociationFS.PurityCut_muon = 0.5
tpToL2MuonAssociationFS.simtracksXFTag = "mix:famosSimHitsMuonSimTracks"
tpToL2MuonAssociationFS.DTsimhitsXFTag  = "mix:MuonSimHitsMuonDTHits"
tpToL2MuonAssociationFS.CSCsimHitsXFTag = "mix:MuonSimHitsMuonCSCHits"
tpToL2MuonAssociationFS.RPCsimhitsXFTag = "mix:MuonSimHitsMuonRPCHits"
tpToL2MuonAssociationFS.ROUList = ['famosSimHitsTrackerHits']
tpToL2MuonAssociationFS.ignoreMissingTrackCollection = True

tpToL3MuonAssociationFS.tpTag = 'mergedtruth:MergedTrackTruth'
tpToL3MuonAssociationFS.tracksTag = 'hltL3Muons'
tpToL3MuonAssociationFS.UseTracker = True
tpToL3MuonAssociationFS.UseMuon = True
tpToL3MuonAssociationFS.EfficiencyCut_muon = 0.5
tpToL3MuonAssociationFS.PurityCut_muon = 0.5
tpToL3MuonAssociationFS.EfficiencyCut_track = 0.5
tpToL3MuonAssociationFS.PurityCut_track = 0.75
tpToL3MuonAssociationFS.simtracksXFTag = "mix:famosSimHitsMuonSimTracks"
tpToL3MuonAssociationFS.DTsimhitsXFTag  = "mix:MuonSimHitsMuonDTHits"
tpToL3MuonAssociationFS.CSCsimHitsXFTag = "mix:MuonSimHitsMuonCSCHits"
tpToL3MuonAssociationFS.RPCsimhitsXFTag = "mix:MuonSimHitsMuonRPCHits"
tpToL3MuonAssociationFS.ROUList = ['famosSimHitsTrackerHits']
tpToL3MuonAssociationFS.ignoreMissingTrackCollection = True



muonAssociationFastSim_seq = cms.Sequence((tpToTkMuonAssociationFS+tpToStaMuonAssociationFS+tpToGlbMuonAssociationFS)
                                   +(tpToTkmuTrackAssociationFS+tpToStaTrackAssociationFS+tpToGlbTrackAssociationFS))

muonAssociationHLTFastSim_seq = cms.Sequence(
    (tpToL2MuonAssociationFS
     +tpToL3MuonAssociationFS
     +tpToL3TkMuonAssociationFS)
    +(
    tpToL2TrackAssociationFS
    +tpToL3TrackAssociationFS
    +tpToL3TkTrackTrackAssociationFS
    )
)

