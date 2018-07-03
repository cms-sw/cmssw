# configuration for FullSim: muon track associations done with TrackAssociatorByPosition
#  (backup solution, incomplete, not run by default)
#
import FWCore.ParameterSet.Config as cms

#Track selector
from Validation.RecoMuon.selectors_cff import *

#TrackAssociation
from SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi import *
import SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi
import SimTracker.TrackAssociatorProducers.trackAssociatorByPosition_cfi

import SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi

trackAssociatorByHits = SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone()

onlineTrackAssociatorByHits = SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone()
onlineTrackAssociatorByHits.UseGrouped = cms.bool(False)
onlineTrackAssociatorByHits.UseSplitting = cms.bool(False)
onlineTrackAssociatorByHits.ThreeHitTracksAreSpecial = False

trackAssociatorByPosDeltaR = SimTracker.TrackAssociatorProducers.trackAssociatorByPosition_cfi.trackAssociatorByPosition.clone()
trackAssociatorByPosDeltaR.method = cms.string('momdr')
trackAssociatorByPosDeltaR.QCut = cms.double(0.5)
trackAssociatorByPosDeltaR.ConsiderAllSimHits = cms.bool(True)

# select probe tracks
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
probeTracks = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
probeTracks.quality = cms.vstring('highPurity')
probeTracks.tip = cms.double(3.5)
probeTracks.lip = cms.double(30.)
probeTracks.ptMin = cms.double(4.0)
probeTracks.minRapidity = cms.double(-2.4)
probeTracks.maxRapidity = cms.double(2.4)
probeTracks_seq = cms.Sequence( probeTracks )

#
# Associators for Full Sim + Reco:
#

tpToTkmuTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByHits'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
#    label_tr = cms.InputTag('generalTracks')
    label_tr = cms.InputTag('probeTracks')
)

tpToStaTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneMuons','')
)

tpToStaUpdTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneMuons','UpdatedAtVtx')
)

tpToGlbTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('extractedGlobalMuons')
)

tpToStaSETTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneSETMuons','')
)

tpToStaSETUpdTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneSETMuons','UpdatedAtVtx')
)

tpToGlbSETTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('globalSETMuons')
)

tpToTevFirstTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','firstHit')
)

tpToTevPickyTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','picky')
)
tpToTevDytTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','dyt')
)

#
# Associators for HLT muon reco
#

tpToL2TrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL2Muons','')
)

tpToL2UpdTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL2Muons','UpdatedAtVtx')
)

tpToL3TrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons')
)

tpToL3TkTrackTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('onlineTrackAssociatorByHits'),
    label_tp = cms.InputTag('mix','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3TkTracksFromL2','')
)

tpToL3L2TrackTrackAssociation = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('onlineTrackAssociatorByHits'),
    label_tp = cms.InputTag('mix','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons:L2Seeded')
)

#
# Associators for cosmics:
#

tpToTkCosmicTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByHits'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('ctfWithMaterialTracksP5LHCNavigation')
)

tpToStaCosmicTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('cosmicMuons')
)

tpToGlbCosmicTrackAssociation = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('globalCosmicMuons')
)

#
# The full-sim association sequences
#

muonAssociation_seq = cms.Sequence(
    tpToTkmuTrackAssociation+tpToStaTrackAssociation+tpToStaUpdTrackAssociation+tpToGlbTrackAssociation
    )
muonAssociationTEV_seq = cms.Sequence(
    tpToTevFirstTrackAssociation+tpToTevPickyTrackAssociation+tpToTevDytTrackAssociation
    )
muonAssociationSET_seq = cms.Sequence(
    tpToStaSETTrackAssociation+tpToStaSETUpdTrackAssociation+tpToGlbSETTrackAssociation
    )

muonAssociationCosmic_seq = cms.Sequence(
    tpToTkCosmicTrackAssociation+tpToStaCosmicTrackAssociation+tpToGlbCosmicTrackAssociation
    )

muonAssociationHLT_seq = cms.Sequence(
    tpToL2TrackAssociation+tpToL2UpdTrackAssociation+tpToL3TrackAssociation+tpToL3TkTrackTrackAssociation
    )
