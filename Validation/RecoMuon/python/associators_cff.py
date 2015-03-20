import FWCore.ParameterSet.Config as cms

#Track selector
from Validation.RecoMuon.selectors_cff import *

#TrackAssociation
from SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi import *
import SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi
import SimTracker.TrackAssociatorProducers.trackAssociatorByPosition_cfi

import SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi

#TrackAssociatorByHits = SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone( ComponentName = 'TrackAssociatorByHits' )
trackAssociatorByHits = SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone()


onlineTrackAssociatorByHits = SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone()
onlineTrackAssociatorByHits.UseGrouped = cms.bool(False)
onlineTrackAssociatorByHits.UseSplitting = cms.bool(False)
onlineTrackAssociatorByHits.ThreeHitTracksAreSpecial = False

trackAssociatorByPosDeltaR = SimTracker.TrackAssociatorProducers.trackAssociatorByPosition_cfi.trackAssociatorByPosition.clone()
trackAssociatorByPosDeltaR.method = cms.string('momdr')
trackAssociatorByPosDeltaR.QCut = cms.double(0.5)
trackAssociatorByPosDeltaR.ConsiderAllSimHits = cms.bool(True)

#
# Configuration for Muon track extractor
#

import SimMuon.MCTruth.MuonTrackProducer_cfi
extractedGlobalMuons = SimMuon.MCTruth.MuonTrackProducer_cfi.muonTrackProducer.clone()
extractedGlobalMuons.selectionTags = ('AllGlobalMuons',)
extractedGlobalMuons.trackType = "globalTrack"
extractedMuonTracks_seq = cms.Sequence( extractedGlobalMuons )

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



#MuonAssociation
import SimMuon.MCTruth.MuonAssociatorByHits_cfi

tpToTkMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaSeedAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaUpdMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaRefitMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaRefitUpdMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToDisplacedTrkMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToDisplacedStaSeedAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToDisplacedStaMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToDisplacedGlbMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaSETMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaSETUpdMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbSETMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTevFirstMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTevPickyMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToTevDytMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL3TkMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL2MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL2UpdMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToL3MuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToTkMuonAssociation.tpTag = 'mix:MergedTrackTruth'
#tpToTkMuonAssociation.tracksTag = 'generalTracks'
tpToTkMuonAssociation.tracksTag = 'probeTracks'
tpToTkMuonAssociation.UseTracker = True
tpToTkMuonAssociation.UseMuon = False

tpToStaSeedAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaSeedAssociation.tracksTag = 'seedsOfSTAmuons'
tpToStaSeedAssociation.UseTracker = False
tpToStaSeedAssociation.UseMuon = True


tpToStaMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaMuonAssociation.tracksTag = 'standAloneMuons'
tpToStaMuonAssociation.UseTracker = False
tpToStaMuonAssociation.UseMuon = True

tpToStaUpdMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaUpdMuonAssociation.tracksTag = 'standAloneMuons:UpdatedAtVtx'
tpToStaUpdMuonAssociation.UseTracker = False
tpToStaUpdMuonAssociation.UseMuon = True

tpToGlbMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToGlbMuonAssociation.tracksTag = 'extractedGlobalMuons'
tpToGlbMuonAssociation.UseTracker = True
tpToGlbMuonAssociation.UseMuon = True

tpToStaRefitMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaRefitMuonAssociation.tracksTag = 'refittedStandAloneMuons'
tpToStaRefitMuonAssociation.UseTracker = False
tpToStaRefitMuonAssociation.UseMuon = True

tpToStaRefitUpdMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaRefitUpdMuonAssociation.tracksTag = 'refittedStandAloneMuons:UpdatedAtVtx'
tpToStaRefitUpdMuonAssociation.UseTracker = False
tpToStaRefitUpdMuonAssociation.UseMuon = True

tpToDisplacedTrkMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToDisplacedTrkMuonAssociation.tracksTag = 'displacedTracks'
tpToDisplacedTrkMuonAssociation.UseTracker = True
tpToDisplacedTrkMuonAssociation.UseMuon = False

tpToDisplacedStaSeedAssociation.tpTag = 'mix:MergedTrackTruth'
tpToDisplacedStaSeedAssociation.tracksTag = 'seedsOfDisplacedSTAmuons'
tpToDisplacedStaSeedAssociation.UseTracker = False
tpToDisplacedStaSeedAssociation.UseMuon = True

tpToDisplacedStaMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToDisplacedStaMuonAssociation.tracksTag = 'displacedStandAloneMuons'
tpToDisplacedStaMuonAssociation.UseTracker = False
tpToDisplacedStaMuonAssociation.UseMuon = True

tpToDisplacedGlbMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToDisplacedGlbMuonAssociation.tracksTag = 'displacedGlobalMuons'
tpToDisplacedGlbMuonAssociation.UseTracker = True
tpToDisplacedGlbMuonAssociation.UseMuon = True

tpToStaSETMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaSETMuonAssociation.tracksTag = 'standAloneSETMuons'
tpToStaSETMuonAssociation.UseTracker = False
tpToStaSETMuonAssociation.UseMuon = True

tpToStaSETUpdMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaSETUpdMuonAssociation.tracksTag = 'standAloneSETMuons:UpdatedAtVtx'
tpToStaSETUpdMuonAssociation.UseTracker = False
tpToStaSETUpdMuonAssociation.UseMuon = True

tpToGlbSETMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToGlbSETMuonAssociation.tracksTag = 'globalSETMuons'
tpToGlbSETMuonAssociation.UseTracker = True
tpToGlbSETMuonAssociation.UseMuon = True

tpToTevFirstMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTevFirstMuonAssociation.tracksTag = 'tevMuons:firstHit'
tpToTevFirstMuonAssociation.UseTracker = True
tpToTevFirstMuonAssociation.UseMuon = True

tpToTevPickyMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTevPickyMuonAssociation.tracksTag = 'tevMuons:picky'
tpToTevPickyMuonAssociation.UseTracker = True
tpToTevPickyMuonAssociation.UseMuon = True

tpToTevDytMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTevDytMuonAssociation.tracksTag = 'tevMuons:dyt'
tpToTevDytMuonAssociation.UseTracker = True
tpToTevDytMuonAssociation.UseMuon = True

tpToL3TkMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToL3TkMuonAssociation.tracksTag = 'hltL3TkTracksFromL2'
tpToL3TkMuonAssociation.DTrechitTag = 'hltDt1DRecHits'
tpToL3TkMuonAssociation.UseTracker = True
tpToL3TkMuonAssociation.UseMuon = False
tpToL3TkMuonAssociation.ignoreMissingTrackCollection = True
tpToL3TkMuonAssociation.UseSplitting = False
tpToL3TkMuonAssociation.UseGrouped = False

tpToL2MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToL2MuonAssociation.tracksTag = 'hltL2Muons'
tpToL2MuonAssociation.DTrechitTag = 'hltDt1DRecHits'
tpToL2MuonAssociation.UseTracker = False
tpToL2MuonAssociation.UseMuon = True
tpToL2MuonAssociation.ignoreMissingTrackCollection = True

tpToL2UpdMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToL2UpdMuonAssociation.tracksTag = 'hltL2Muons:UpdatedAtVtx'
tpToL2UpdMuonAssociation.DTrechitTag = 'hltDt1DRecHits'
tpToL2UpdMuonAssociation.UseTracker = False
tpToL2UpdMuonAssociation.UseMuon = True
tpToL2UpdMuonAssociation.ignoreMissingTrackCollection = True

tpToL3MuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToL3MuonAssociation.tracksTag = 'hltL3Muons'
tpToL3MuonAssociation.DTrechitTag = 'hltDt1DRecHits'
tpToL3MuonAssociation.UseTracker = True
tpToL3MuonAssociation.UseMuon = True
tpToL3MuonAssociation.ignoreMissingTrackCollection = True
tpToL3MuonAssociation.UseSplitting = False
tpToL3MuonAssociation.UseGrouped = False

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

# 2-legs cosmics reco: simhits can be twice the reconstructed ones in any single leg
# (Quality cut have to be set at 0.25, purity cuts can stay at 0.75)
# T.B.D. upper and lower leg should be analyzed separately 
tpToTkCosmicSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaCosmicSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbCosmicSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
# 1-leg cosmics reco
tpToTkCosmic1LegSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToStaCosmic1LegSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
tpToGlbCosmic1LegSelMuonAssociation = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()

tpToTkCosmicSelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTkCosmicSelMuonAssociation.tracksTag = 'ctfWithMaterialTracksP5LHCNavigation'
tpToTkCosmicSelMuonAssociation.UseTracker = True
tpToTkCosmicSelMuonAssociation.UseMuon = False
tpToTkCosmicSelMuonAssociation.EfficiencyCut_track = cms.double(0.25)
tpToTkCosmicSelMuonAssociation.PurityCut_track = cms.double(0.75)

tpToStaCosmicSelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaCosmicSelMuonAssociation.tracksTag = 'cosmicMuons'
tpToStaCosmicSelMuonAssociation.UseTracker = False
tpToStaCosmicSelMuonAssociation.UseMuon = True
tpToStaCosmicSelMuonAssociation.includeZeroHitMuons = False
tpToStaCosmicSelMuonAssociation.EfficiencyCut_muon = cms.double(0.25)
tpToStaCosmicSelMuonAssociation.PurityCut_muon = cms.double(0.75)

tpToGlbCosmicSelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToGlbCosmicSelMuonAssociation.tracksTag = 'globalCosmicMuons'
tpToGlbCosmicSelMuonAssociation.UseTracker = True
tpToGlbCosmicSelMuonAssociation.UseMuon = True
tpToGlbCosmicSelMuonAssociation.EfficiencyCut_track = cms.double(0.25)
tpToGlbCosmicSelMuonAssociation.PurityCut_track = cms.double(0.75)
tpToGlbCosmicSelMuonAssociation.EfficiencyCut_muon = cms.double(0.25)
tpToGlbCosmicSelMuonAssociation.PurityCut_muon = cms.double(0.75)
tpToGlbCosmicSelMuonAssociation.acceptOneStubMatchings = False
tpToGlbCosmicSelMuonAssociation.includeZeroHitMuons = False

tpToTkCosmic1LegSelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToTkCosmic1LegSelMuonAssociation.tracksTag = 'ctfWithMaterialTracksP5'
tpToTkCosmic1LegSelMuonAssociation.UseTracker = True
tpToTkCosmic1LegSelMuonAssociation.UseMuon = False
tpToTkCosmic1LegSelMuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToTkCosmic1LegSelMuonAssociation.PurityCut_track = cms.double(0.75)

tpToStaCosmic1LegSelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToStaCosmic1LegSelMuonAssociation.tracksTag = 'cosmicMuons1Leg'
tpToStaCosmic1LegSelMuonAssociation.UseTracker = False
tpToStaCosmic1LegSelMuonAssociation.UseMuon = True
tpToStaCosmic1LegSelMuonAssociation.includeZeroHitMuons = False
tpToStaCosmic1LegSelMuonAssociation.EfficiencyCut_muon = cms.double(0.5)
tpToStaCosmic1LegSelMuonAssociation.PurityCut_muon = cms.double(0.75)

tpToGlbCosmic1LegSelMuonAssociation.tpTag = 'mix:MergedTrackTruth'
tpToGlbCosmic1LegSelMuonAssociation.tracksTag = 'globalCosmicMuons1Leg'
tpToGlbCosmic1LegSelMuonAssociation.UseTracker = True
tpToGlbCosmic1LegSelMuonAssociation.UseMuon = True
tpToGlbCosmic1LegSelMuonAssociation.EfficiencyCut_track = cms.double(0.5)
tpToGlbCosmic1LegSelMuonAssociation.PurityCut_track = cms.double(0.75)
tpToGlbCosmic1LegSelMuonAssociation.EfficiencyCut_muon = cms.double(0.5)
tpToGlbCosmic1LegSelMuonAssociation.PurityCut_muon = cms.double(0.75)
tpToGlbCosmic1LegSelMuonAssociation.acceptOneStubMatchings = False
tpToGlbCosmic1LegSelMuonAssociation.includeZeroHitMuons = False

#
# The full-sim association sequences
#

muonAssociation_seq = cms.Sequence(
    probeTracks_seq+tpToTkMuonAssociation
    +trackAssociatorByHits+tpToTkmuTrackAssociation
    +seedsOfSTAmuons_seq+tpToStaSeedAssociation+tpToStaMuonAssociation+tpToStaUpdMuonAssociation
    +extractedMuonTracks_seq+tpToGlbMuonAssociation
)

muonAssociationTEV_seq = cms.Sequence(
    tpToTevFirstMuonAssociation+tpToTevPickyMuonAssociation+tpToTevDytMuonAssociation
)

muonAssociationDisplaced_seq = cms.Sequence(
    seedsOfDisplacedSTAmuons_seq+tpToDisplacedStaSeedAssociation+tpToDisplacedStaMuonAssociation
    +tpToDisplacedTrkMuonAssociation+tpToDisplacedGlbMuonAssociation
)

muonAssociationRefit_seq = cms.Sequence(tpToStaRefitMuonAssociation+tpToStaRefitUpdMuonAssociation)

muonAssociationSET_seq = cms.Sequence(tpToStaSETMuonAssociation+tpToStaSETUpdMuonAssociation+tpToGlbSETMuonAssociation)

muonAssociationCosmic_seq = cms.Sequence(
    tpToTkCosmicSelMuonAssociation+ tpToTkCosmic1LegSelMuonAssociation
    +tpToStaCosmicSelMuonAssociation+tpToStaCosmic1LegSelMuonAssociation
    +tpToGlbCosmicSelMuonAssociation+tpToGlbCosmic1LegSelMuonAssociation
)

muonAssociationHLT_seq = cms.Sequence(
    tpToL2MuonAssociation+tpToL2UpdMuonAssociation+tpToL3TkMuonAssociation+tpToL3MuonAssociation
)


#
# Associators for Fast Sim
#

tpToTkmuTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByHits'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
#    label_tr = cms.InputTag('generalTracks')
    label_tr = cms.InputTag('probeTracks')
)

tpToStaTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneMuons','')
)

tpToStaUpdTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('standAloneMuons','UpdatedAtVtx')
)

tpToGlbTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('extractedGlobalMuons')
)

tpToTevFirstTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','firstHit')
)

tpToTevPickyTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','picky')
)

tpToTevDytTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('tevMuons','dyt')
)

tpToL2TrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL2Muons','')
)

tpToL2UpdTrackAssociationFS = cms.EDProducer('TrackAssociatorEDProducer',
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL2Muons','UpdatedAtVtx')
)

tpToL3TrackAssociationFS = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.InputTag('trackAssociatorByDeltaR'),
    label_tp = cms.InputTag('mix', 'MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons')
)

tpToL3TkTrackTrackAssociationFS = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('onlineTrackAssociatorByHits'),
    label_tp = cms.InputTag('mix','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3TkTracksFromL2','')
)

tpToL3L2TrackTrackAssociationFS = cms.EDProducer("TrackAssociatorEDProducer",
    ignoremissingtrackcollection=cms.untracked.bool(True),
    associator = cms.string('onlineTrackAssociatorByHits'),
    label_tp = cms.InputTag('mix','MergedTrackTruth'),
    label_tr = cms.InputTag('hltL3Muons:L2Seeded')
)


#MuonAssociation
import SimMuon.MCTruth.MuonAssociatorByHits_cfi

baseMuonAssociatorFS = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
baseMuonAssociatorFS.tpTag = 'mix:MergedTrackTruth'
baseMuonAssociatorFS.UseTracker = True
baseMuonAssociatorFS.UseMuon = True
baseMuonAssociatorFS.simtracksTag = "famosSimHits"
baseMuonAssociatorFS.DTsimhitsTag  = "MuonSimHits:MuonDTHits"
baseMuonAssociatorFS.CSCsimHitsTag = "MuonSimHits:MuonCSCHits"
baseMuonAssociatorFS.RPCsimhitsTag = "MuonSimHits:MuonRPCHits"
baseMuonAssociatorFS.simtracksXFTag = "mix:famosSimHits"
baseMuonAssociatorFS.DTsimhitsXFTag  = "mix:MuonSimHitsMuonDTHits"
baseMuonAssociatorFS.CSCsimHitsXFTag = "mix:MuonSimHitsMuonCSCHits"
baseMuonAssociatorFS.RPCsimhitsXFTag = "mix:MuonSimHitsMuonRPCHits"
baseMuonAssociatorFS.ROUList = ['famosSimHitsTrackerHits']


tpToTkMuonAssociationFS  = baseMuonAssociatorFS.clone()
tpToStaSeedAssociationFS = baseMuonAssociatorFS.clone()
tpToStaMuonAssociationFS  = baseMuonAssociatorFS.clone()
tpToStaUpdMuonAssociationFS  = baseMuonAssociatorFS.clone()
tpToStaRefitMuonAssociationFS  = baseMuonAssociatorFS.clone()
tpToStaRefitUpdMuonAssociationFS  = baseMuonAssociatorFS.clone()
tpToGlbMuonAssociationFS  = baseMuonAssociatorFS.clone()
tpToTevFirstMuonAssociationFS = baseMuonAssociatorFS.clone()
tpToTevPickyMuonAssociationFS = baseMuonAssociatorFS.clone()
tpToTevDytMuonAssociationFS = baseMuonAssociatorFS.clone()
tpToL3TkMuonAssociationFS = baseMuonAssociatorFS.clone()
tpToL2MuonAssociationFS   = baseMuonAssociatorFS.clone()
tpToL2UpdMuonAssociationFS   = baseMuonAssociatorFS.clone()
tpToL3MuonAssociationFS   = baseMuonAssociatorFS.clone()

#tpToTkMuonAssociationFS.tracksTag = 'generalTracks'
tpToTkMuonAssociationFS.tracksTag = 'probeTracks'
tpToTkMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToTkMuonAssociationFS.UseTracker = True
tpToTkMuonAssociationFS.UseMuon = False

tpToStaSeedAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToStaSeedAssociationFS.tracksTag = 'seedsOfSTAmuons'
tpToStaSeedAssociationFS.UseTracker = False
tpToStaSeedAssociationFS.UseMuon = True

tpToStaMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToStaMuonAssociationFS.tracksTag = 'standAloneMuons'
tpToStaMuonAssociationFS.UseTracker = False
tpToStaMuonAssociationFS.UseMuon = True

tpToStaUpdMuonAssociationFS.tracksTag = 'standAloneMuons:UpdatedAtVtx'
tpToStaUpdMuonAssociationFS.UseTracker = False
tpToStaUpdMuonAssociationFS.UseMuon = True

tpToStaRefitMuonAssociationFS.tracksTag = 'refittedStandAloneMuons'
tpToStaRefitMuonAssociationFS.UseTracker = False
tpToStaRefitMuonAssociationFS.UseMuon = True

tpToStaRefitUpdMuonAssociationFS.tracksTag = 'refittedStandAloneMuons:UpdatedAtVtx'
tpToStaRefitUpdMuonAssociationFS.UseTracker = False
tpToStaRefitUpdMuonAssociationFS.UseMuon = True

tpToGlbMuonAssociationFS.tracksTag = 'extractedGlobalMuons'
tpToGlbMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToGlbMuonAssociationFS.UseTracker = True
tpToGlbMuonAssociationFS.UseMuon = True

tpToTevFirstMuonAssociationFS.tracksTag = 'tevMuons:firstHit'
tpToTevFirstMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToTevFirstMuonAssociationFS.UseTracker = True
tpToTevFirstMuonAssociationFS.UseMuon = True

tpToTevPickyMuonAssociationFS.tracksTag = 'tevMuons:picky'
tpToTevPickyMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToTevPickyMuonAssociationFS.UseTracker = True
tpToTevPickyMuonAssociationFS.UseMuon = True

tpToTevDytMuonAssociationFS.tracksTag = 'tevMuons:dyt'
tpToTevDytMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToTevDytMuonAssociationFS.UseTracker = True
tpToTevDytMuonAssociationFS.UseMuon = True

tpToL3TkMuonAssociationFS.tracksTag = 'hltL3TkTracksFromL2'
tpToL3TkMuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToL3TkMuonAssociationFS.UseTracker = True
tpToL3TkMuonAssociationFS.UseMuon = False
tpToL3TkMuonAssociationFS.ignoreMissingTrackCollection = True
tpToL3TkMuonAssociationFS.UseSplitting = False
tpToL3TkMuonAssociationFS.UseGrouped = False

tpToL2MuonAssociationFS.tracksTag = 'hltL2Muons'
tpToL2MuonAssociationFS.UseTracker = False
tpToL2MuonAssociationFS.UseMuon = True
tpToL2MuonAssociationFS.ignoreMissingTrackCollection = True

tpToL2UpdMuonAssociationFS.tracksTag = 'hltL2Muons:UpdatedAtVtx'
tpToL2UpdMuonAssociationFS.UseTracker = False
tpToL2UpdMuonAssociationFS.UseMuon = True
tpToL2UpdMuonAssociationFS.ignoreMissingTrackCollection = True

tpToL3MuonAssociationFS.tracksTag = 'hltL3Muons'
tpToL3MuonAssociationFS.tpTag = 'mix:MergedTrackTruth'
tpToL3MuonAssociationFS.UseTracker = True
tpToL3MuonAssociationFS.UseMuon = True
tpToL3MuonAssociationFS.ignoreMissingTrackCollection = True
tpToL3MuonAssociationFS.UseSplitting = False
tpToL3MuonAssociationFS.UseGrouped = False



muonAssociationFastSim_seq = cms.Sequence(
        extractedMuonTracks_seq
        +seedsOfSTAmuons_seq
        +probeTracks+trackAssociatorByHits+(tpToTkMuonAssociationFS+tpToTkmuTrackAssociationFS) 
        +(tpToStaSeedAssociationFS+tpToStaMuonAssociationFS+tpToStaUpdMuonAssociationFS+tpToGlbMuonAssociationFS)
        +(tpToStaRefitMuonAssociationFS+tpToStaRefitUpdMuonAssociationFS)
        +(tpToTevFirstMuonAssociationFS+tpToTevPickyMuonAssociationFS+tpToTevDytMuonAssociationFS)
#        +tpToStaTrackAssociationFS+tpToStaUpdTrackAssociationFS+tpToGlbTrackAssociationFS
#        +tpToTevFirstTrackAssociationFS+tpToTevPickyTrackAssociationFS
        )
muonAssociationHLTFastSim_seq = cms.Sequence(
    tpToL2MuonAssociationFS+tpToL2UpdMuonAssociationFS+tpToL3MuonAssociationFS+tpToL3TkMuonAssociationFS
#    +tpToL2TrackAssociationFS+tpToL2UpdTrackAssociationFS+tpToL3TrackAssociationFS+tpToL3TkTrackTrackAssociationFS
    )

