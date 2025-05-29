import FWCore.ParameterSet.Config as cms
from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock
from RecoLocalMuon.CSCRecHitD.cscRecHitD_cfi import csc2DRecHits
from RecoLocalMuon.CSCSegment.cscSegments_cfi import cscSegments
from RecoLocalMuon.DTRecHit.dt1DRecHits_LinearDriftFromDB_cfi import dt1DCosmicRecHits, dt1DRecHits
from RecoLocalMuon.DTSegment.dt4DSegments_MTPatternReco4D_LinearDriftFromDB_cfi import dt4DCosmicSegments, dt4DSegments
from RecoLocalMuon.RPCRecHit.rpcRecHits_cfi import rpcRecHits
# maybe replace with /RecoLocalMuon/Configuration/python/RecoLocalMuon_cff.
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import ecalPreshowerRecHit
# from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import hbhereco
from RecoLocalCalo.Configuration.hcalGlobalReco_cff import hcalGlobalRecoTask, hcalOnlyGlobalRecoTask
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import hfreco
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import horeco
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import siStripClusters
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cff import ecalCalibratedRecHitTask
from Configuration.StandardSequences.Reconstruction_cff import *
# from Configuration.ProcessModifiers.tau_embedding_mu_to_mu_cff import tau_embedding_mu_to_mu
# from Configuration.ProcessModifiers.tau_embedding_mu_to_e_cff import tau_embedding_mu_to_e
from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run3_common_cff import run3_common


TrackAssociatorParameterBlock.TrackAssociatorParameters.CSCSegmentCollectionLabel = cms.InputTag("cscSegments", "", "SELECT")
TrackAssociatorParameterBlock.TrackAssociatorParameters.CaloTowerCollectionLabel = cms.InputTag("towerMaker", "", "SELECT")
TrackAssociatorParameterBlock.TrackAssociatorParameters.DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments", "", "SELECT")
TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("ecalRecHit", "EcalRecHitsEB", "SELECT")
TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit", "EcalRecHitsEE", "SELECT")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("hbhereco", "", "SELECT")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("horeco", "", "SELECT")
TrackAssociatorParameterBlock.TrackAssociatorParameters.ME0HitCollectionLabel = cms.InputTag("me0RecHits", "", "SELECT")
TrackAssociatorParameterBlock.TrackAssociatorParameters.ME0SegmentCollectionLabel = cms.InputTag("me0Segments", "", "SELECT")
TrackAssociatorParameterBlock.TrackAssociatorParameters.RPCHitCollectionLabel = cms.InputTag("rpcRecHits", "", "SELECT")
TrackAssociatorParameterBlock.TrackAssociatorParameters.usePreshower = cms.bool(True)

common_parameters = {
    "MuonCollection": cms.InputTag("selectedMuonsForEmbedding"),
    "TrackAssociatorParameters": TrackAssociatorParameterBlock.TrackAssociatorParameters,
    "cscDigiCollectionLabel": cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    "digiMaxDistanceX": cms.double(25.0),
    "dtDigiCollectionLabel": cms.InputTag("muonDTDigis"),
}
(run2_common | run3_common).toReplaceWith(csc2DRecHits, cms.EDProducer("CSCRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("csc2DRecHits","","SELECT")),
    **common_parameters
))


(run2_common | run3_common).toReplaceWith(cscSegments, cms.EDProducer("CSCSegmentColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("cscSegments","","SELECT")),
    **common_parameters
))


(run2_common | run3_common).toReplaceWith(dt1DCosmicRecHits, cms.EDProducer("DTRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("dt1DCosmicRecHits","","SELECT")),
    **common_parameters
))


(run2_common | run3_common).toReplaceWith(dt1DRecHits, cms.EDProducer("DTRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("dt1DRecHits","","SELECT")),
    **common_parameters
))

(run2_common | run3_common).toReplaceWith(dt4DCosmicSegments, cms.EDProducer("DTRecSegment4DColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("dt4DCosmicSegments","","SELECT")),
    **common_parameters
))

(run2_common | run3_common).toReplaceWith(dt4DSegments, cms.EDProducer("DTRecSegment4DColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("dt4DSegments","","SELECT")),
    **common_parameters
))
(run2_common | run3_common).toReplaceWith(ecalPreshowerRecHit, cms.EDProducer("EcalRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES","SELECT")),
    **common_parameters
))
# This only worked by replacing the Task and not by replacing the Producer as in the other cases
# because the hbhereco is of the type SwitchProducerCUDA and not of the type EDProducer
# use walrus operator ":=" to give a label to the Producer
(run2_common | run3_common).toReplaceWith(hcalGlobalRecoTask, cms.Task(hbhereco := cms.EDProducer("HBHERecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("hbhereco","","SELECT")),
    **common_parameters
)))
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toReplaceWith(hcalOnlyGlobalRecoTask, cms.Task(hbhereco))

(run2_common | run3_common).toReplaceWith(hfreco, cms.EDProducer("HFRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("hfreco","","SELECT")),
    **common_parameters
))
(run2_common | run3_common).toReplaceWith(horeco, cms.EDProducer("HORecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("horeco","","SELECT")),
    **common_parameters
))
(run2_common | run3_common).toReplaceWith(rpcRecHits, cms.EDProducer("RPCRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("rpcRecHits","","SELECT")),
    **common_parameters
))
(run2_common | run3_common).toReplaceWith(siPixelClusters, cms.EDProducer("PixelColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("siPixelClusters","","SELECT")),
    **common_parameters
))
(run2_common | run3_common).toReplaceWith(siStripClusters, cms.EDProducer("StripColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("siStripClusters","","SELECT")),
    **common_parameters
))

(run2_common | run3_common).toReplaceWith(ecalCalibratedRecHitTask, cms.Task(ecalRecHit := cms.EDProducer("EcalRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB","SELECT"), cms.InputTag("ecalRecHit","EcalRecHitsEE","SELECT")),
    **common_parameters
)))