"""
This config fragment generates removes the energy deposites of LHE information for tau embedding. The selection step must be carried out beforehand.
It's normally used together with the LHE step.
To use this config fragment, a cmsDriver command like the following can be used:
```
cmsDriver.py \
	--step USER:TauAnalysis/MCEmbeddingTools/LHE_USER_cff.embeddingLHEProducerTask,RAW2DIGI,RECO:TauAnalysis/MCEmbeddingTools/Cleaning_RECO_cff.reconstruction \
	--processName LHEembeddingCLEAN \
	--data \
	--scenario pp \
	--eventcontent RAWRECO \
	--datatier RAWRECO \
	--outputCommands 'drop *_*_*_SELECT','drop recoIsoDepositedmValueMap_muIsoDepositTk_*_*','drop recoIsoDepositedmValueMap_muIsoDepositTkDisplaced_*_*','drop *_ctppsProtons_*_*','drop *_ctppsLocalTrackLiteProducer_*_*','drop *_ctppsDiamondLocalTracks_*_*','drop *_ctppsDiamondRecHits_*_*','drop *_ctppsDiamondRawToDigi_*_*','drop *_ctppsPixelLocalTracks_*_*','drop *_ctppsPixelRecHits_*_*','drop *_ctppsPixelClusters_*_*','drop *_ctppsPixelDigis_*_*','drop *_totemRPLocalTrackFitter_*_*','drop *_totemRPUVPatternFinder_*_*','drop *_totemRPRecHitProducer_*_*','drop *_totemRPClusterProducer_*_*','drop *_totemRPRawToDigi_*_*','drop *_muonSimClassifier_*_*','keep *_patMuonsAfterID_*_SELECT','keep *_slimmedMuons_*_SELECT','keep *_selectedMuonsForEmbedding_*_SELECT','keep recoVertexs_offlineSlimmedPrimaryVertices_*_SELECT','keep *_firstStepPrimaryVertices_*_SELECT','keep *_offlineBeamSpot_*_SELECT','keep *_l1extraParticles_*_SELECT','keep TrajectorySeeds_*_*_*','keep recoElectronSeeds_*_*_*','keep *_generalTracks_*_LHEembeddingCLEAN','keep *_generalTracks_*_CLEAN','keep *_cosmicsVetoTracksRaw_*_LHEembeddingCLEAN','keep *_cosmicsVetoTracksRaw_*_CLEAN','keep *_electronGsfTracks_*_LHEembeddingCLEAN','keep *_electronGsfTracks_*_CLEAN','keep *_lowPtGsfEleGsfTracks_*_LHEembeddingCLEAN','keep *_lowPtGsfEleGsfTracks_*_CLEAN','keep *_displacedTracks_*_LHEembeddingCLEAN','keep *_displacedTracks_*_CLEAN','keep *_ckfOutInTracksFromConversions_*_LHEembeddingCLEAN','keep *_ckfOutInTracksFromConversions_*_CLEAN','keep *_muons1stStep_*_LHEembeddingCLEAN','keep *_muons1stStep_*_CLEAN','keep *_displacedMuons1stStep_*_LHEembeddingCLEAN','keep *_displacedMuons1stStep_*_CLEAN','keep *_conversions_*_LHEembeddingCLEAN','keep *_conversions_*_CLEAN','keep *_allConversions_*_LHEembeddingCLEAN','keep *_allConversions_*_CLEAN','keep *_particleFlowTmp_*_LHEembeddingCLEAN','keep *_particleFlowTmp_*_CLEAN','keep *_ecalDigis_*_LHEembeddingCLEAN','keep *_ecalDigis_*_CLEAN','keep *_hcalDigis_*_LHEembeddingCLEAN','keep *_hcalDigis_*_CLEAN','keep *_ecalRecHit_*_LHEembeddingCLEAN','keep *_ecalRecHit_*_CLEAN','keep *_ecalPreshowerRecHit_*_LHEembeddingCLEAN','keep *_ecalPreshowerRecHit_*_CLEAN','keep *_hbhereco_*_LHEembeddingCLEAN','keep *_hbhereco_*_CLEAN','keep *_horeco_*_LHEembeddingCLEAN','keep *_horeco_*_CLEAN','keep *_hfreco_*_LHEembeddingCLEAN','keep *_hfreco_*_CLEAN','keep *_standAloneMuons_*_LHEembeddingCLEAN','keep *_glbTrackQual_*_LHEembeddingCLEAN','keep *_externalLHEProducer_*_LHEembedding','keep *_externalLHEProducer_*_LHEembeddingCLEAN' \
	--procModifiers tau_embedding_mu_to_mu \
    --era ... \
    --conditions ... \
    --filein ... \
    --fileout ...
```
"""

# The order of the imports is important, as some modules depend on others.
# It breaks if you run isort on this file.
# I haven't found out which module is responsible for the breakage, but it is reproducible.
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock
from RecoLocalMuon.CSCRecHitD.cscRecHitD_cfi import csc2DRecHits
from RecoLocalMuon.CSCSegment.cscSegments_cfi import cscSegments
from RecoLocalMuon.DTRecHit.dt1DRecHits_LinearDriftFromDB_cfi import (
    dt1DCosmicRecHits,
    dt1DRecHits,
)
from RecoLocalMuon.DTSegment.dt4DSegments_MTPatternReco4D_LinearDriftFromDB_cfi import (
    dt4DCosmicSegments,
    dt4DSegments,
)
from RecoLocalMuon.RPCRecHit.rpcRecHits_cfi import rpcRecHits
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import ecalPreshowerRecHit
# maybe replace with /RecoLocalMuon/Configuration/python/RecoLocalMuon_cff.
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cff import ecalCalibratedRecHitTask

from RecoLocalCalo.Configuration.hcalGlobalReco_cff import (
    hcalGlobalRecoTask,
    hcalOnlyGlobalRecoTask,
)
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import hfreco
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import horeco
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import siStripClusters
from Configuration.StandardSequences.Reconstruction_cff import *  # this imports the standard reconstruction sequence, which is needed for the RECO step

# As we want to exploit the toModify and toReplaceWith features of the FWCore/ParameterSet/python/Config.py Modifier class,
# we need a general modifier that is always applied.
# maybe this can also be replaced by a specific embedding process modifier
generalModifier = run2_common | run3_common

# Adjust sources for the TrackAssociatorParameters
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

# some common parameters which are used by most of the modules
common_parameters = {
    "MuonCollection": cms.InputTag("selectedMuonsForEmbedding"),
    "TrackAssociatorParameters": TrackAssociatorParameterBlock.TrackAssociatorParameters,
    "cscDigiCollectionLabel": cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    "digiMaxDistanceX": cms.double(25.0),
    "dtDigiCollectionLabel": cms.InputTag("muonDTDigis"),
}

# The following modules are replaced by the correspondig ColCleaner versions, which remove the energy deposites of the measured event

### Muon system modules
generalModifier.toReplaceWith(csc2DRecHits, cms.EDProducer("CSCRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("csc2DRecHits","","SELECT")),
    **common_parameters
))

generalModifier.toReplaceWith(cscSegments, cms.EDProducer("CSCSegmentColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("cscSegments","","SELECT")),
    **common_parameters
))

generalModifier.toReplaceWith(dt1DCosmicRecHits, cms.EDProducer("DTRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("dt1DCosmicRecHits","","SELECT")),
    **common_parameters
))

generalModifier.toReplaceWith(dt1DRecHits, cms.EDProducer("DTRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("dt1DRecHits","","SELECT")),
    **common_parameters
))

generalModifier.toReplaceWith(dt4DCosmicSegments, cms.EDProducer("DTRecSegment4DColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("dt4DCosmicSegments","","SELECT")),
    **common_parameters
))

generalModifier.toReplaceWith(dt4DSegments, cms.EDProducer("DTRecSegment4DColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("dt4DSegments","","SELECT")),
    **common_parameters
))

generalModifier.toReplaceWith(rpcRecHits, cms.EDProducer("RPCRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("rpcRecHits","","SELECT")),
    **common_parameters
))

### ECAL modules 
generalModifier.toReplaceWith(ecalPreshowerRecHit, cms.EDProducer("EcalRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES","SELECT")),
    **common_parameters
))

# use walrus operator ":=" to give a label to the Producer
generalModifier.toReplaceWith(ecalCalibratedRecHitTask, cms.Task(ecalRecHit := cms.EDProducer("EcalRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB","SELECT"), cms.InputTag("ecalRecHit","EcalRecHitsEE","SELECT")),
    **common_parameters
)))

### HCAL modules
# This only worked by replacing the Task and not by replacing the Producer as in the other cases
# because the hbhereco is of the type SwitchProducerCUDA and not of the type EDProducer
generalModifier.toReplaceWith(hcalGlobalRecoTask, cms.Task(hbhereco := cms.EDProducer("HBHERecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("hbhereco","","SELECT")),
    **common_parameters
)))

run3_HB.toReplaceWith(hcalOnlyGlobalRecoTask, cms.Task(hbhereco))

generalModifier.toReplaceWith(hfreco, cms.EDProducer("HFRecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("hfreco","","SELECT")),
    **common_parameters
))

generalModifier.toReplaceWith(horeco, cms.EDProducer("HORecHitColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("horeco","","SELECT")),
    **common_parameters
))

### Tracker modules
generalModifier.toReplaceWith(siPixelClusters, cms.EDProducer("PixelColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("siPixelClusters","","SELECT")),
    **common_parameters
))

generalModifier.toReplaceWith(siStripClusters, cms.EDProducer("StripColCleaner",
    oldCollection = cms.VInputTag(cms.InputTag("siStripClusters","","SELECT")),
    **common_parameters
))