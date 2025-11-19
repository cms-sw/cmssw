"""
This config fragment generates removes the energy deposites of LHE information for tau embedding. The selection step must be carried out beforehand.
It's normally used together with the LHE step.
To use this config fragment, a cmsDriver command like the following can be used:
```
cmsDriver.py \
	--step USER:TauAnalysis/MCEmbeddingTools/LHE_USER_cff.embeddingLHEProducerTask,RAW2DIGI,RECO \
	--processName LHEembeddingCLEAN \
	--data \
	--scenario pp \
	--eventcontent TauEmbeddingCleaning \
	--datatier RAWRECO \
	--procModifiers tau_embedding_cleaning,tau_embedding_mu_to_mu \
    --era ... \
    --conditions ... \
    --filein ... \
    --fileout ...
```
"""

import FWCore.ParameterSet.Config as cms
from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

# Define TrackAssociatorParameters for the tau embedding cleaning modules to use the correct collections from the selection step
tau_embedding_TrackAssociatorParameters = (
    TrackAssociatorParameterBlock.TrackAssociatorParameters.clone(
        CSCSegmentCollectionLabel=cms.InputTag("cscSegments", "", "SELECT"),
        CaloTowerCollectionLabel=cms.InputTag("towerMaker", "", "SELECT"),
        DTRecSegment4DCollectionLabel=cms.InputTag("dt4DSegments", "", "SELECT"),
        EBRecHitCollectionLabel=cms.InputTag("ecalRecHit", "EcalRecHitsEB", "SELECT"),
        EERecHitCollectionLabel=cms.InputTag("ecalRecHit", "EcalRecHitsEE", "SELECT"),
        HBHERecHitCollectionLabel=cms.InputTag("hbhereco", "", "SELECT"),
        HORecHitCollectionLabel=cms.InputTag("horeco", "", "SELECT"),
        ME0HitCollectionLabel=cms.InputTag("me0RecHits", "", "SELECT"),
        ME0SegmentCollectionLabel=cms.InputTag("me0Segments", "", "SELECT"),
        RPCHitCollectionLabel=cms.InputTag("rpcRecHits", "", "SELECT"),
        usePreshower=cms.bool(True),
        preselectMuonTracks=cms.bool(True),
    )
)
# some common parameters which are used by most of the modules
common_parameters = {
    "MuonCollection": cms.InputTag("selectedMuonsForEmbedding"),
    "TrackAssociatorParameters": tau_embedding_TrackAssociatorParameters,
    "cscDigiCollectionLabel": cms.InputTag("muonCSCDigis", "MuonCSCStripDigi"),
    "digiMaxDistanceX": cms.double(25.0),
    "dtDigiCollectionLabel": cms.InputTag("muonDTDigis"),
}

# The following modules are replaced by the correspondig ColCleaner versions, which removes the energy deposites of the measured event
# The replacement is done using the tau_embedding process modifier, which is included in all the different tau embedding process modifiers.
# Each of the following tau embedding cleaner modules has a comment indicating the code where the replacment takes place.

### Muon system modules

# replaced in RecoLocalMuon/CSCRecHitD/python/cscRecHitD_cfi.py
tau_embedding_csc2DRecHits_cleaner = cms.EDProducer(
    "CSCRecHitColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("csc2DRecHits", "", "SELECT")),
    **common_parameters
)

# replaced in RecoLocalMuon/CSCSegment/python/cscSegments_cfi.py
tau_embedding_cscSegments_cleaner = cms.EDProducer(
    "CSCSegmentColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("cscSegments", "", "SELECT")),
    **common_parameters
)

# replaced in RecoLocalMuon/DTRecHit/python/dt1DRecHits_LinearDriftFromDB_cfi.py
tau_embedding_dt1DCosmicRecHits_cleaner = cms.EDProducer(
    "DTRecHitColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("dt1DCosmicRecHits", "", "SELECT")),
    **common_parameters
)

# replaced in RecoLocalMuon/DTRecHit/python/dt1DRecHits_LinearDriftFromDB_cfi.py
tau_embedding_dt1DRecHits_cleaner = cms.EDProducer(
    "DTRecHitColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("dt1DRecHits", "", "SELECT")),
    **common_parameters
)
# replaced in RecoLocalMuon/DTSegment/python/dt4DSegments_MTPatternReco4D_LinearDriftFromDB_cfi.py
tau_embedding_dt4DCosmicSegments_cleaner = cms.EDProducer(
    "DTRecSegment4DColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("dt4DCosmicSegments", "", "SELECT")),
    **common_parameters
)

# replaced in RecoLocalMuon/DTSegment/python/dt4DSegments_MTPatternReco4D_LinearDriftFromDB_cfi.py
tau_embedding_dt4DSegments_cleaner = cms.EDProducer(
    "DTRecSegment4DColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("dt4DSegments", "", "SELECT")),
    **common_parameters
)

# replaced in RecoLocalMuon/RPCRecHit/python/rpcRecHits_cfi.py
tau_embedding_rpcRecHits_cleaner = cms.EDProducer(
    "RPCRecHitColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("rpcRecHits", "", "SELECT")),
    **common_parameters
)

### ECAL modules

# replaced in RecoLocalCalo/EcalRecProducers/python/ecalPreshowerRecHit_cfi.py
tau_embedding_ecalPreshowerRecHit_cleaner = cms.EDProducer(
    "EcalRecHitColCleaner",
    oldCollection=cms.VInputTag(
        cms.InputTag("ecalPreshowerRecHit", "EcalRecHitsES", "SELECT")
    ),
    **common_parameters
)

# replaced in RecoLocalCalo/EcalRecProducers/python/ecalRecHitGPU_cfi.py
# and in RecoLocalCalo/EcalRecProducers/python/ecalRecHit_cfi.py
tau_embedding_ecalRecHit_cleaner = cms.EDProducer(
    "EcalRecHitColCleaner",
    oldCollection=cms.VInputTag(
        cms.InputTag("ecalRecHit", "EcalRecHitsEB", "SELECT"),
        cms.InputTag("ecalRecHit", "EcalRecHitsEE", "SELECT"),
    ),
    **common_parameters
)

### HCAL modules

# replaced in RecoLocalCalo/Configuration/python/hcalGlobalReco_cff.py
tau_embedding_hbhereco_cleaner = cms.EDProducer(
    "HBHERecHitColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("hbhereco", "", "SELECT")),
    **common_parameters
)

# replaced in RecoLocalCalo/HcalRecProducers/python/HFPhase1Reconstructor_cfi.py
tau_embedding_hfreco_cleaner = cms.EDProducer(
    "HFRecHitColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("hfreco", "", "SELECT")),
    **common_parameters
)

# replaced in RecoLocalCalo/HcalRecProducers/python/HcalHitReconstructor_ho_cfi.py
tau_embedding_horeco_cleaner = cms.EDProducer(
    "HORecHitColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("horeco", "", "SELECT")),
    **common_parameters
)

### Tracker modules

# replaced in RecoTracker/IterativeTracking/python/InitialStepPreSplitting_cff.py
tau_embedding_siPixelClusters_cleaner = cms.EDProducer(
    "PixelColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("siPixelClusters", "", "SELECT")),
    **common_parameters
)

# replaced in RecoLocalTracker/SiStripClusterizer/python/SiStripClusterizer_RealData_cfi.py
tau_embedding_siStripClusters_cleaner = cms.EDProducer(
    "StripColCleaner",
    oldCollection=cms.VInputTag(cms.InputTag("siStripClusters", "", "SELECT")),
    **common_parameters
)
