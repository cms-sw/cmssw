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
	--eventcontent TauEmbeddingCleaning \
	--datatier RAWRECO \
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