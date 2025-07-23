"""
This config fragment is used to modify the RECO step to reconstruct the simulated taus (or electrons/muons) in the embedding samples.
The execution of a BeamSpotProducer is removed, as well as the vertex is put to the same as the measured one.
The Simulation HLT step must be carried out beforehand.
To use this config fragment, a cmsDriver command like the following can be used:
```
cmsDriver.py \
	--step RAW2DIGI,L1Reco,RECO:TauAnalysis/MCEmbeddingTools/Simulation_RECO_cff.reconstruction,RECOSIM \
	--processName SIMembedding \
	--mc \
	--beamspot DBrealistic \
	--geometry DB:Extended \
	--eventcontent TauEmbeddingSimReco \
	--datatier RAW-RECO-SIM \
    --era ... \
    --conditions ... \
    --filein ... \
    --fileout ...
```
"""
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from Configuration.StandardSequences.Reconstruction_cff import *  # this imports the standard reconstruction sequence, which is needed for the RECO step
from RecoTracker.IterativeTracking.InitialStep_cff import (
    firstStepPrimaryVerticesUnsorted,
)
from RecoTracker.IterativeTracking.InitialStepPreSplitting_cff import (
    firstStepPrimaryVerticesPreSplitting,
)
from RecoVertex.Configuration.RecoVertex_cff import offlinePrimaryVertices

# As we want to exploit the toModify and toReplaceWith features of the FWCore/ParameterSet/python/Config.py Modifier class,
# we need a general modifier that is always applied.
# maybe this can also be replaced by a specific embedding process modifier
generalModifier = run2_common | run3_common

# replace the vertice producers with a custom producer that uses the measured vertex position
generalModifier.toReplaceWith(offlinePrimaryVertices, cms.EDProducer('EmbeddingHltPixelVerticesProducer'))
generalModifier.toReplaceWith(firstStepPrimaryVerticesUnsorted, cms.EDProducer('EmbeddingHltPixelVerticesProducer'))
generalModifier.toReplaceWith(firstStepPrimaryVerticesPreSplitting, cms.EDProducer('EmbeddingHltPixelVerticesProducer'))

# remove the beam spot producer from the RECO step, as we want to use the measured one
globalreco_trackingTask.remove(offlineBeamSpotTask)
reconstruction_pixelTrackingOnlyTask.remove(offlineBeamSpotTask)
globalreco_trackingTask.remove(offlineBeamSpotTask)