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
	--procModifiers tau_embedding_sim \
    --era ... \
    --conditions ... \
    --filein ... \
    --fileout ...
```
"""

import FWCore.ParameterSet.Config as cms

# replace the vertice producers with a custom producer that uses the measured vertex position
# The replacement is done in the following files:
# RecoTracker/IterativeTracking/python/InitialStep_cff.py
# RecoTracker/IterativeTracking/python/InitialStepPreSplitting_cff.py
# RecoVertex/Configuration/python/RecoVertex_cff.py
tau_embedding_correct_hlt_vertices = cms.EDProducer("EmbeddingHltPixelVerticesProducer")

# remove the beam spot producer from the RECO step, as we want to use the measured one in
# Configuration/StandardSequences/python/Reconstruction_cff.py
