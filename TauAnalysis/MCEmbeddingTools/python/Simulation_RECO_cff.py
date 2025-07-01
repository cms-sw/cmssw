"""
This config fragment is used to modify the RECO step to reconstruct the simulated taus (or electrons/muons) in the embedding samples.
Only the execution of a BeamSpotProducer is removed. 
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
from Configuration.StandardSequences.Reconstruction_cff import *  # this imports the standard reconstruction sequence, which is needed for the RECO step

globalreco_trackingTask.remove(offlineBeamSpotTask)
reconstruction_pixelTrackingOnlyTask.remove(offlineBeamSpotTask)
