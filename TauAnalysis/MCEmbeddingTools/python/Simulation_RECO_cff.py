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
	--eventcontent RAWRECOSIMHLT \
	--datatier RAW-RECO-SIM \
	--outputCommands 'keep *_*_*_LHEembeddingCLEAN','keep *_*_*_SELECT','keep *_genParticles_*_SIMembedding','keep *_standAloneMuons_*_SIMembedding','keep *_glbTrackQual_*_SIMembedding','keep *_generator_*_SIMembedding','keep *_addPileupInfo_*_SIMembedding','keep *_selectedMuonsForEmbedding_*_*','keep *_slimmedAddPileupInfo_*_*','keep *_embeddingHltPixelVertices_*_*','keep *_*_vertexPosition_*','keep recoMuons_muonsFromCosmics_*_*','keep recoTracks_cosmicMuons1Leg_*_*','keep recoMuons_muonsFromCosmics1Leg_*_*','keep *_muonDTDigis_*_*','keep *_muonCSCDigis_*_*','keep TrajectorySeeds_*_*_*','keep recoElectronSeeds_*_*_*','drop recoIsoDepositedmValueMap_muIsoDepositTk_*_*','drop recoIsoDepositedmValueMap_muIsoDepositTkDisplaced_*_*','drop *_ctppsProtons_*_*','drop *_ctppsLocalTrackLiteProducer_*_*','drop *_ctppsDiamondLocalTracks_*_*','drop *_ctppsDiamondRecHits_*_*','drop *_ctppsDiamondRawToDigi_*_*','drop *_ctppsPixelLocalTracks_*_*','drop *_ctppsPixelRecHits_*_*','drop *_ctppsPixelClusters_*_*','drop *_ctppsPixelDigis_*_*','drop *_totemRPLocalTrackFitter_*_*','drop *_totemRPUVPatternFinder_*_*','drop *_totemRPRecHitProducer_*_*','drop *_totemRPClusterProducer_*_*','drop *_totemRPRawToDigi_*_*','drop *_muonSimClassifier_*_*','keep *_generalTracks_*_SIMembedding','keep *_cosmicsVetoTracksRaw_*_SIMembedding','keep *_electronGsfTracks_*_SIMembedding','keep *_lowPtGsfEleGsfTracks_*_SIMembedding','keep *_displacedTracks_*_SIMembedding','keep *_ckfOutInTracksFromConversions_*_SIMembedding','keep *_muons1stStep_*_SIMembedding','keep *_displacedMuons1stStep_*_SIMembedding','keep *_conversions_*_SIMembedding','keep *_allConversions_*_SIMembedding','keep *_particleFlowTmp_*_SIMembedding','keep *_ecalDigis_*_SIMembedding','keep *_hcalDigis_*_SIMembedding','keep *_ecalRecHit_*_SIMembedding','keep *_ecalPreshowerRecHit_*_SIMembedding','keep *_hbhereco_*_SIMembedding','keep *_horeco_*_SIMembedding','keep *_hfreco_*_SIMembedding','keep *_*_unsmeared_SIMembeddingpreHLT','keep *_hltScalersRawToDigi_*_SIMembeddingHLT' \
    --era ... \
    --conditions ... \
    --filein ... \
    --fileout ...
```
"""
from Configuration.StandardSequences.Reconstruction_cff import *  # this imports the standard reconstruction sequence, which is needed for the RECO step

globalreco_trackingTask.remove(offlineBeamSpotTask)
reconstruction_pixelTrackingOnlyTask.remove(offlineBeamSpotTask)
