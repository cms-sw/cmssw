# Tau embedding method

The Tau Embedding runs in four steps:

1. Selection events with two muons
2. Cleaning the events from the two muons
3. Simulating the Z -> tau tau events
4. Merging the Simulated and Cleaned event

The RECO sequence must be run through for all these steps.
The input of the first step must be RAW, and the inputs of the other samples are the outputs of the step before.

1. **Selection**

   The FILTER step adds the code to select events with two muons suited for tau embedding and produces a collection where those muons are stored.
   ```bash
   cmsDriver.py \
      --step RAW2DIGI,L1Reco,RECO,PAT,FILTER:TauAnalysis/MCEmbeddingTools/Selection_FILTER_cff.makePatMuonsZmumuSelection \
      --processName SELECT \
      --data \
      --scenario pp \
      --eventcontent RAWRECO \
      --datatier RAWRECO \
      --outputCommands 'keep *_selectedMuonsForEmbedding_*_SELECT','keep *_patMuonsAfterID_*_SELECT','keep *_slimmedMuons_*_SELECT','keep *_slimmedMuonTrackExtras_*_SELECT','keep recoVertexs_offlineSlimmedPrimaryVertices_*_SELECT','keep *_firstStepPrimaryVertices_*_SELECT','keep *_ecalDrivenElectronSeeds_*_SELECT' \
      --conditions ... \
      --era ... \
      --filein ... \
      --fileout ...
   ```

2. **LHE and Cleaning**

   In this step two things are done:
   - The LHEProducer is run to generate the LHE file for the Z -> tau tau events in the USER step.
   - The energy deposits of the two muons selected in the previous step are removed from the event record. This is done by modifying the RECO sequence by replacing some producers with special cleaning producers.

   This step can be modified using the following `--procModifiers` options:
   - `tau_embedding_mu_to_mu`: Muons are simulated instead of a taus. That means this method is replacing a muon with a simulated muon. This is very useful for the validation of the method and for calculating some corrections.
   - `tau_embedding_mu_to_e`: Electrons are simulated instead of taus. This is used to calculate corrections.
  
   ```bash
   cmsDriver.py \
      --step USER:TauAnalysis/MCEmbeddingTools/LHE_USER_cff.embeddingLHEProducerTask,RAW2DIGI,RECO:TauAnalysis/MCEmbeddingTools/Cleaning_RECO_cff.reconstruction \
      --processName LHEembeddingCLEAN \
      --data \
      --scenario pp \
      --eventcontent RAWRECO \
      --datatier RAWRECO \
      --outputCommands 'drop *_*_*_SELECT','drop recoIsoDepositedmValueMap_muIsoDepositTk_*_*','drop recoIsoDepositedmValueMap_muIsoDepositTkDisplaced_*_*','drop *_ctppsProtons_*_*','drop *_ctppsLocalTrackLiteProducer_*_*','drop *_ctppsDiamondLocalTracks_*_*','drop *_ctppsDiamondRecHits_*_*','drop *_ctppsDiamondRawToDigi_*_*','drop *_ctppsPixelLocalTracks_*_*','drop *_ctppsPixelRecHits_*_*','drop *_ctppsPixelClusters_*_*','drop *_ctppsPixelDigis_*_*','drop *_totemRPLocalTrackFitter_*_*','drop *_totemRPUVPatternFinder_*_*','drop *_totemRPRecHitProducer_*_*','drop *_totemRPClusterProducer_*_*','drop *_totemRPRawToDigi_*_*','drop *_muonSimClassifier_*_*','keep *_patMuonsAfterID_*_SELECT','keep *_slimmedMuons_*_SELECT','keep *_selectedMuonsForEmbedding_*_SELECT','keep recoVertexs_offlineSlimmedPrimaryVertices_*_SELECT','keep *_firstStepPrimaryVertices_*_SELECT','keep *_offlineBeamSpot_*_SELECT','keep *_l1extraParticles_*_SELECT','keep TrajectorySeeds_*_*_*','keep recoElectronSeeds_*_*_*','keep *_generalTracks_*_LHEembeddingCLEAN','keep *_generalTracks_*_CLEAN','keep *_cosmicsVetoTracksRaw_*_LHEembeddingCLEAN','keep *_cosmicsVetoTracksRaw_*_CLEAN','keep *_electronGsfTracks_*_LHEembeddingCLEAN','keep *_electronGsfTracks_*_CLEAN','keep *_lowPtGsfEleGsfTracks_*_LHEembeddingCLEAN','keep *_lowPtGsfEleGsfTracks_*_CLEAN','keep *_displacedTracks_*_LHEembeddingCLEAN','keep *_displacedTracks_*_CLEAN','keep *_ckfOutInTracksFromConversions_*_LHEembeddingCLEAN','keep *_ckfOutInTracksFromConversions_*_CLEAN','keep *_muons1stStep_*_LHEembeddingCLEAN','keep *_muons1stStep_*_CLEAN','keep *_displacedMuons1stStep_*_LHEembeddingCLEAN','keep *_displacedMuons1stStep_*_CLEAN','keep *_conversions_*_LHEembeddingCLEAN','keep *_conversions_*_CLEAN','keep *_allConversions_*_LHEembeddingCLEAN','keep *_allConversions_*_CLEAN','keep *_particleFlowTmp_*_LHEembeddingCLEAN','keep *_particleFlowTmp_*_CLEAN','keep *_ecalDigis_*_LHEembeddingCLEAN','keep *_ecalDigis_*_CLEAN','keep *_hcalDigis_*_LHEembeddingCLEAN','keep *_hcalDigis_*_CLEAN','keep *_ecalRecHit_*_LHEembeddingCLEAN','keep *_ecalRecHit_*_CLEAN','keep *_ecalPreshowerRecHit_*_LHEembeddingCLEAN','keep *_ecalPreshowerRecHit_*_CLEAN','keep *_hbhereco_*_LHEembeddingCLEAN','keep *_hbhereco_*_CLEAN','keep *_horeco_*_LHEembeddingCLEAN','keep *_horeco_*_CLEAN','keep *_hfreco_*_LHEembeddingCLEAN','keep *_hfreco_*_CLEAN','keep *_standAloneMuons_*_LHEembeddingCLEAN','keep *_glbTrackQual_*_LHEembeddingCLEAN','keep *_externalLHEProducer_*_LHEembedding','keep *_externalLHEProducer_*_LHEembeddingCLEAN' \
      --conditions ... \
      --era ... \
      --filein ... \
      --fileout ...
   ```

3. **Simulation**

   This step is divided into three different steps to give the possibility to change to an older CMSSW version for the HLT step, if needed. This is needed because we want to run the same HLT menu which was used to record the data.
   For all three steps the beamspot needs to be set to the one measured in the data. For the generation and HLT step the vertex is also set to the one measured in the data.

   1. **Generation**

      The generation of the Z -> tau tau events is done with the Pythia8Hadronizer. The input is the LHE file produced in the previous step.

      This step needs to be modified using the following `--procModifiers` options to specify the final state of the taus:
      - `tau_embedding_emu`: One tau decays to an electron and the other to a muon.
      - `tau_embedding_etauh`: One tau decays to an electron and the other to a hadronic tau decay.
      - `tau_embedding_mutauh`: One tau decays to a muon and the other to a hadronic tau decay.
      - `tau_embedding_tauhtauh`: Both taus decay to hadronic tau decays.
      - `tau_embedding_mu_to_e`: Must also be specified in the LHE step. Simulates electrons instead of a taus.
      - `tau_embedding_mu_to_mu`: Must also be specified in the LHE step. Simulates muons instead of a taus.

      ```bash
      cmsDriver.py TauAnalysis/MCEmbeddingTools/python/Simulation_GEN_cfi.py \
         --step GEN,SIM,DIGI,L1,DIGI2RAW \
         --processName SIMembeddingpreHLT \
         --mc \
         --beamspot DBrealistic \
         --geometry DB:Extended \
         --eventcontent RAWSIM \
         --datatier RAWSIM \
         --outputCommands 'keep *_*_*_LHEembeddingCLEAN','keep *_*_*_SELECT','drop *_muonReducedTrackExtras_*_*','drop *_*_uncleanedConversions_*','drop *_diamondSampicLocalTracks_*_*','keep *_*_unsmeared_*', \
         --procModifiers tau_embedding_mutauh \
         --era ... \
         --conditions ... \
         --filein ... \
         --fileout ...
      ```

   2. **HLT**

      The HLT needs to be run with the same menu as used for the data taking. Therefore, this steps may need to be run in an older CMSSW version, where the HLT menu is still available.

      ```bash
      cmsDriver.py \
         --step HLT:TauAnalysis/MCEmbeddingTools/Simulation_HLT_customiser_cff.embeddingHLTCustomiser.Fake2 \
         --processName SIMembeddingHLT \
         --mc \
         --beamspot DBrealistic \
         --geometry DB:Extended \
         --eventcontent RAWSIM \
         --datatier RAWSIM \
         --outputCommands 'keep *_*_*_SELECT','keep *_*_*_LHEembeddingCLEAN','keep *_*_unsmeared_SIMembeddingpreHLT','keep DcsStatuss_hltScalersRawToDigi_*_*' \
         --era ... \
         --conditions ... \
         --filein ... \
         --fileout ...
      ```

   3. **Reconstruction**

      ```bash
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

4. **Merge**

   In the last step the simulated Z -> tau tau events are merged with the cleaned events from the second step. The output is a MINIAODSIM file.

   ```bash
   cmsDriver.py \
      --step USER:TauAnalysis/MCEmbeddingTools/Merging_USER_cff.merge_step,PAT \
      --processName MERGE \
      --data \
      --scenario pp \
      --eventcontent MINIAODSIM \
      --datatier USER \
      --inputCommands 'keep *_*_*_*' \
      --outputCommands 'drop *_*_*_SELECT','keep *_prunedGenParticles_*_MERGE','keep *_generator_*_SIMembeddingpreHLT','keep *_generator_*_SIMembeddingHLT','keep *_generator_*_SIMembedding','keep *_selectedMuonsForEmbedding_*_*','keep *_unpackedPatTrigger_*_*','keep patPackedGenParticles_packedGenParticles_*_*','keep recoGenParticles_prunedGenParticles_*_*','keep *_packedPFCandidateToGenAssociation_*_*','keep *_lostTracksToGenAssociation_*_*','keep LHEEventProduct_*_*_*','keep GenFilterInfo_*_*_*','keep GenLumiInfoHeader_generator_*_*','keep GenLumiInfoProduct_*_*_*','keep GenEventInfoProduct_generator_*_*','keep recoGenParticles_genPUProtons_*_*','keep *_slimmedGenJetsFlavourInfos_*_*','keep *_slimmedGenJets__*','keep *_slimmedGenJetsAK8__*','keep *_slimmedGenJetsAK8SoftDropSubJets__*','keep *_genMetTrue_*_*','keep LHERunInfoProduct_*_*_*','keep GenRunInfoProduct_*_*_*','keep *_genParticles_xyz0_*','keep *_genParticles_t0_*' \
      --conditions ... \
      --era ... \
      --filein ... \
      --fileout ...
   ```

5. **NanoAOD**

   If you want to create a NanoAOD file you can use the following command, so that a special table with information about the initial muons and event is included.

   ```bash
   cmsDriver.py \
      --step NANO:TauAnalysis/MCEmbeddingTools/Nano_cff.embedding_nanoAOD_seq \
      --data \
      --scenario pp \
      --eventcontent NANOAODSIM \
      --datatier NANOAODSIM \
      --outputCommands 'keep edmTriggerResults_*_*_SIMembeddingpreHLT','keep edmTriggerResults_*_*_SIMembeddingHLT','keep edmTriggerResults_*_*_SIMembedding','keep edmTriggerResults_*_*_MERGE','keep edmTriggerResults_*_*_NANO' \
      --conditions ... \
      --era ... \
      --filein ... \
      --fileout ...
   ```

You can find some working examples for different eras and CMSSW versions in this folder and in the test folder.