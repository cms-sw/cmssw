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
      --eventcontent TauEmbeddingSelection \
      --datatier RAWRECO \
      --conditions ... \
      --era ... \
      --filein ... \
      --fileout ...
   ```

2. **LHE and Cleaning**

   In this step two things are done:
   - The LHEProducer is run to generate the LHE file for the Z -> tau tau events in the USER step.
   - The energy deposits of the two muons selected in the previous step are removed from the event record. This is done by modifying the RECO sequence by replacing some producers with special cleaning producers. To achieve this the `tau_embedding_cleaning` process modifier has to be used.

   This step can be modified using the following `--procModifiers` options:
   - `tau_embedding_mu_to_mu`: Muons are simulated instead of a taus. That means this method is replacing a muon with a simulated muon. This is very useful for the validation of the method and for calculating some corrections.
   - `tau_embedding_mu_to_e`: Electrons are simulated instead of taus. This is used to calculate corrections.
  
   ```bash
   cmsDriver.py \
      --step USER:TauAnalysis/MCEmbeddingTools/LHE_USER_cff.embeddingLHEProducerTask,RAW2DIGI,RECO \
      --processName LHEembeddingCLEAN \
      --data \
      --scenario pp \
      --eventcontent TauEmbeddingCleaning \
      --datatier RAWRECO \
      --procModifiers tau_embedding_cleaning,tau_embedding_mu_to_mu \
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
         --eventcontent TauEmbeddingSimGen \
         --datatier RAWSIM \
         --procModifiers tau_embedding_sim,tau_embedding_mutauh \
         --era ... \
         --conditions ... \
         --filein ... \
         --fileout ...
      ```

   2. **HLT**

      The HLT needs to be run with the same menu as used for the data taking. Therefore, this steps may need to be run in an older CMSSW version, where the HLT menu is still available.

      ```bash
      cmsDriver.py \
         --step HLT:Fake2+TauAnalysis/MCEmbeddingTools/Simulation_HLT_customiser_cff.embeddingHLTCustomiser \
         --processName SIMembeddingHLT \
         --mc \
         --beamspot DBrealistic \
         --geometry DB:Extended \
         --eventcontent TauEmbeddingSimHLT \
         --datatier RAWSIM \
         --era ... \
         --conditions ... \
         --filein ... \
         --fileout ...
      ```

   3. **Reconstruction**

      ```bash
      cmsDriver.py \
      --step RAW2DIGI,L1Reco,RECO,RECOSIM \
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

4. **Merge**

   In the last step the simulated Z -> tau tau events are merged with the cleaned events from the second step. The output is a MINIAODSIM file.

   ```bash
   cmsDriver.py \
      --step USER:TauAnalysis/MCEmbeddingTools/Merging_USER_cff.merge_step,PAT \
      --processName MERGE \
      --data \
      --scenario pp \
      --eventcontent TauEmbeddingMergeMINIAOD \
      --datatier USER \
      --procModifiers tau_embedding_merging \
      --inputCommands 'keep *_*_*_*' \
      --conditions ... \
      --era ... \
      --filein ... \
      --fileout ...
   ```

5. **NanoAOD**

   If you want to create a NanoAOD file you can use the following command, so that a special table with information about the initial muons and event is included.

   ```bash
   cmsDriver.py \
      --step NANO:@TauEmbedding \
      --data \
      --scenario pp \
      --eventcontent TauEmbeddingNANOAOD \
      --datatier NANOAODSIM \
      --conditions ... \
      --era ... \
      --filein ... \
      --fileout ...
   ```

You can find some working examples for different eras and CMSSW versions in this folder and in the test folder.