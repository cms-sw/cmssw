# Run the validation of the standard MM
echo 'Start MixCollection validation in standard mode'

cmsRun mixCollectionTest_4sources_cfg.py #configure mixLowL
#cmsRun EDMtoMEConverter_standardMM_cfg.py
#mv DQM_V0001_R000000001__ConverterTester__Test__RECO.root DQM_V0001_R000000001__ConverterTester__Test__RECO_StandardMM.root

# Run the validation of the Step2 mode of the MM
echo 'Start MixCollection validation in step2 mode'

# Run the MM in the Step1 mode, produce the root file containing the CFs
cmsRun ../../../SimGeneral/MixingModule/test/mm_CFWriter_4sources_cfg.py # produce MixedData311.root
cmsRun mixCollectionTest_4sources_Step2_cfg.py

# Extract histograms and compare to the histograms from 
# the MM validation in the standard mode
echo 'Extract histograms and compare them'
#cmsRun EDMtoMEConverter_Step2_cfg.py
#mv DQM_V0001_R000000001__ConverterTester__Test__RECO.root DQM_V0001_R000000001__ConverterTester__Test__RECO_Step2MM.root

