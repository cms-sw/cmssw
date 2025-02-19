# Validation of the standard MM
echo 'Start MixCollection validation in standard mode'

cmsRun mixCollectionTest_StandardMM_cfg.py #configure mixLowL, produce histo file : histosMixCollStandardMM.root

# Extract histograms 
echo 'Extract histograms for the Standard mode'
cmsRun EDMtoMEConverter_standardMM_cfg.py
mv DQM_V0001_R000000001__ConverterTester__Test__RECO.root DQM_V0001_R000000001__ConverterTester__Test__RECO_StandardMM.root

########################################################

# Validation of the Step2 mode of the MM
# 1. Run the MM in the Step1 mode, produce the root file containing the CFs

echo 'Run the MM in the Step1 mode, produce the root file containing the CFs'
cmsRun ../../../SimGeneral/MixingModule/test/mm_CFWriter_cfg.py # produce MixedData311.root


# 2. Run the mix collection validation for the Step2 mode
echo 'Start MixCollection validation in step2 mode'
cmsRun mixCollectionTest_Step2_cfg.py # produce histo file histosMixCollStep2MM.root

# Extract histograms 
echo 'Extract histograms for the Step2 mode'
cmsRun EDMtoMEConverter_Step2_cfg.py
mv DQM_V0001_R000000001__ConverterTester__Test__RECO.root DQM_V0001_R000000001__ConverterTester__Test__RECO_Step2MM.root


#########################################################
# Comparison of the histograms from  the histosMixCollStandardMM.root and histosMixCollStep2MM.root files
# histosMixCollStandardMM.root is considered as a reference file
# root -l
# root[].x compareHistos.C("histosMixCollStep2MM","histosMixCollStandardMM")
