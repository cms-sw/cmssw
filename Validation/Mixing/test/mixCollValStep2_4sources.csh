# Validation of the standard MM
echo 'Start MixCollection validation in standard mode'

cmsRun mixCollectionTest_4sources_cfg.py #configure mixLowL,  produce histo file : histosMixCollStandardMM4sources.root

# Extract histograms 
echo 'Extract histograms for the Standard mode'
cmsRun EDMtoMEConverter_standardMM_cfg.py
mv DQM_V0001_R000000001__ConverterTester__Test__RECO.root DQM_V0001_R000000001__ConverterTester__Test__RECO_StandardMM4sources.root

########################################################

# Validation of the Step2 mode of the MM

# 1. Run the MM in the Step1 mode, produce the root file containing the CFs
cmsRun ../../../SimGeneral/MixingModule/test/mm_CFWriter_4sources_cfg.py # produce MixedData311_4sources.root

# 2. Run the mix collection validation for the Step2 mode
echo 'Start MixCollection validation in step2 mode'
cmsRun mixCollectionTest_4sources_Step2_cfg.py # produce histo file histosMixCollStep2MM4sources.root

# Extract histograms 
echo 'Extract histograms for the Step2 mode'
cmsRun EDMtoMEConverter_Step2_cfg.py
mv DQM_V0001_R000000001__ConverterTester__Test__RECO.root DQM_V0001_R000000001__ConverterTester__Test__RECO_Step2MM4sources.root


#########################################################
# Comparison of the histograms from  the histosMixCollStandardMM4sources.root and histosMixCollStep2MM4sources.root files
# histosMixCollStandardMM4sources.root is considered as a reference file
# root -l
# root[].x compareHistos.C("histosMixCollStep2MM4sources","histosMixCollStandardMM4sources")
