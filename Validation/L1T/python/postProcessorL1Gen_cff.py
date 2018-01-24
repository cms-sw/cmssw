import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

postProcessorL1Gen = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("L1T/L1TStage2uGT/"),
    efficiency = cms.vstring(
       "Ratio_First_Bunch_In_Train 'Trigger Bits vs BX' first_bunch_in_train den_first_bunch_in_train", 
       "Ratio_Last_Bunch_In_Train 'Trigger Bits vs BX' last_bunch_in_train den_last_bunch_in_train", 
       "Ratio_Isolated_Bunch_In_Train 'Trigger Bits vs BX' isolated_bunch den_isolated_bunch_in_train", 

    ),
    resolution = cms.vstring(),
    outputFileName = cms.untracked.string(""),
    verbose = cms.untracked.uint32(0)
)

L1GenPostProcessor = cms.Sequence(postProcessorL1Gen)
