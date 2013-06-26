#!/bin/tcsh
#$1 is the argument, ex : source ReValMove.csh 600

mv TTbar_Startup/DQM_V0001_R000000001__POG__BTAG__BJET.root    BTagRelVal_TTbar_Startup_$1.root
mv TTbar_Startup_PU/DQM_V0001_R000000001__POG__BTAG__BJET.root BTagRelVal_TTbar_Startup_PU_$1.root
mv TTbar_FastSim/DQM_V0001_R000000001__POG__BTAG__BJET.root    BTagRelVal_TTbar_FastSim_$1.root
mv QCD_Startup/DQM_V0001_R000000001__POG__BTAG__BJET.root      BTagRelVal_QCD_Startup_$1.root
#mv DATA/DQM_V0001_R000000001__POG__BTAG__BJET.root             BTagRelVal_DATA_$1.root
