#! /bin/bash


# MET RunRelVal script
# by: Michael Schmitt and Bobby Scurlock (The University of Florida)
##### Modification by : Ronny Remington (UF) 6/12/08
##### Modification by : Dayong Wang 12/2009
##### Modification by : Samantha Hewamanage 01/2012
##### This script will validate all 12 met collections, both calotower thresholds, ECAL, and HCAL : Takes 14 mins to run over all layers for a given sample


# user input area:

dirlist="QCDFlat TTbar"

CaloMetList="met metNoHF metHO metNoHFHO metOpt metOptNoHF metOptHO metOptNoHFHO"
tcMetList="tcMet"
GenMetList="genMetTrue genMetCalo genMetCaloAndNonPrompt"
CaloTowerList="SchemeB"
MCaloTowerList="SchemeB Optimized"

datadirRef="/uscms_data/d3/samantha/METRelValTesting_new/CMSSW_5_0_0/src/Validation/RecoMET/test/FullSim/500"
datadirNew="/uscms_data/d3/samantha/METRelValTesting_new/CMSSW_5_0_0/src/Validation/RecoMET/test/FullSim/500g4emtest"

release_ref="500"
release_new="500_g4emtest"

cond_ref="500"
cond_new="500_g4emtest"

# error handling

function fail { echo "$0: line $LINENO: exit status $?"; exit 1; }

trap fail ERR;

# get program options (all OFF by default)
runAll=0; runMET=0; runCaloTowerRecHits=0; runECALRecHits=0; runHCALRecHits=0; isFastFull=0; isFastFast=0; isFullFull=1; islocal=0;RESULTDIR=results;

function usage { echo "usage: $0 [-mchelto]"; echo; }
while getopts ":mchelto" option; do
  case $option in
    m ) runAll=0; runMET=1;;
    c ) runAll=0; runCaloTowerRecHits=1;;
    e ) runAll=0; runECALRecHits=1;;
    h ) runAll=0; runHCALRecHits=1;;
    l ) isFullFull=0; isFastFull=1; RESULTDIR=results-fastfull;;
    t ) isFullFull=0; isFastFast=1; RESULTDIR=results-fastfast;;
    o ) isFullFull=0; islocal=1; RESULTDIR=results-local;;
   \? ) usage; exit 1;;
    * ) usage; exit 2;;
  esac;
done;


if [ $isFastFull -eq 1 ] || [ $isFastFast -eq 1 ]; then 
    dirlist="TTbar QCD_FlatPt_15_3000";
fi

if [ $isFastFull -eq 1 ]; then 
    release=$release_new;
    release_ref=$release"_FullSim";
    release_new=$release"_FastSim";
fi
# ## for 64bits 
#     release=$release_new;
#     release_ref=$release"_32bit";
#     release_new=$release"_64bit";

echo $isFullFull $isFastFull $isFastFast $dirlist

if [ $# -eq 0 ]; then
  echo "All options are off by default. Please specify an option."
  usage; exit 3;
fi

# make sure the latest code is being used
echo "Checking to make sure latest code is being used...";
#gmake all > /dev/null;
gmake;

# add the executable area to the PATH
export PATH=`pwd`/bin:$PATH


# make the results working area
if [ ! -d $RESULTDIR ]; then mkdir $RESULTDIR; fi;

# place required html/images in RelVal area for later publishing
if [ ! -d $RESULTDIR/html ]; then cp -r html_templates $RESULTDIR/html; fi; 
if [ ! -d $RESULTDIR/images ]; then cp -r image_templates $RESULTDIR/images; fi;

# enter the RelVal area
cd $RESULTDIR;

# start a new SummaryTable for this run
if [ -f index.html ]; then rm index.html; fi;
cp html/SummaryTable-Top.html index.html;

removeAll=0; removeNone=0;
for i in $dirlist; do

  # if old directories exist, prompt for user action
  if [ -d $i -a $removeAll -eq 0 -a $removeNone -eq 0 ]; then

    while true; do
      echo -n "Directory $i exists. Delete? (yes/never/always): ";
      read -n1 delopt; echo;
      case `echo $delopt | tr "A-Z" "a-z"` in
        y) echo -e "Deleting!\n\nDeleting old directory $i"; rm -r $i; break;;
        n) echo "Continuing previous validation."; removeNone=1; break;;
        a) echo -e "Deleting!\n\nDeleting old directory $i"; rm -r $i; removeAll=1; break;;
       \?) echo "Huh? Unknown option: $delopt";;
        *) echo "Huh? Unknown option: $delopt";;
      esac;
    done;

  # directory exists and removeAll flag is set...
  elif [ -d $i -a $removeAll -eq 1 ]; then echo "Deleting old directory $i"; rm -r $i; fi;

  # add the appropriate lines to the SummaryTable.html file
  cat html/SummaryTable-Block.html | sed -e "s/INSERT_DIR/$i/g" >> index.html;

  # create and enter the working directory
  if [ ! -d $i ]; then mkdir $i; fi; cd $i;

  # create a dummy MET area if needed
  if [ ! -d MET ]; then

    mkdir MET; cd MET;

    ln -s ../../images/preview.gif .;
    ln -s ../../images/NotDone_button.gif Result-button.gif;
    ln -s ../../html/Not_done.html Compatibility_with_Reference_Histograms.html;
    touch job_not_run; cd ..;
  fi;

  # create a dummy CaloTowers area if needed
  if [ ! -d CaloTowers ]; then
    mkdir CaloTowers; cd CaloTowers;
    ln -s ../../images/preview.gif .;
    ln -s ../../images/NotDone_button.gif Result-button.gif;
    ln -s ../../html/Not_done.html Compatibility_with_Reference_Histograms.html;
    touch job_not_run; cd ..;
  fi;

  # create a dummy HCAL area if needed
  if [ ! -d HCAL ]; then
    mkdir HCAL; cd HCAL;
    ln -s ../../images/preview.gif .;
    ln -s ../../images/NotDone_button.gif Result-button.gif;
    ln -s ../../html/Not_done.html Compatibility_with_Reference_Histograms.html;
    touch job_not_run; cd ..;
  fi;

  # create a dummy ECAL area if needed
  if [ ! -d ECAL ]; then
    mkdir ECAL; cd ECAL;
    ln -s ../../images/preview.gif .;
    ln -s ../../images/NotDone_button.gif Result-button.gif;
    ln -s ../../html/Not_done.html Compatibility_with_Reference_Histograms.html;
    touch job_not_run; cd ..;
  fi;
  cd ..;


done;

echo Beginning Job at `date`;
# finish the SummaryTable.html file
cat html/SummaryTable-Bottom.html >> index.html;

for i in $dirlist; do

  # enter area
  if [ $islocal -eq 1 ]; then

      refRoot=$datadirRef/METTester_data_$i.root;
      newRoot=$datadirNew/METTester_data_$i.root;
  fi

  if [ $isFullFull -eq 1 ]; then
      refRoot=$datadirRef/DQM_V000?_R000000001__RelVal"$i"__CMSSW_"$release_ref"-"$cond_ref"-v?__*.root;
      newRoot=$datadirNew/DQM_V000?_R000000001__RelVal"$i"__CMSSW_"$release_new"-"$cond_new"-v?__*.root;
# ## for 64 vs 32 bit
#       refRoot=$datadirRef/DQM_V000?_R000000001__RelVal"$i"__CMSSW_"$release"-"$cond_ref"-v?__GEN-SIM-RECO.root;
#       newRoot=$datadirNew/DQM_V000?_R000000001__RelVal"$i"__CMSSW_"$release"-"$cond_new"-v?__GEN-SIM-RECO.root;


  fi
  
  if [ $isFastFast -eq 1 ]; then

      refRoot=$datadirRef/DQM_V000?_R000000001__RelVal"$i"__CMSSW_"$release_ref"-"$cond_ref"_FastSim-v?__GEN-SIM-DIGI-RECO.root;
      refRoot=$datadirRef/DQM_V000?_R000000001__RelVal"$i"__CMSSW_"$release_ref"-"$cond_ref"_FastSim_64bit-v?__GEN-SIM-DIGI-RECO.root;
      newRoot=$datadirNew/DQM_V000?_R000000001__RelVal"$i"__CMSSW_"$release_new"-"$cond_new"_FastSim-v?__GEN-SIM-DIGI-RECO.root;
  fi

  if [ $isFastFull -eq 1 ]; then
      echo "!!!!!!!!!!!!!!   " $isFastFull  $release; 
      refRoot=$datadirRef/DQM_V000?_R000000001__RelVal"$i"__CMSSW_"$release"-"$cond_new"-v?__GEN-SIM-RECO.root;
      refRoot=$datadirRef/DQM_V000?_R000000001__RelVal"$i"__CMSSW_"$release"-"$cond_new"-v?__DQM.root;
      newRoot=$datadirNew/DQM_V000?_R000000001__RelVal"$i"__CMSSW_"$release"-"$cond_new"_FastSim-v?__GEN-SIM-DIGI-RECO.root;
  fi

  if [ -e $refRoot ]; then
	  echo "Found $refRoot"
  else
	  echo "NOT FOUND!! $refRoot"
	  exit
  fi

  if [ -e $newRoot ]; then
	  echo "Found $newRoot"
	  exit
  else
	  echo "NOT FOUND!! $newRoot"
  fi

  #========Run MET Validation=============#
  cd $i/MET;
  failtrue=0
#  if [ -f Result-button.gif ]; then rm Result-button.gif; fi;
  if [ -f METSummaryTable.html ]; then rm METSummaryTable.html; fi;

  # job has not yet been run and is requested
  if [ -f job_not_run -a $runMET -eq 1 -o -f job_not_run -a $runAll -eq 1 ]; then

      sed -e "s|INSERT_MET|METSummaryTable|" ../../index.html > ../../index2.html;
      mv ../../index2.html ../../index.html;

      cp ../../html/MET/METSummaryTable_Top.html ../../html/MET/METSummaryTable.html;


  if [ $islocal -eq 1 ]; then
      Base=DQMData/RecoMETV/MET_Global;
  else
      Base="DQMData/Run 1/JetMET/Run summary/METv";
  fi
  
  for iCaloMet in $CaloMetList ; do 

      cat ../../html/MET/METSummaryTable_Middle.html | sed -e "s/INSERT_LIST/$iCaloMet/g" >> ../../html/MET/METSummaryTable.html;

      echo -n "Examining $iCaloMet................ ";
      mkdir $iCaloMet;
      cd $iCaloMet;

    #Define directory within histogram root file where this met collection lives
      dir=${Base}/${iCaloMet};
    # run Root
#      echo "metplotCompare" $refRoot $newRoot "$dir" $release_new $release_ref
      metplotCompare $refRoot $newRoot "$dir" $release_new $release_ref > MET-log.txt 2> MET-err.txt;
#      metplotCompare 
      result=`cat MET-log.txt | grep "Final Result:" | cut -f3 -d' '`;

    # clean up the area
      if [ -f Result-button.gif ]; then rm Result-button.gif; fi;
      if [ -f Compatibility_with_Reference_Histograms.html ]; then
	  rm Compatibility_with_Reference_Histograms.html; fi;
      if [ -f job_not_run ]; then rm job_not_run; fi;
      
    # link to the appropriate html pages
      ln -s ../../../html/MET/${iCaloMet}/*.html .;

      case $result in
	  "pass") ln -s ../../../images/Pass_button.gif Result-button.gif; echo "pass";;
	  "fail") ln -s ../../../images/Fail_button.gif Result-button.gif; failtrue=1;  echo "fail";;
	  "no_data") ln -s ../../../images/NoData_button.gif Result-button.gif; echo "no data";
              rm Compatibility_with_Reference_Histograms.html;
              ln -s ../../../html/No_data_available.html Compatibility_with_Reference_Histograms.html;;
	  "not_done") ln -s ../../../images/NotDone_button.gif Result-button.gif;
              rm Compatibility_with_Reference_Histograms.html;
              ln -s ../../../html/Not_done.html Compatibility_with_Reference_Histograms.html;;
	  "")     echo "Root has experienced a crash prior to completion"; fail;;
	  "*")    echo "unexpected result: $result"; fail;;
      esac
      cd ../

  done;


  for iGenMet in $GenMetList ; do 

    #=====  EXAMINING genmet ====# 
      
      cat ../../html/MET/METSummaryTable_Middle.html | sed -e "s/INSERT_LIST/$iGenMet/g" >> ../../html/MET/METSummaryTable.html;
      echo -n "Examining $iGenMet ................ ";
      mkdir $iGenMet;
      cd $iGenMet;

    #Define directory within histogram root file where this met collection lives
      dir=${Base}/$iGenMet 

    #run Root
      genMetplotCompare $refRoot $newRoot "$dir" $release_new $release_ref > MET-log.txt 2> MET-err.txt;
    #root -b -q .x METplotCompare.C > MET-log.txt 2> MET-err.txt;
      result=`cat MET-log.txt | grep "Final Result:" | cut -f3 -d' '`;

    # clean up the area
    if [ -f Result-button.gif ]; then rm Result-button.gif; fi;
    if [ -f Compatibility_with_Reference_Histograms.html ]; then
      rm Compatibility_with_Reference_Histograms.html; fi;
    if [ -f job_not_run ]; then rm job_not_run; fi;

    # link to the appropriate html pages
    ln -s ../../../html/MET/${iGenMet}/*.html .;

    case $result in
	"pass") ln -s ../../../images/Pass_button.gif Result-button.gif; echo "pass";;
	"fail") ln -s ../../../images/Fail_button.gif Result-button.gif;failtrue=1; echo "fail";;
	"no_data") ln -s ../../../images/NoData_button.gif Result-button.gif; echo "no data";
	    rm Compatibility_with_Reference_Histograms.html;
	    ln -s ../../../html/No_data_available.html Compatibility_with_Reference_Histograms.html;;
	"not_done") ln -s ../../../images/NotDone_button.gif Result-button.gif;
	    rm Compatibility_with_Reference_Histograms.html;
	    ln -s ../../../html/Not_done.html Compatibility_with_Reference_Histograms.html;;
	"")     echo "Root has experienced a crash prior to completion"; fail;;
	"*")    echo "unexpected result: $result"; fail;;
    esac;
    cd ../
  done;

 # Validate pfMet
  cat ../../html/MET/METSummaryTable_Middle.html | sed -e "s/INSERT_LIST/pfMet/g" >> ../../html/MET/METSummaryTable.html;
  
  echo -n "Examining $pfMet ................ ";
  mkdir pfMet;
  cd pfMet;
  
    #Define directory within histogram root file where this met collection lives                                                                            
  dir=${Base}/pfMet
  
    # run Root                                                                                                                                              
  pfMetplotCompare $refRoot $newRoot "$dir" $release_new $release_ref > MET-log.txt 2> MET-err.txt;
  result=`cat MET-log.txt | grep "Final Result:" | cut -f3 -d' '`;
  
    # clean up the area                                                                                                                                     
  if [ -f Result-button.gif ]; then rm Result-button.gif; fi;
  if [ -f Compatibility_with_Reference_Histograms.html ]; then
      rm Compatibility_with_Reference_Histograms.html; fi;
  if [ -f job_not_run ]; then rm job_not_run; fi;

   # link to the appropriate html pages                                                                                                                    
  ln -s ../../../html/MET/pfMet/*.html .;
  
  case $result in
      "pass") ln -s ../../../images/Pass_button.gif Result-button.gif; echo "pass";;
      "fail") ln -s ../../../images/Fail_button.gif Result-button.gif; failtrue=1; echo "fail";;
      "no_data") ln -s ../../../images/NoData_button.gif Result-button.gif; echo "no data";
	  rm Compatibility_with_Reference_Histograms.html;
	  ln -s ../../../html/No_data_available.html Compatibility_with_Reference_Histograms.html;;
      "not_done") ln -s ../../../images/NotDone_button.gif Result-button.gif;
	  rm Compatibility_with_Reference_Histograms.html;
	  ln -s ../../../html/Not_done.html Compatibility_with_Reference_Histograms.html;;
      "")     echo "Root has experienced a crash prior to completion"; fail;;
      "*")    echo "unexpected result: $result"; fail;;
  esac
  cd ../
  
  
  for ihtMet in $htMetList ; do

      cat ../../html/MET/METSummaryTable_Middle.html | sed -e "s/INSERT_LIST/$ihtMet/g" >> ../../html/MET/METSummaryTable.html;

      echo -n "Examining $ihtMet ................ ";
      mkdir $ihtMet;
      cd $ihtMet;
      
    #Define directory within histogram root file where this met collection lives
      dir=${Base}/${ihtMet}
      
    # run Root
      htMetplotCompare $refRoot $newRoot "$dir" $release_new $release_ref > MET-log.txt 2> MET-err.txt;
      result=`cat MET-log.txt | grep "Final Result:" | cut -f3 -d' '`;
      
    # clean up the area
      if [ -f Result-button.gif ]; then rm Result-button.gif; fi;
      if [ -f Compatibility_with_Reference_Histograms.html ]; then
	  rm Compatibility_with_Reference_Histograms.html; fi;
      if [ -f job_not_run ]; then rm job_not_run; fi;

    # link to the appropriate html pages
      ln -s ../../../html/MET/${ihtMet}/*.html .;
      
      case $result in
	  "pass") ln -s ../../../images/Pass_button.gif Result-button.gif; echo "pass";;
	  "fail") ln -s ../../../images/Fail_button.gif Result-button.gif; failtrue=1; echo "fail";;
	  "no_data") ln -s ../../../images/NoData_button.gif Result-button.gif; echo "no data";
              rm Compatibility_with_Reference_Histograms.html;
              ln -s ../../../html/No_data_available.html Compatibility_with_Reference_Histograms.html;;
	  "not_done") ln -s ../../../images/NotDone_button.gif Result-button.gif;
              rm Compatibility_with_Reference_Histograms.html;
              ln -s ../../../html/Not_done.html Compatibility_with_Reference_Histograms.html;;
	  "")     echo "Root has experienced a crash prior to completion"; fail;;
	  "*")    echo "unexpected result: $result"; fail;;
      esac
      cd ../
     
  done;



  #======Run MetWithMuons Validation========#

 
  for iMetMuons in $MetWithMuonsList; do
     
      cat ../../html/MET/METSummaryTable_Middle.html | sed -e "s/INSERT_LIST/$iMetMuons/g" >> ../../html/MET/METSummaryTable.html;

      echo -n "Examining $iMetMuons........ ";
      mkdir $iMetMuons;
      cd $iMetMuons;	
      
    #Define directory within histogram root file where MetWithMuons rechits live
      dir=$Base/${iMetMuons};
      
    # run Root
      metwithmuonsplotCompare $refRoot $newRoot "$dir" $release_new $release_ref > MET-log.txt 2> MET-err.txt;
      result=`cat MET-log.txt | grep "Final Result:" | cut -f3 -d' '`;
    # clean up the area

      if [ -f Result-button.gif ]; then rm Result-button.gif; fi;
      if [ -f Compatibility_with_Reference_Histograms.html ]; then
	  rm Compatibility_with_Reference_Histograms.html; fi;
      if [ -f job_not_run ]; then rm job_not_run; fi;

    # link to the appropriate html pages
      ln -s ../../../html/MET/${iMetMuons}/*.html .;
     
      case $result in
	  "pass") ln -s ../../../images/Pass_button.gif Result-button.gif; echo "pass";;
	  "fail") ln -s ../../../images/Fail_button.gif Result-button.gif; echo "fail";;
	  "no_data") ln -s ../../../images/NoData_button.gif Result-button.gif; echo "no data";
              rm Compatibility_with_Reference_Histograms.html;
              ln -s ../../../html/No_data_available.html Compatibility_with_Reference_Histograms.html;;
	  "not_done") ln -s ../../../images/NotDone_button.gif Result-button.gif; echo "not done";
              rm Compatibility_with_Reference_Histograms.html;
              ln -s ../../../html/Not_done.html Compatibility_with_Reference_Histograms.html;;
	  "")     echo "Root has experienced a crash prior to completion"; fail;;
	  "*")    echo "unexpected result: $result"; fail;;
      esac;
      cd ../	 
  done;  


  #======Run tcMet Validation========#

 
  for itcMet in $tcMetList; do
     
      cat ../../html/MET/METSummaryTable_Middle.html | sed -e "s/INSERT_LIST/$itcMet/g" >> ../../html/MET/METSummaryTable.html;

      echo -n "Examining $itcMet........ ";
      mkdir $itcMet;
      cd $itcMet;	
      
    #Define directory within histogram root file where MetWithMuons rechits live
      dir=$Base/${itcMet};
      
    # run Root
      tcmetplotCompare $refRoot $newRoot "$dir" $release_new $release_ref > MET-log.txt 2> MET-err.txt;
      result=`cat MET-log.txt | grep "Final Result:" | cut -f3 -d' '`;
    # clean up the area

      if [ -f Result-button.gif ]; then rm Result-button.gif; fi;
      if [ -f Compatibility_with_Reference_Histograms.html ]; then
	  rm Compatibility_with_Reference_Histograms.html; fi;
      if [ -f job_not_run ]; then rm job_not_run; fi;

    # link to the appropriate html pages
      ln -s ../../../html/MET/${itcMet}/*.html .;
     
      case $result in
	  "pass") ln -s ../../../images/Pass_button.gif Result-button.gif; echo "pass";;
	  "fail") ln -s ../../../images/Fail_button.gif Result-button.gif; echo "fail";;
	  "no_data") ln -s ../../../images/NoData_button.gif Result-button.gif; echo "no data";
              rm Compatibility_with_Reference_Histograms.html;
              ln -s ../../../html/No_data_available.html Compatibility_with_Reference_Histograms.html;;
	  "not_done") ln -s ../../../images/NotDone_button.gif Result-button.gif; echo "not done";
              rm Compatibility_with_Reference_Histograms.html;
              ln -s ../../../html/Not_done.html Compatibility_with_Reference_Histograms.html;;
	  "")     echo "Root has experienced a crash prior to completion"; fail;;
	  "*")    echo "unexpected result: $result"; fail;;
      esac;
      cd ../	 
  done;  

  cat ../../html/MET/METSummaryTable_End.html >> ../../html/MET/METSummaryTable.html;
  ln -s ../../html/MET/METSummaryTable.html .;

  if [ -f Result-button.gif ]; then rm Result-button.gif; fi;
  if [ $failtrue == "1" ]; then ln -s ../../images/Fail_button.gif Result-button.gif;
  fi;
  if [ $failtrue == "0" ] && [ $runMET -eq 1 ]; then ln -s ../../images/Pass_button.gif Result-button.gif;
  fi;
  
  else
	sed -e "s|INSERT_MET|Compatibility_with_Reference_Histograms|" ../../index.html > ../../index2.html;
      mv ../../index2.html ../../index.html;
  fi;

  cd ..;
  #========End MET Validation=============#


  Base="DQMData/Run 1/JetMET/Run summary/CaloTowers";
  #======Run CaloTower Validation========#
  cd CaloTowers;
  failtrue=0;
  if [ -f CTSummaryTable.html ]; then rm CTSummaryTable.html; fi;

  # job has not yet been run and is requested
  if [ -f job_not_run -a $runCaloTowerRecHits -eq 1 -o -f job_not_run -a $runAll -eq 1 ]; then
      ln -s ../../html/CaloTowers/CTSummaryTable.html .;	

      sed -e "s|INSERT_CALO|CTSummaryTable|" ../../index.html > ../../index2.html;
      mv ../../index2.html ../../index.html;

      for iTowerCollection in $CaloTowerList ; do
      
      mkdir ${iTowerCollection}
      cd ${iTowerCollection}
      
      echo -n "Examining CaloTowerRecHits (${iTowerCollection} threshholds) ... ";
    #Define directory within histogram root file where CaloTowers lives
      dir=${Base}/${iTowerCollection};
    # add appropriate links to the CT area for the Root script to function
      
    # run Root
      CaloTowerplotCompare $refRoot $newRoot "$dir" $release_new $release_ref > CaloTower-log.txt 2> CaloTower-err.txt;
    #root -b -q .x CaloTowerplotCompare.C > CaloTower-log.txt 2> CaloTower-err.txt;
      result=`cat CaloTower-log.txt | grep "Final Result:" | cut -f3 -d' '`;
      
    # clean up the area
      if [ -f Result-button.gif ]; then rm Result-button.gif; fi;
      if [ -f Compatibility_with_Reference_Histograms.html ]; then
	  rm Compatibility_with_Reference_Histograms.html; fi;
      if [ -f job_not_run ]; then rm job_not_run; fi;
      
    # link to the appropriate html pages
      ln -s ../../../html/CaloTowers/*.html .;
      if [ "X$result" = "Xpass" -o "X$result" = "Xfail" ]; then 
	  for subdir in `ls */ -1d`; do
	      cd $subdir;
	      ln -s ../../../../html/CaloTowers/specific/*.html .; 
	      ln -s ../../../../images/zerobin.gif .; 
	      cd ..;
	  done;
      fi;
      
      case $result in
	  "pass") ln -s ../../../images/Pass_button.gif Result-button.gif; echo "pass";;
	  "fail") ln -s ../../../images/Fail_button.gif Result-button.gif; failtrue=1; echo "fail";;
	  "no_data") ln -s ../../../images/NoData_button.gif Result-button.gif; echo "no data";
	      rm Compatibility_with_Reference_Histograms;	
              ln -s ../../../html/No_data_available.html Compatibility_with_Reference_Histograms.html;;
	  "not_done") ln -s ../../../images/NotDone_button.gif Result-button.gif;
              rm Compatibility_with_Reference_Histograms.html;
              ln -s ../../../html/Not_done.html Compatibility_with_Reference_Histograms.html;;
	  "")     echo "Root has experienced a crash prior to completion"; fail;;
	  "*")    echo "unexpected result: $result"; fail;;
      esac;
      
      cd ..;

      done;

  if [ -f Result-button.gif ]; then rm Result-button.gif; fi;
  if [ $failtrue == "1" ]; then ln -s ../../images/Fail_button.gif Result-button.gif;
  fi;
  if [ $failtrue == "0" ] && [ $runCaloTowerRecHits -eq 1 ]; then ln -s ../../images/Pass_button.gif Result-button.gif;
  fi;
  else
      sed -e "s|INSERT_CALO|Compatibility_with_Reference_Histograms|" ../../index.html > ../../index2.html;
      mv ../../index2.html ../../index.html;  
  fi;

  for iCalo in $MCaloTowerList; do

  if [ ! -d $iCalo ]; then

    mkdir $iCalo; cd $iCalo;

    ln -s ../../../images/preview.gif .;
    ln -s ../../../images/NotDone_button.gif Result-button.gif;
    ln -s ../../../html/Not_done.html Compatibility_with_Reference_Histograms.html;
    cd ..;
  fi;
  done;
 
  cd ..;
  #======End CaloTower Validation========#
  
  cd ..; echo Finished job at `date`;
  
done;

