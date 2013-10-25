#include <iostream>
#include <cassert>
#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TGraph.h>
#include <TF1.h>
#include <TFormula.h>
#include <TStyle.h>
#include <TKey.h>
#include <vector>
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "TopQuarkAnalysis/TopTools/interface/LRHelpFunctions.h"

///////////////////////
// Constants         //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//input files
const  int       signal_nrDir			= 5;
const  int       signal_nrFiles[signal_nrDir]   = {45,40,30,15,17};
const  TString   signal_path[signal_nrDir]      = {
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt0j/Alpgen_tt0j_TtSemiMuEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt1j/Alpgen_tt1j_TtSemiMuEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt2j/Alpgen_tt2j_TtSemiMuEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt3j/Alpgen_tt3j_TtSemiMuEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt4j/Alpgen_tt4j_TtSemiMuEvents_"
						  };
const  int       signal_NrEv[signal_nrDir]	= {76250,68000,40000,16000,24425};


const  int       bckgd_nrDir			= 8;
const  int       bckgd_nrFiles[bckgd_nrDir]     = {45,40,30,15,17,10,6,2};//{9,4,2};//{35,30,20,10,10};
const  TString   bckgd_path[bckgd_nrDir]        = {
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt0j/Alpgen_tt0j_TtOtherEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt1j/Alpgen_tt1j_TtOtherEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt2j/Alpgen_tt2j_TtOtherEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt3j/Alpgen_tt3j_TtOtherEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt4j/Alpgen_tt4j_TtOtherEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/W4j/Alpgen_W4j_TtSemiMuEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/W5j/Alpgen_W5j_TtSemiMuEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/W6j/Alpgen_W6j_TtSemiMuEvents_"
						   };
						   
const  int       bckgd_NrEv[bckgd_nrDir]	= {76250,68000,40000,16000,24425,69700,18000,12500};//{69700,18000,12500};//{152500,136000,80000,32000,49000};

//observable histogram variables
const  int      nrSignalSelObs  		= 7;
const  int      SignalSelObs[nrSignalSelObs] 	= {1,3,6,12,15,16,31};
const  TString  SignalSelInputFileName   	= "./TtSemiLRSignalSelAllObs.root";

//likelihood histogram variables
const  int   	nrSignalSelLRtotBins   		= 35;
const  double 	SignalSelLRtotMin   		= -10;
const  double 	SignalSelLRtotMax      		= 8;
const  char* 	SignalSelLRtotFitFunction      	= "[0]/(1 + 1/exp([1]*([2] - x)))+[3]";

//output files ps/root
const  TString  SignalSelOutfileName   		= "./TtSemiLRSignalSelSelObsAndPurity.root";
const  TString  SignalSelPSfile     		= "./TtSemiLRSignalSelSelObsAndPurity.ps";

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//
// Global variables
//
LRHelpFunctions *myLRhelper;
void doEventloop();
std::vector<int> obsNrs;

//
// Main analysis
//

int main() { 
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  
  
  // define all histograms & fit functions
  //to replace with something more elegant
  for(int j = 0; j < nrSignalSelObs; j++){
    obsNrs.push_back(SignalSelObs[j]);
  }
  myLRhelper = new LRHelpFunctions(nrSignalSelLRtotBins, SignalSelLRtotMin, SignalSelLRtotMax, SignalSelLRtotFitFunction);

  // read in S/S+N fits of observables to use
  myLRhelper -> readObsHistsAndFits(SignalSelInputFileName, obsNrs, false);

  
  // fill calculated LR value for each signal or background contributions
  doEventloop(); 
  
  // make Purity vs logLR and Purity vs. Efficiency plot
  myLRhelper -> makeAndFitPurityHists();       
    
  // store histograms and fits in root-file
  myLRhelper -> storeToROOTfile(SignalSelOutfileName);
     
  // make some control plots and put them in a .ps file
  myLRhelper -> storeControlPlots(SignalSelPSfile);
}





//
// Loop over the events (with the definition of what is considered signal and background)
//

void doEventloop(){ 
  std::cout<<std::endl<<std::endl<<"**** STARTING EVENT LOOP FOR SIGNAL ****"<<std::endl;

  /********************************************** for the signal **********************************************/

  int okEvents = 0, totNrEv = 0;
for (int nrDir =0; nrDir < signal_nrDir; nrDir++){

  std::cout<< " Signal : "<<signal_path[nrDir]<<std::endl;

  int Signal_totNrEv =0, Signal_okEvents =0;
  for (int nr = 1; nr <= signal_nrFiles[nrDir]; nr++) {
    TString signal_ft = signal_path[nrDir]; 
    signal_ft += nr;
    signal_ft += ".root";
    if (!gSystem->AccessPathName(signal_ft)) {
      TFile *signal_file = TFile::Open(signal_ft);
      TTree *signal_events = dynamic_cast<TTree*>( signal_file->Get( "Events" ) );
      assert( signal_events != 0 );

      TBranch * signal_solsbranch  = signal_events->GetBranch( "TtSemiEvtSolutions_solutions__TEST.obj" );
      assert(   signal_solsbranch != 0 );
      std::vector<TtSemiEvtSolution> signal_sols;
      signal_solsbranch->SetAddress( & signal_sols );

      //loop over all events in a file 
      for( int ev = 0; ev < signal_events->GetEntries(); ++ ev ) {
        if(Signal_totNrEv>signal_NrEv[nrDir] && signal_NrEv[nrDir] != -1) continue;
	++Signal_totNrEv;
        ++totNrEv;
        if((double)((totNrEv*1.)/5000.) == (double) (totNrEv/5000)) std::cout<< "  Processing signal event "<< totNrEv<<std::endl; 
        signal_solsbranch->GetEntry( ev );
        if(signal_sols.size()== 12){
          // get observable values
	  std::vector<double> signal_obsVals;
	  for(int j = 0; j < nrSignalSelObs; j++){
	    if( myLRhelper->obsFitIncluded(obsNrs[j]) ) signal_obsVals.push_back(signal_sols[0].getLRSignalEvtObsVal(obsNrs[j]));
	  }

	  double logLR =  myLRhelper -> calcLRval(signal_obsVals);
	  myLRhelper -> fillLRSignalHist(logLR);
	  ++Signal_okEvents;
	  ++okEvents;
        }
      }
      signal_file->Close();
    }
    else
    {
      std::cout<<signal_ft<<" doesn't exist"<<std::endl;
    }
  }
  std::cout<<std::endl<<"********************  STATISTICS FOR SIGNAL "<<signal_path[nrDir]<<" ***********************"<<std::endl;
  std::cout<<std::endl<<" Nb of processed events  :"<<(Signal_totNrEv)<<std::endl;
  std::cout<<std::endl<<" Nb of events filled in the histo :"<<(Signal_okEvents)<<std::endl;
  std::cout<<std::endl<<"******************************************************************"<<std::endl;
 }
  std::cout<<std::endl<<"********************  STATISTICS FOR SIGNAL ***********************"<<std::endl;
  std::cout<<std::endl<<" Nb of processed events  :"<<(totNrEv)<<std::endl;
  std::cout<<std::endl<<" Nb of events filled in the histo :"<<(okEvents)<<std::endl;
  std::cout<<std::endl<<"******************************************************************"<<std::endl;

  std::cout<<std::endl<<std::endl<<"**** STARTING EVENT LOOP FOR BCKGD ****"<<std::endl;


  /********************************************** for the background **********************************************/

  okEvents = 0, totNrEv = 0;
for (int nrDir =0; nrDir < bckgd_nrDir; nrDir++){

  std::cout<< " Background : "<<bckgd_path[nrDir]<<std::endl;

  int Bckgd_totNrEv =0, Bckgd_okEvents =0;
  for (int nr = 1; nr <= bckgd_nrFiles[nrDir]; nr++) {
    TString bckgd_ft = bckgd_path[nrDir]; 
    bckgd_ft += nr; 
    bckgd_ft += ".root";
    if (!gSystem->AccessPathName(bckgd_ft)) {
      TFile *bckgd_file = TFile::Open(bckgd_ft);
      TTree *bckgd_events = dynamic_cast<TTree*>( bckgd_file->Get( "Events" ) );
      assert( bckgd_events != 0 );

      TBranch * bckgd_solsbranch  = bckgd_events->GetBranch( "TtSemiEvtSolutions_solutions__TEST.obj" );
      assert(   bckgd_solsbranch != 0 );
      std::vector<TtSemiEvtSolution> bckgd_sols;
      bckgd_solsbranch->SetAddress( & bckgd_sols );
    
      //loop over all events in a file 
      for( int ev = 0; ev < bckgd_events->GetEntries(); ++ ev ) {
        if(Bckgd_totNrEv>bckgd_NrEv[nrDir]) continue;
	++Bckgd_totNrEv;
        ++totNrEv;
        if((double)((totNrEv*1.)/1000.) == (double) (totNrEv/1000)) std::cout<< "  Processing bckgd event "<< totNrEv<<std::endl; 
        bckgd_solsbranch->GetEntry( ev );
        if(bckgd_sols.size()== 12){
          // get observable values
	  std::vector<double> bckgd_obsVals;
	  for(int j = 0; j < nrSignalSelObs; j++){
	    if( myLRhelper->obsFitIncluded(obsNrs[j]) ) bckgd_obsVals.push_back(bckgd_sols[0].getLRSignalEvtObsVal(obsNrs[j]));
	  }

	  double logLR =  myLRhelper -> calcLRval(bckgd_obsVals);
	  myLRhelper -> fillLRBackgroundHist(logLR);
	  ++okEvents;
	  ++Bckgd_okEvents;

        }
      }
      bckgd_file->Close();
    }
    else
    {
      std::cout<<bckgd_ft<<" doesn't exist"<<std::endl;
    }
  }
    std::cout<<std::endl<<"********************  STATISTICS FOR BCKGD "<<bckgd_path[nrDir]<<" ***********************"<<std::endl;
    std::cout<<std::endl<<" Nb of processed events  :"<<(Bckgd_totNrEv)<<std::endl;
    std::cout<<std::endl<<" Nb of events filled in the histo :"<<(Bckgd_okEvents)<<std::endl;   
    std::cout<<std::endl<<"******************************************************************"<<std::endl;
 }
    std::cout<<std::endl<<"********************  STATISTICS FOR BCKGD ***********************"<<std::endl;
    std::cout<<std::endl<<" Nb of processed events  :"<<(totNrEv)<<std::endl;
    std::cout<<std::endl<<" Nb of events filled in the histo :"<<(okEvents)<<std::endl;   
    std::cout<<std::endl<<"******************************************************************"<<std::endl;
}
