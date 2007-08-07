#include <iostream>
#include <cassert>
#include <TROOT.h>
#include <TSystem.h>
#include <Cintex/Cintex.h>
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

using namespace std;



///////////////////////
// Constants         //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//input files
const  int       signal_nrFiles                 = 51;
const  TString   signal_path                    = "dcap://maite.iihe.ac.be:/pnfs/iihe/becms/heyninck/TtSemiMuEvents_TopRex_Juni/TtSemiMuEvents_";
const  int       bckgd_nrFiles                  = 51;
const  TString   bckgd_path                     = "dcap://maite.iihe.ac.be:/pnfs/iihe/becms/heyninck/TtOtherTtEvents_TopRex_Juni/TtOtherTtEvents_";

//observable histogram variables
const  int      nrSignalSelObs  		= 3;
const  int      SignalSelObs[nrSignalSelObs] 	= {1,9,11};
const  TString  SignalSelInputFileName   	= "../data/TtSemiLRSignalSelAllObs.root";

//likelihood histogram variables
const  int   	nrSignalSelLRtotBins   		= 40;
const  double 	SignalSelLRtotMin   		= -5;
const  double 	SignalSelLRtotMax      		= 5;
const  char* 	SignalSelLRtotFitFunction      	= "[0]/(1 + 1/exp([1]*([2] - x)))";

//output files ps/root
const  TString  SignalSelOutfileName   		= "../data/TtSemiLRSignalSelSelObsAndPurity.root";
const  TString  SignalSelPSfile     		= "../data/TtSemiLRSignalSelSelObsAndPurity.ps";

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//
// Global variables
//
LRHelpFunctions *myLRhelper;
void doEventloop();
vector<int> obsNrs;




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
  cout<<endl<<endl<<"**** STARTING EVENT LOOP FOR SIGNAL ****"<<endl;

  /********************************************** for the signal **********************************************/

  int okEvents = 0, totNrEv = 0;
  for (int nr = 1; nr <= signal_nrFiles; nr++) {
    TString signal_ft = signal_path; 
    signal_ft += nr-1;
    signal_ft += ".root";
    if (!gSystem->AccessPathName(signal_ft)) {
      TFile *signal_file = TFile::Open(signal_ft);
      TTree *signal_events = dynamic_cast<TTree*>( signal_file->Get( "Events" ) );
      assert( signal_events != 0 );
      TBranch * signal_solsbranch  = signal_events->GetBranch( "TtSemiEvtSolutions_solutions__TtEventReco.obj" );
      assert(   signal_solsbranch != 0 );
      vector<TtSemiEvtSolution> signal_sols;
      signal_solsbranch->SetAddress( & signal_sols );

      //loop over all events in a file 
      for( int ev = 0; ev < signal_events->GetEntries(); ++ ev ) {
        ++totNrEv;
        if((double)((totNrEv*1.)/1000.) == (double) (totNrEv/1000)) cout<< "  Processing signal event "<< totNrEv<<endl; 
        signal_solsbranch->GetEntry( ev );
        if(signal_sols.size()== 12){
          // get observable values
	  vector<double> signal_obsVals;
	  for(int j = 0; j < nrSignalSelObs; j++){
	    if( myLRhelper->obsFitIncluded(obsNrs[j]) ) signal_obsVals.push_back(signal_sols[0].getLRSignalEvtObsVal(obsNrs[j]));
	  }
	  double logLR =  myLRhelper -> calcLRval(signal_obsVals);
	  myLRhelper -> fillLRSignalHist(logLR);
        }
      }
      signal_file->Close();
    }
    else
    {
      cout<<signal_ft<<" doesn't exist"<<endl;
    }
  }


  cout<<endl<<endl<<"**** STARTING EVENT LOOP FOR BCKGD ****"<<endl;


  /********************************************** for the background **********************************************/

  okEvents = 0, totNrEv = 0;
  for (int nr = 1; nr <= bckgd_nrFiles; nr++) {
    TString bckgd_ft = bckgd_path; 
    bckgd_ft += nr-1; 
    bckgd_ft += ".root";
    if (!gSystem->AccessPathName(bckgd_ft)) {
      TFile *bckgd_file = TFile::Open(bckgd_ft);
      TTree *bckgd_events = dynamic_cast<TTree*>( bckgd_file->Get( "Events" ) );
      assert( bckgd_events != 0 );
      TBranch * bckgd_solsbranch  = bckgd_events->GetBranch( "TtSemiEvtSolutions_solutions__TtEventReco.obj" );
      assert(   bckgd_solsbranch != 0 );
      vector<TtSemiEvtSolution> bckgd_sols;
      bckgd_solsbranch->SetAddress( & bckgd_sols );
    
      //loop over all events in a file 
      for( int ev = 0; ev < bckgd_events->GetEntries(); ++ ev ) {
        ++totNrEv;
        if((double)((totNrEv*1.)/1000.) == (double) (totNrEv/1000)) cout<< "  Processing bckgd event "<< totNrEv<<endl; 
        bckgd_solsbranch->GetEntry( ev );
        if(bckgd_sols.size()== 12){
          // get observable values
	  vector<double> bckgd_obsVals;
	  for(int j = 0; j < nrSignalSelObs; j++){
	    if( myLRhelper->obsFitIncluded(obsNrs[j]) ) bckgd_obsVals.push_back(bckgd_sols[0].getLRSignalEvtObsVal(obsNrs[j]));
	  }
	  double logLR =  myLRhelper -> calcLRval(bckgd_obsVals);
	  myLRhelper -> fillLRBackgroundHist(logLR);
        }
      }
      bckgd_file->Close();
    }
    else
    {
      cout<<bckgd_ft<<" doesn't exist"<<endl;
    }
  }
}
