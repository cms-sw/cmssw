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
#include "FWCore/FWLite/src/AutoLibraryLoader.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "TopQuarkAnalysis/TopTools/interface/LRHelpFunctions.h"

using namespace std;



///////////////////////
// Constants         //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//input files
const  int       signal_nrFiles  	  	= 1;
const  TString   signal_path     	  	= "/beo5/ghammad/CMSSW/src/TopQuarkAnalysis/TopEventProducers/test/RootFiles/TopRex_TtSemiMuEventsSignal_60pb_LooseCuts_";
const  int       bckgd_nrFiles                  = 1;
const  TString   bckgd_path     	  	= "/beo5/ghammad/CMSSW/src/TopQuarkAnalysis/TopEventProducers/test/RootFiles/TopRex_TtSemiMuEventsBckgd_60pb_LooseCuts_";

//observable histogram variables
const  int      nrSignalSelObs  		= 18;
const  int      SignalSelObs[nrSignalSelObs] 	= {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
const  int   	nrSignalSelHistBins    		= 50;
const  double   SignalSelObsMin[nrSignalSelObs]	= {0,0,0,0,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0};
const  double   SignalSelObsMax[nrSignalSelObs]	= {250,3,1,800,1500,1,7,8,250,0.7,300,1,1,1,1,1,60,1};

//observable fit functions
const char*     SignalSelObsFits[nrSignalSelObs]= {           
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs1	
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs2
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs3
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs4
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs5
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs6
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs7
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs8
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs9
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs10
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs11
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs12
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs13
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs14
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs15
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs16
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs17
						     "[0]/(1 + 1/exp([1]*([2] - x)))"  //obs18
                                          	  };

//output files ps/root
const  TString  SignalSelOutfileName   		= "../data/TtSemiLRSignalSelAllObs.root";
const  TString  SignalSelPSfile     		= "../data/TtSemiLRSignalSelAllObs.ps";

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//
// Global variables
//
LRHelpFunctions *myLRhelper;
void doEventloop();
vector<int> obsNrs;
vector<double> obsMin,obsMax;
vector<const char*> obsFits;




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
    obsMin.push_back(SignalSelObsMin[j]);
    obsMax.push_back(SignalSelObsMax[j]);
    obsFits.push_back(SignalSelObsFits[j]);
  }
  myLRhelper = new LRHelpFunctions(obsNrs, nrSignalSelHistBins, obsMin, obsMax, obsFits);  

  // fill signal and background contributions to S and B histograms
  doEventloop(); 
  
  // normalize the S and B histograms to construct the pdf's
  myLRhelper -> normalizeSandBhists();
  
  // produce and fit the S/S+N histograms
  myLRhelper -> makeAndFitSoverSplusBHists();
   
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
    TString signal_ft = signal_path; signal_ft += nr-1; signal_ft += ".root";
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
	  // Fill the observables 
	  // signal: semileptonic top event 
	  myLRhelper -> fillToSignalHists(signal_obsVals);
	  ++okEvents;
        }
      }
      signal_file->Close();
    }
    else
    {
      cout<<signal_ft<<" doesn't exist"<<endl;
    }
  }
  cout<<endl<<"********************  STATISTICS FOR SIGNAL ***********************"<<endl;
  cout<<endl<<" Nb of processed events  :"<<(totNrEv)<<endl;
  cout<<endl<<" Nb of events filled in the histo :"<<(okEvents)<<endl;
  cout<<endl<<"******************************************************************"<<endl;


  cout<<endl<<endl<<"**** STARTING EVENT LOOP ****"<<endl;


  /********************************************** for the background **********************************************/

  okEvents = 0, totNrEv = 0;
  for (int nr = 1; nr <= bckgd_nrFiles; nr++) {
    TString bckgd_ft = bckgd_path; bckgd_ft += nr-1; bckgd_ft += ".root";
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
	  // Fill the observables 
	  // bckgd: semileptonic top event 
	  myLRhelper -> fillToBackgroundHists(bckgd_obsVals);
	  ++okEvents;
        }
      }
      bckgd_file->Close();
    }
    else
    {
      cout<<bckgd_ft<<" doesn't exist"<<endl;
    }
  }
    cout<<endl<<"********************  STATISTICS FOR BCKGD ***********************"<<endl;
    cout<<endl<<" Nb of processed events  :"<<(totNrEv)<<endl;
    cout<<endl<<" Nb of events filled in the histo :"<<(okEvents)<<endl;   
    cout<<endl<<"******************************************************************"<<endl;
}
