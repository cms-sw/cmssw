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
const  int       nrFiles  	  		= 51;
const  TString   path     	  		= "dcap://maite.ac.be:/pnfs/iihe/becms/heyninck/TtSemiMuEvents_TopRex_Juni/TtSemiMuEvents_";

//matching variables
const  bool  	 useSpaceAngle    		= true;
const  double 	 SumAlphaCut  	  		= 0.7;

//select which observables to use
const  int      nrJetCombObs  			= 7;
const  int      JetCombObs[nrJetCombObs] 	= {1,2,3,4,5,6,7};
const  TString  JetCombInputFileName   		= "../data/TtSemiLRJetCombAllObs.root";

//likelihood histogram variables
const  int   	nrJetCombLRtotBins   		= 30;
const  double 	JetCombLRtotMin   		= -5;
const  double 	JetCombLRtotMax      		= 7;
const  char* 	JetCombLRtotFitFunction      	= "[0]/(1 + 1/exp([1]*([2] - x)))";

//output files ps/root
const  TString  JetCombOutFileName   		= "../data/TtSemiLRJetCombSelObsAndPurity.root";
const  TString  JetCombPSfile     		= "../data/TtSemiLRJetCombSelObsAndPurity.ps";

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
  for(int j = 0; j < nrJetCombObs; j++){
    obsNrs.push_back(JetCombObs[j]);
  }
  myLRhelper = new LRHelpFunctions(nrJetCombLRtotBins, JetCombLRtotMin, JetCombLRtotMax, JetCombLRtotFitFunction);

  // read in S/S+N fits of observables to use
  myLRhelper -> readObsHistsAndFits(JetCombInputFileName, obsNrs, false);
  
  // fill calculated LR value for each signal or background contributions
  doEventloop(); 
  
  // make Purity vs logLR and Purity vs. Efficiency plot
  myLRhelper -> makeAndFitPurityHists();       
    
  // store histograms and fits in root-file
  myLRhelper -> storeToROOTfile(JetCombOutFileName);
     
  // make some control plots and put them in a .ps file
  myLRhelper -> storeControlPlots(JetCombPSfile);
}





//
// Loop over the events (with the definition of what is considered signal and background)
//

void doEventloop(){ 
  cout<<endl<<endl<<"**** STARTING EVENT LOOP ****"<<endl;
  int totNrEv = 0;
  for (int nr = 1; nr <= nrFiles; nr++) {
   TString ft = path; 
   ft += nr-1;
   ft += ".root";
    if (!gSystem->AccessPathName(ft)) {
      TFile *file = TFile::Open(ft);
      TTree * events = dynamic_cast<TTree*>( file->Get( "Events" ) );
      assert( events != 0 );
      TBranch * solsbranch = events->GetBranch( "TtSemiEvtSolutions_solutions__TtEventReco.obj" );
      //TBranch * solsbranch = events->GetBranch( "TtSemiEvtSolutions_solutions__CommonBranchSel.obj" );
      assert( solsbranch != 0 );
      vector<TtSemiEvtSolution> sols;
      solsbranch->SetAddress( & sols );
    
      //loop over all events in a file 
      for( int ev = 0; ev < events->GetEntries(); ++ ev ) {
        ++totNrEv;
        if((double)((totNrEv*1.)/1000.) == (double) (totNrEv/1000)) cout<< "  Processing event "<< totNrEv<<endl; 
        solsbranch->GetEntry( ev );
        if(sols.size()== 12){
          // check if good matching solution exists
          bool trueSolExists = false;
          for(int s=0; s<12; s++){
            if(sols[s].getSumDeltaRjp()<SumAlphaCut) trueSolExists = true;
          }
          if(trueSolExists){
	    double maxLogLRVal = -999.;
	    int    maxLogLRSol = -999;
	    //loop over solutions
	    for(int s=0; s<12; s++){
              // get observable values
	      vector<double> obsVals;
	      for(int j = 0; j < nrJetCombObs; j++){
	        if( myLRhelper->obsFitIncluded(obsNrs[j]) ) obsVals.push_back(sols[s].getLRJetCombObsVal(obsNrs[j]));
	      }
	      double logLR =  myLRhelper -> calcLRval(obsVals);
	      if(logLR>maxLogLRVal) { maxLogLRVal = logLR; maxLogLRSol = s; };
	    }
	    if(sols[maxLogLRSol].getSumDeltaRjp()<SumAlphaCut && sols[maxLogLRSol].getMCCorrJetComb()==maxLogLRSol) {
	      myLRhelper -> fillLRSignalHist(maxLogLRVal);
	      //cout << "mxLR " << maxLogLRVal << endl;
	    }
	    else
	    {
	      myLRhelper -> fillLRBackgroundHist(maxLogLRVal);
	      //cout << "mxLR (bg) " << maxLogLRVal << endl;
	    }
          }
        }  
      }
      file->Close();
    }
    else
    {
      cout<<ft<<" doesn't exist"<<endl;
    }
  }
}
