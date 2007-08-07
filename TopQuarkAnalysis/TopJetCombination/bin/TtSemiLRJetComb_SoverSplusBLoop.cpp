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
const  int       nrFiles  	  		= 51;
const  TString   path     	  		= "dcap://maite.iihe.ac.be:/pnfs/iihe/becms/heyninck/TtSemiMuEvents_TopRex_Juni/TtSemiMuEvents_";

//matching variables
const  bool  	 useSpaceAngle    		= true;
const  double 	 SumAlphaCut  	  		= 0.7;


//observable histogram variables (include all defined observables!!!)
const  int      nrJetCombObs  			= 7;
const  int      JetCombObs[nrJetCombObs] 	= {1,2,3,4,5,6,7};
const  int   	nrJetCombHistBins    		= 50;
const  double   JetCombObsMin[nrJetCombObs]	= {0,0,0,0,0,-15,-6};
const  double   JetCombObsMax[nrJetCombObs]	= {3,3.5,5,5,5,70,0};


//observable fit functions
const char*     JetCombObsFits[nrJetCombObs] 	= {  "[0]/(1 + 1/exp([1]*([2] - x)))",  //obs1	
						     "[0]/(1 + 1/exp([1]*([2] - x)))",  //obs2	
						     "gaus",  //obs3
						     "gaus", //obs4
						     "gaus", //obs5
						     "([0]+[3]*abs(x)/x)*(1-exp([1]*(abs(x)-[2])))",  //obs6	
						     "[0]/(1 + 1/exp([1]*([2] - x)))"  //obs7
						  };

//output files ps/root
const  TString  JetCombOutfileName   		= "../data/TtSemiLRJetCombAllObs.root";
const  TString  JetCombPSfile     		= "../data/TtSemiLRJetCombAllObs.ps";

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
  for(int j = 0; j < nrJetCombObs; j++){
    obsNrs.push_back(JetCombObs[j]);
    obsMin.push_back(JetCombObsMin[j]);
    obsMax.push_back(JetCombObsMax[j]);
    obsFits.push_back(JetCombObsFits[j]);
  }
  myLRhelper = new LRHelpFunctions(obsNrs, nrJetCombHistBins, obsMin, obsMax, obsFits); 
  
  vector<double> parsFobs6; 
  parsFobs6.push_back(0.8);
  parsFobs6.push_back(-0.1);
  parsFobs6.push_back(-0.8);
  parsFobs6.push_back(0.2);
  myLRhelper -> setObsFitParameters(6,parsFobs6);


  // fill signal and background contributions to S and B histograms
  doEventloop(); 
  
  // normalize the S and B histograms to construct the pdf's
  myLRhelper -> normalizeSandBhists();
  
  // produce and fit the S/S+N histograms
  myLRhelper -> makeAndFitSoverSplusBHists();
  
  // store histograms and fits in root-file
  myLRhelper -> storeToROOTfile(JetCombOutfileName);
     
  // make some control plots and put them in a .ps file
  myLRhelper -> storeControlPlots(JetCombPSfile);
}





//
// Loop over the events (with the definition of what is considered signal and background)
//

void doEventloop(){ 
  cout<<endl<<endl<<"**** STARTING EVENT LOOP ****"<<endl;
  int okEvents = 0, totNrEv = 0;
  for (int nr = 1; nr <= nrFiles; nr++) {
    TString ft = path; 
    ft += nr-1; 
    ft += ".root";
    if (!gSystem->AccessPathName(ft)) {
      TFile *file = TFile::Open(ft);
      TTree * events = dynamic_cast<TTree*>( file->Get( "Events" ) );
      assert( events != 0 );
      TBranch * solsbranch = events->GetBranch( "TtSemiEvtSolutions_solutions__TtEventReco.obj" );
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
            if(sols[s].getMCBestSumAngles()<SumAlphaCut) trueSolExists = true;
          }
          if(trueSolExists){
  	    for(int s=0; s<12; s++){
              // get observable values
	      vector<double> obsVals;
	      for(int j = 0; j < nrJetCombObs; j++){
	        if( myLRhelper->obsFitIncluded((unsigned int)obsNrs[j]) ) obsVals.push_back(sols[s].getLRJetCombObsVal((unsigned int)obsNrs[j]));
	      }
	      // Fill the observables for each jet combination if a good matching exists
	      // signal: best matching solution
              // background: all other solutions 
	      if(sols[s].getMCCorrJetComb()==s) {
	        myLRhelper -> fillToSignalHists(obsVals);
	        ++okEvents;
	      }
	      else
	      {
	        myLRhelper -> fillToBackgroundHists(obsVals);
       	      }
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
  cout<<endl<<"***********************  STATISTICS  *************************"<<endl;
  cout<<" Probability that a correct jet combination exists:"<<endl;
  cout<<" (fraction events with ";
  if(useSpaceAngle) cout<<"min SumAngle_jp < ";
  if(!useSpaceAngle) cout<<"min DR_jp < ";
  cout<<SumAlphaCut<<" )"<<endl;
  cout<<endl<<"                 "<<(100.*okEvents)/(1.*totNrEv)<<" %"<<endl;
  cout<<endl<<"******************************************************************"<<endl;
}
