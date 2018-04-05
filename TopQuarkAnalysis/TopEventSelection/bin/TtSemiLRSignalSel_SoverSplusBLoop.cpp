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
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "TopQuarkAnalysis/TopTools/interface/LRHelpFunctions.h"

///////////////////////
// Constants         //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//input files
const  int       signal_nrDir			= 5;
const  int       signal_nrFiles[signal_nrDir]   = {30,25,20,15,10};
const  TString   signal_path[signal_nrDir]      = {
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt0j/Alpgen_tt0j_TtSemiMuEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt1j/Alpgen_tt1j_TtSemiMuEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt2j/Alpgen_tt2j_TtSemiMuEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt3j/Alpgen_tt3j_TtSemiMuEvents_",
						   "dcap:///pnfs/iihe/becms/ghammad/TQAF136Final/tt4j/Alpgen_tt4j_TtSemiMuEvents_"
						  };
const  int       signal_NrEv[signal_nrDir]	= {76250,68000,40000,16000,24425}; //nb of events you want to process


const  int       bckgd_nrDir			= 8;
const  int       bckgd_nrFiles[bckgd_nrDir]     = {30,25,20,15,10,10,10,5};
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
						   
const  int       bckgd_NrEv[bckgd_nrDir]	= {76250,68000,40000,16000,24425,69700,18000,12500}; //nb of events you want to process

//observable histogram variables
const  int      nrSignalSelObs  		= 38;
const  int      SignalSelObs[nrSignalSelObs] 	= {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38};
const  int   	nrSignalSelHistBins    		= 35;
const  double   SignalSelObsMin[nrSignalSelObs]	= {10,0,0,0,0,0.5,0,-5,-2,-10,0,0,0,0.15,0,0,0,0,0,0,0.1,0,0,0,0,0,0,0,0,0,0.55,0,0,0,0,0,0,0};
const  double   SignalSelObsMax[nrSignalSelObs]	= {200,200,1000,300,0.5,1,1600,5,20,15,50,1,1,1,1,0.3,0.9,0.3,0.35,1000,500,0.8,70,2,1,1,0.8,1,0.65,0.75,1,0.7,0.65,0.6,1,0.35,1,0.4};

//observable fit functions

TFormula gauss("gauss", "gaus");
TFormula symgauss("symgauss", "[0]*(exp(-0.5*(x/[1])**2))");
TFormula dblgauss("dblgauss", "[0]*(exp(-0.5*((x-[1])/[2])**2)+exp(-0.5*((x+[3])/[4])**2))");
TFormula symdblgauss("symdblgauss", "[0]*(exp(-0.5*((x-[1])/[2])**2)+exp(-0.5*((x+[1])/[2])**2))");
TFormula sigm("sigm", "[0]/(1 + 1/exp([1]*([2] - x)))");
TFormula sigmc("sigmc", "[0]/(1 + 1/exp([1]*([2] - x)))+[3]");
TFormula dblsigm("dblsigm", "[0]/(1 + 1/exp([1]**2*([2] - x)))/(1 + 1/exp([3]**2*(x - [4])))");
TFormula symdblsigm("symdblsigm", "[0]/(1 + 1/exp([1]**2*([2] - x)))/(1 + 1/exp([1]**2*([2] + x)))"); 

const char*     SignalSelObsFits[nrSignalSelObs]= {           
						   
						     "[0]/(1 + 1/exp([1]*([2] - x)))",//"[0]*(1-exp(-[1]*x))-[2]*(1-exp(-[3]*x))",//"[0]*exp(-pow((x-[1])/[2],2))+[3]*(1-exp(-[4]*x))", //obs1
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]",//"[0]/(1 + 1/exp([1]*([2] - x)))*(exp(-[3]*x))", //obs2
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]*pol4", //obs3
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]", //obs4
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs5
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]+pol3(4)+pol5(8)", //obs6
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]*pol4", //obs7
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]*pol3+[4]", //obs8
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]*pol3+[4]", //obs9
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]", //obs10
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]", //obs11
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]", //obs12
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]", //obs13
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs14
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]*pol3", //obs15
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]*pol4", //obs16
						     "pol9",//"[0]/(1 + 1/exp([1]*([2] - x)))+[3]*pol4", //obs17
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]*pol4", //obs18
						     "pol6",//"[0]/(1 + 1/exp([1]*([2] - x)))+[3]*pol4", //obs19
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]*exp([4]*([5] - x))", //obs20
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]*pol4", //obs21
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs22
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs23
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs24
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs25
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs26
						     "pol6",//"[0]/(1 + 1/exp([1]*([2] - x)))", //obs27
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs28
						     "[0]/(1 + 1/exp([1]*([2] - x)))+[3]*exp([4]*([5] - x))", //obs29
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs30
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs31
						     "pol6",//"[0]/(1 + 1/exp([1]*([2] - x)))", //obs32
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs33
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs34
						     "pol3+[3]*exp([4]*([5] - x))",//"[0]/(1 + 1/exp([1]*([2] - x)))", //obs35
						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs36
						     "pol3+[3]*exp([4]*([5] - x))",//"[0]/(1 + 1/exp([1]*([2] - x)))", //obs37
						     "pol5+[3]*exp([4]*([5] - x))",//"[0]/(1 + 1/exp([1]*([2] - x)))", //obs38
//						     "pol3+[3]*exp([4]*([5] - x))",    //obs39
//						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs40
//						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs41
//						     "[0]/(1 + 1/exp([1]*([2] - x)))", //obs42
//						     "[0]/(1 + 1/exp([1]*([2] - x)))"  //obs43
                                          	  };

//output files ps/root
const  TString  SignalSelOutfileName   		= "./TtSemiLRSignalSelAllObs.root";
const  TString  SignalSelPSfile     		= "./TtSemiLRSignalSelAllObs.ps";

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//
// Global variables
//
LRHelpFunctions *myLRhelper;
void doEventloop();
std::vector<int> obsNrs;
std::vector<double> obsMin,obsMax;
std::vector<const char*> obsFits;

bool MuonIso = true;

//
// Main analysis
//

int main() { 
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable();
  
  
  // define all histograms & fit functions
  //to replace with something more elegant
  for(int j = 0; j < nrSignalSelObs; j++){
    obsNrs.push_back(SignalSelObs[j]);
    obsMin.push_back(SignalSelObsMin[j]);
    obsMax.push_back(SignalSelObsMax[j]);
    obsFits.push_back(SignalSelObsFits[j]);
  }
  myLRhelper = new LRHelpFunctions(obsNrs, nrSignalSelHistBins, obsMin, obsMax, obsFits);  
  
  // manually set some initial values for fit function parameters
  std::vector<double> parsFobs1; parsFobs1.push_back(20); parsFobs1.push_back(0.04); parsFobs1.push_back(21); parsFobs1.push_back(0.04);
  myLRhelper -> setObsFitParameters(1,parsFobs1);

  std::vector<double> parsFobs2; parsFobs2.push_back(0.495); parsFobs2.push_back(-0.148); parsFobs2.push_back(60.33); parsFobs2.push_back(0.396); //parsFobs2.push_back(0.03);
  myLRhelper -> setObsFitParameters(2,parsFobs2);

  std::vector<double> parsFobs3; parsFobs3.push_back(7.60418e-01); parsFobs3.push_back(-3.31635e-02); parsFobs3.push_back(1.57387e+02); parsFobs3.push_back(-1.23931e-08); parsFobs3.push_back(-1.90918e-04);
  myLRhelper -> setObsFitParameters(3,parsFobs3);

  std::vector<double> parsFobs4; parsFobs4.push_back(1.087); parsFobs4.push_back(-0.1978); parsFobs4.push_back(22.803); parsFobs4.push_back(-0.126); //parsFobs4.push_back(0.04);
  myLRhelper -> setObsFitParameters(4,parsFobs4);

  std::vector<double> parsFobs7; parsFobs7.push_back(0.606878); parsFobs7.push_back(-1.52796e-02); parsFobs7.push_back(2.50574e+02); parsFobs7.push_back(-4.46936e-10); parsFobs7.push_back(-1.48804e-04);
  myLRhelper -> setObsFitParameters(7,parsFobs7);

  std::vector<double> parsFobs8; parsFobs8.push_back(3.30611e-01); parsFobs8.push_back(-8.34406e+00); parsFobs8.push_back(1.04307e+00); parsFobs8.push_back(-1.75190e-03); parsFobs8.push_back(5.66972e-01);
  myLRhelper -> setObsFitParameters(8,parsFobs8);

  std::vector<double> parsFobs9; parsFobs9.push_back(6.37793e-01); parsFobs9.push_back(-1.71768e+00); parsFobs9.push_back(1.88952e+00); parsFobs9.push_back(-1.03833e-03); parsFobs9.push_back(3.30284e-01);
  myLRhelper -> setObsFitParameters(9,parsFobs9);

  std::vector<double> parsFobs10; parsFobs10.push_back(0.618); parsFobs10.push_back(-1.579); parsFobs10.push_back(-0.10812); parsFobs10.push_back(0.342);
  myLRhelper -> setObsFitParameters(10,parsFobs10);

  std::vector<double> parsFobs11; parsFobs11.push_back(0.7624); parsFobs11.push_back(-0.64975); parsFobs11.push_back(3.1225); parsFobs11.push_back(0.218675);
  myLRhelper -> setObsFitParameters(11,parsFobs11);

  std::vector<double> parsFobs12; parsFobs12.push_back(1.57736e-01); parsFobs12.push_back(-2.01467e+01); parsFobs12.push_back(5.97867e-01); parsFobs12.push_back(3.81101e-01);
  myLRhelper -> setObsFitParameters(12,parsFobs12);

  std::vector<double> parsFobs13; parsFobs13.push_back(1.57736e-01); parsFobs13.push_back(-2.01467e+01); parsFobs13.push_back(5.97867e-01); parsFobs13.push_back(3.81101e-01);
  myLRhelper -> setObsFitParameters(13,parsFobs13);

  std::vector<double> parsFobs15; parsFobs15.push_back(0.6672); parsFobs15.push_back(-9.3022); parsFobs15.push_back(0.03384); parsFobs15.push_back(0.00014967); parsFobs15.push_back(-4315.96);
  myLRhelper -> setObsFitParameters(15,parsFobs15);

  std::vector<double> parsFobs16; parsFobs16.push_back(0.56855); parsFobs16.push_back(-165.768); parsFobs16.push_back(0.0021543); parsFobs16.push_back(0.0148839); parsFobs16.push_back(4391.8);
  myLRhelper -> setObsFitParameters(16,parsFobs16);

  std::vector<double> parsFobs17; parsFobs17.push_back(0.45862); parsFobs17.push_back(-42.3119); parsFobs17.push_back(0.0024431); parsFobs17.push_back(-0.0082168); parsFobs17.push_back(-41.3239);
  myLRhelper -> setObsFitParameters(17,parsFobs17);

  std::vector<double> parsFobs18; parsFobs18.push_back(0.57713); parsFobs18.push_back(-88.4547); parsFobs18.push_back(-0.0079014); parsFobs18.push_back(-0.025394); parsFobs18.push_back(4512.33);
  myLRhelper -> setObsFitParameters(18,parsFobs18);

  std::vector<double> parsFobs33; parsFobs33.push_back(5.99882e-01); parsFobs33.push_back(-1.33575e+01); parsFobs33.push_back(1.24161e-01);
  myLRhelper -> setObsFitParameters(33,parsFobs33);

  std::vector<double> parsFobs35; parsFobs35.push_back(2.49026e-01); parsFobs35.push_back(1.08819e+00); parsFobs35.push_back(-7.26373e-01); parsFobs35.push_back(1.26367e-07); parsFobs35.push_back(5.51754e+02); parsFobs35.push_back(3.94562e-02);
  myLRhelper -> setObsFitParameters(35,parsFobs35);

  std::vector<double> parsFobs37; parsFobs37.push_back(1.43676e-01); parsFobs37.push_back(2.44475e+00); parsFobs37.push_back(-4.56374e+00); parsFobs37.push_back(3.01449e+00); parsFobs37.push_back(4.65671e+01); parsFobs37.push_back(-4.40296e-02);
  myLRhelper -> setObsFitParameters(37,parsFobs37);

  // fill signal and background contributions to S and B histograms
  doEventloop(); 
  
  // normalize the S and B histograms to construct the pdf's
  //myLRhelper -> normalizeSandBhists();
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
      //signal_solsbranch->SetAddress( & signal_sols );

      //loop over all events in a file 
      for( int ev = 0; ev < signal_events->GetEntries(); ++ ev ) {
        if(Signal_totNrEv>signal_NrEv[nrDir] && signal_NrEv[nrDir] != -1) continue;
	++Signal_totNrEv;
        ++totNrEv;
        if((double)((totNrEv*1.)/5000.) == (double) (totNrEv/5000)) std::cout<< "  Processing signal event "<< totNrEv<<std::endl; 
        signal_solsbranch->SetAddress( & signal_sols );
	signal_solsbranch->GetEntry( ev );
	signal_events->GetEntry( ev , 0 );
        if(signal_sols.size()== 12){
          // get observable values
	  std::vector<double> signal_obsVals;
	  for(int j = 0; j < nrSignalSelObs; j++){
	    if( myLRhelper->obsFitIncluded(obsNrs[j]) ) signal_obsVals.push_back(signal_sols[0].getLRSignalEvtObsVal(obsNrs[j]));
	  }
	  
	  // Fill the observables 
	  // signal: semileptonic top event 
	  myLRhelper -> fillToSignalHists(signal_obsVals);
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
  std::cout<<std::endl<<" Nb of requested events  :"<<(signal_NrEv[nrDir])<<std::endl;
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
      //bckgd_solsbranch->SetAddress( & bckgd_sols );

      //loop over all events in a file 
      for( int ev = 0; ev < bckgd_events->GetEntries(); ++ ev ) {
        if(Bckgd_totNrEv > bckgd_NrEv[nrDir] && bckgd_NrEv[nrDir] != -1) continue;
	++Bckgd_totNrEv;
        ++totNrEv;
        if((double)((totNrEv*1.)/5000.) == (double) (totNrEv/5000)) std::cout<< "  Processing bckgd event "<< totNrEv<<std::endl; 
        bckgd_solsbranch->SetAddress( & bckgd_sols );
	bckgd_solsbranch ->GetEntry( ev );
	bckgd_events->GetEntry( ev , 0 );
        if(bckgd_sols.size()== 12){
          // get observable values
	  std::vector<double> bckgd_obsVals;
	  for(int j = 0; j < nrSignalSelObs; j++){
	    if( myLRhelper->obsFitIncluded(obsNrs[j]) ) bckgd_obsVals.push_back(bckgd_sols[0].getLRSignalEvtObsVal(obsNrs[j]));
	  }
	  // Fill the observables 
	  // bckgd: semileptonic top event 
	  myLRhelper -> fillToBackgroundHists(bckgd_obsVals);
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
    std::cout<<std::endl<<" Nb of requested events  :"<<(bckgd_NrEv[nrDir])<<std::endl;
    std::cout<<std::endl<<" Nb of processed events  :"<<(Bckgd_totNrEv)<<std::endl;
    std::cout<<std::endl<<" Nb of events filled in the histo :"<<(Bckgd_okEvents)<<std::endl;   
    std::cout<<std::endl<<"******************************************************************"<<std::endl;
 }
    std::cout<<std::endl<<"********************  STATISTICS FOR BCKGD ***********************"<<std::endl;
    std::cout<<std::endl<<" Nb of processed events  :"<<(totNrEv)<<std::endl;
    std::cout<<std::endl<<" Nb of events filled in the histo :"<<(okEvents)<<std::endl;   
    std::cout<<std::endl<<"******************************************************************"<<std::endl;
}
