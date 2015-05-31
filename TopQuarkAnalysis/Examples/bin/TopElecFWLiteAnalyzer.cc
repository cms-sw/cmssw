#include <memory>
#include <string>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iostream>

#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "TopQuarkAnalysis/Examples/bin/NiceStyle.cc"
#include "TopQuarkAnalysis/Examples/interface/RootSystem.h"
#include "TopQuarkAnalysis/Examples/interface/RootHistograms.h"
#include "TopQuarkAnalysis/Examples/interface/RootPostScript.h"


int main(int argc, char* argv[]) 
{
  if( argc<3 ){
    // ------------------------------------------------- 
    std::cerr << "ERROR:: " 
	      << "Wrong number of arguments! Please specify:" << std::endl
	      << "        * filepath" << std::endl
	      << "        * process name" << std::endl; 
    // -------------------------------------------------  
    return -1;
  }

  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();
  
  // set nice style for histograms
  setNiceStyle();

  // define some histograms 
  TH1I* noElecs  = new TH1I("noElecs", "N_{Elecs}",      10,  0 , 10 );
  TH1F* ptElecs  = new TH1F("ptElecs", "pt_{Elecs}",    100,  0.,300.);
  TH1F* enElecs  = new TH1F("enElecs", "energy_{Elecs}",100,  0.,300.);
  TH1F* etaElecs = new TH1F("etaElecs","eta_{Elecs}",   100, -3.,  3.);
  TH1F* phiElecs = new TH1F("phiElecs","phi_{Elecs}",   100, -5.,  5.);  
  
  // -------------------------------------------------  
  std::cout << "open  file: " << argv[1] << std::endl;
  // -------------------------------------------------
  TFile* inFile = TFile::Open(argv[1]);
  TTree* events_= 0;
  if( inFile ) inFile->GetObject("Events", events_); 
  if( events_==0 ){
    // -------------------------------------------------  
    std::cerr << "ERROR:: " 
	      << "Unable to retrieve TTree Events!" << std::endl
	      << "        Eighter wrong file name or the the tree doesn't exists" << std::endl;
    // -------------------------------------------------  
    return -1;
  }

  // acess branch of elecs
  char elecName[50];
  sprintf(elecName, "patElectrons_selectedPatElectrons__%s.obj", argv[2]);
  TBranch* elecs_ = events_->GetBranch( elecName ); assert( elecs_!=0 );
  
  // loop over events and fill histograms
  std::vector<pat::Electron> elecs;
  int nevt = events_->GetEntries();

  // -------------------------------------------------  
  std::cout << "start looping " << nevt << " events..." << std::endl;
  // -------------------------------------------------
  for(int evt=0; evt<nevt; ++evt){
    // set branch address 
    elecs_->SetAddress( &elecs );
    // get event
    elecs_ ->GetEntry( evt );
    events_->GetEntry( evt, 0 );

    // -------------------------------------------------  
    if(evt>0 && !(evt%10)) std::cout << "  processing event: " << evt << std::endl;
    // -------------------------------------------------  

    // fill histograms
    noElecs->Fill(elecs.size());
    for(unsigned idx=0; idx<elecs.size(); ++idx){
      // fill histograms
      ptElecs ->Fill(elecs[idx].pt()    );
      enElecs ->Fill(elecs[idx].energy());
      etaElecs->Fill(elecs[idx].eta()   );
      phiElecs->Fill(elecs[idx].phi()   );
    }
  }
  // -------------------------------------------------  
  std::cout << "close file" << std::endl;
  // -------------------------------------------------
  inFile->Close();

  // save histograms to file
  TFile outFile( "analyzeElecs.root", "recreate" );
  outFile.mkdir("analyzeElec");
  outFile.cd("analyzeElec");
  noElecs ->Write( );
  ptElecs ->Write( );
  enElecs ->Write( );
  etaElecs->Write( );
  phiElecs->Write( );
  outFile.Close();

  // free allocated space
  delete noElecs;
  delete ptElecs;
  delete enElecs;
  delete etaElecs;
  delete phiElecs;

  return 0;
}
