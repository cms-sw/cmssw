#include <memory>
#include <string>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iostream>

#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

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
  TH1I* noJets  = new TH1I("noJets", "N_{Jets}",      10,  0 , 10 );
  TH1F* ptJets  = new TH1F("ptJets", "pt_{Jets}",    100,  0.,300.);
  TH1F* enJets  = new TH1F("enJets", "energy_{Jets}",100,  0.,300.);
  TH1F* etaJets = new TH1F("etaJets","eta_{Jets}",   100, -3.,  3.);
  TH1F* phiJets = new TH1F("phiJets","phi_{Jets}",   100, -5.,  5.);  
  
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
  char jetsName[50];
  sprintf(jetsName, "patJets_selectedPatJets__%s.obj", argv[2]);
  TBranch* jets_ = events_->GetBranch( jetsName ); assert( jets_!=0 );
  
  // loop over events and fill histograms
  std::vector<pat::Jet> jets;
  int nevt = events_->GetEntries();

  // -------------------------------------------------  
  std::cout << "start looping " << nevt << " events..." << std::endl;
  // -------------------------------------------------
  for(int evt=0; evt<nevt; ++evt){
    // set branch address 
    jets_->SetAddress( &jets );
    // get event
    jets_ ->GetEntry( evt );
    events_->GetEntry( evt, 0 );

    // -------------------------------------------------  
    if(evt>0 && !(evt%10)) std::cout << "  processing event: " << evt << std::endl;
    // -------------------------------------------------  

    // fill histograms
    noJets->Fill(jets.size());
    for(unsigned idx=0; idx<jets.size(); ++idx){
      // fill histograms
      ptJets ->Fill(jets[idx].pt()    );
      enJets ->Fill(jets[idx].energy());
      etaJets->Fill(jets[idx].eta()   );
      phiJets->Fill(jets[idx].phi()   );
    }
  }
  // -------------------------------------------------  
  std::cout << "close file" << std::endl;
  // -------------------------------------------------
  inFile->Close();

  // save histograms to file
  TFile outFile( "analyzeJets.root", "recreate" );
  outFile.mkdir("analyzeJet");
  outFile.cd("analyzeJet");
  noJets ->Write( );
  ptJets ->Write( );
  enJets ->Write( );
  etaJets->Write( );
  phiJets->Write( );
  outFile.Close();

  // free allocated space
  delete noJets;
  delete ptJets;
  delete enJets;
  delete etaJets;
  delete phiJets;

  return 0;
}
