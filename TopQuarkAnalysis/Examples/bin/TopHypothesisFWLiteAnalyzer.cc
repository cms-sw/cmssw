#include <memory>
#include <string>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iostream>

#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"

#include "TopQuarkAnalysis/Examples/bin/NiceStyle.cc"
#include "TopQuarkAnalysis/Examples/interface/RootSystem.h"
#include "TopQuarkAnalysis/Examples/interface/RootHistograms.h"
#include "TopQuarkAnalysis/Examples/interface/RootPostScript.h"

int main(int argc, char* argv[]) 
{
  if( argc<4 ){
    // -------------------------------------------------  
    std::cerr << "ERROR: Wrong number of arguments!"     << std::endl 
	      << "Please specify: * file name"    << std::endl
	      << "                * process name" << std::endl
	      << "                * HypoClassKey" << std::endl
	      << "Example: TopHypothesisFWLiteAnalyzer ttSemiLepEvtBuilder.root TEST kGeom" << std::endl;
    // -------------------------------------------------  
    return -1;
  }

  // get HypoClassKey
  std::string hypoClassKey = argv[3];

  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();
  
  // set nice style for histograms
  setNiceStyle();

  // define some histograms 
  TH1F* hadWPt_    = new TH1F("hadWPt",     "p_{t} (W_{had}) [GeV]", 100,  0., 500.);
  TH1F* hadWMass_  = new TH1F("hadWMass",   "M (W_{had}) [GeV]",      50,  0., 150.);
  TH1F* hadTopPt_  = new TH1F("hadTopPt",   "p_{t} (t_{had}) [GeV]", 100,  0., 500.);
  TH1F* hadTopMass_= new TH1F("hadTopMass", "M (t_{had}) [GeV]",      50, 50., 250.);

  TH1F* lepWPt_    = new TH1F("lepWPt",     "p_{t} (W_{lep}) [GeV]", 100,  0., 500.);
  TH1F* lepWMass_  = new TH1F("lepWMass",   "M (W_{lep}) [GeV]",      50,  0., 150.);
  TH1F* lepTopPt_  = new TH1F("lepTopPt",   "p_{t} (t_{lep}) [GeV]", 100,  0., 500.);
  TH1F* lepTopMass_= new TH1F("lepTopMass", "M (t_{lep}) [GeV]",      50, 50., 250.);

  // -------------------------------------------------  
  std::cout << "open file: " << argv[1] << std::endl;
  // -------------------------------------------------
  TFile* inFile = TFile::Open(argv[1]);
  TTree* events_= 0;
  if( inFile ) inFile->GetObject("Events", events_); 
  if( events_==0 ){
    // -------------------------------------------------  
    std::cerr << "ERROR: Unable to retrieve TTree Events!"           << std::endl
	      << "Either wrong file name or the tree doesn't exist." << std::endl;
    // -------------------------------------------------  
    return -1;
  }

  // acess branch of ttSemiLepEvent
  char decayName[50];
  sprintf(decayName, "recoGenParticles_decaySubset__%s.obj", argv[2]);
  TBranch* decay_   = events_->GetBranch( decayName ); // referred to from within TtGenEvent class
  assert( decay_ != 0 ); 
  char genEvtName[50];
  sprintf(genEvtName, "TtGenEvent_genEvt__%s.obj", argv[2]);
  TBranch* genEvt_  = events_->GetBranch( genEvtName ); // referred to from within TtSemiLeptonicEvent class
  assert( genEvt_ != 0 ); 
  char semiLepEvtName[50];
  sprintf(semiLepEvtName, "TtSemiLeptonicEvent_ttSemiLepEvent__%s.obj", argv[2]);
  TBranch* semiLepEvt_ = events_->GetBranch( semiLepEvtName ); 
  assert( semiLepEvt_ != 0 );
  
  // loop over events and fill histograms  
  int nevt = events_->GetEntries();
  TtSemiLeptonicEvent semiLepEvt;
  // -------------------------------------------------  
  std::cout << "start looping " << nevt << " events..." << std::endl;
  // -------------------------------------------------
  for(int evt=0; evt<nevt; ++evt){
    // set branch address
    semiLepEvt_->SetAddress( &semiLepEvt );
    // get event
    decay_     ->GetEntry( evt );
    genEvt_    ->GetEntry( evt );
    semiLepEvt_->GetEntry( evt );
    events_    ->GetEntry( evt, 0 );

    // -------------------------------------------------  
    if(evt>0 && !(evt%10)) std::cout << "  processing event: " << evt << std::endl;
    // -------------------------------------------------  

    // fill histograms
    if( !semiLepEvt.isHypoAvailable(hypoClassKey) ){
      std::cerr << "NonValidHyp:: " << "Hypothesis not available for this event" << std::endl;
      continue;
    }
    if( !semiLepEvt.isHypoValid(hypoClassKey) ){
      std::cerr << "NonValidHyp::" << "Hypothesis not valid for this event" << std::endl;
      continue;
    }
    
    const reco::Candidate* hadTop = semiLepEvt.hadronicDecayTop(hypoClassKey);
    const reco::Candidate* hadW   = semiLepEvt.hadronicDecayW  (hypoClassKey);
    const reco::Candidate* lepTop = semiLepEvt.leptonicDecayTop(hypoClassKey);
    const reco::Candidate* lepW   = semiLepEvt.leptonicDecayW  (hypoClassKey);
    
    if(hadTop && hadW && lepTop && lepW){
      hadWPt_    ->Fill( hadW->pt()    );
      hadWMass_  ->Fill( hadW->mass()  );
      hadTopPt_  ->Fill( hadTop->pt()  );
      hadTopMass_->Fill( hadTop->mass());
      
      lepWPt_    ->Fill( lepW->pt()    );
      lepWMass_  ->Fill( lepW->mass()  );
      lepTopPt_  ->Fill( lepTop->pt()  );
      lepTopMass_->Fill( lepTop->mass());
    }
  }
  // -------------------------------------------------  
  std::cout << "close file" << std::endl;
  // -------------------------------------------------
  inFile->Close();
  
  // save histograms to file
  TFile outFile( "analyzeHypothesis.root", "recreate" );
  // strip the leading "k" from the hypoClassKey to build directory name
  TString outDir = "analyze" + std::string(hypoClassKey, 1, hypoClassKey.length());
  outFile.mkdir(outDir);
  outFile.cd(outDir);
  hadWPt_    ->Write( );
  hadWMass_  ->Write( );
  hadTopPt_  ->Write( );
  hadTopMass_->Write( );
  lepWPt_    ->Write( );
  lepWMass_  ->Write( );
  lepTopPt_  ->Write( );
  lepTopMass_->Write( );
  outFile.Close();
  
  // free allocated space
  delete hadWPt_;
  delete hadWMass_;
  delete hadTopPt_;
  delete hadTopMass_;
  delete lepWPt_;
  delete lepWMass_;
  delete lepTopPt_;
  delete lepTopMass_;

  return 0;
}
