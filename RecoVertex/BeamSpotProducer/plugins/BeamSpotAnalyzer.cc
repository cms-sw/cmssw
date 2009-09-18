/**_________________________________________________________________
   class:   BeamSpotAnalyzer.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)

 version $Id: BeamSpotAnalyzer.cc,v 1.10 2009/09/17 21:49:42 jengbou Exp $

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamSpotAnalyzer.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TMath.h"

BeamSpotAnalyzer::BeamSpotAnalyzer(const edm::ParameterSet& iConfig)
{
  // get parameter
  write2DB_ = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("WriteToDB");
  runallfitters_ = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("RunAllFitters");
  
  theBeamFitter = new BeamFitter(iConfig);
  theBeamFitter->resetTrkVector();

  ftotalevents = 0;
  
}


BeamSpotAnalyzer::~BeamSpotAnalyzer()
{
  delete theBeamFitter;
}


void
BeamSpotAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	
  theBeamFitter->readEvent(iEvent);
  ftotalevents++;

}



void 
BeamSpotAnalyzer::beginJob(const edm::EventSetup&)
{
}

void 
BeamSpotAnalyzer::endJob() {
  std::cout << "\n-------------------------------------\n" << std::endl;
  std::cout << "\n Total number of events processed: "<< ftotalevents << std::endl;
  std::cout << "\n-------------------------------------\n\n" << std::endl;
  
  if(theBeamFitter->runFitter()){
    reco::BeamSpot beam_default = theBeamFitter->getBeamSpot();
    
    std::cout << "\n RESULTS OF DEFAULT FIT:" << std::endl;
    std::cout << beam_default << std::endl;
    
    if (write2DB_) {
      std::cout << "\n-------------------------------------\n\n" << std::endl;
      std::cout << " write results to DB..." << std::endl;
      theBeamFitter->write2DB();
    }
    
    if (runallfitters_) {
      theBeamFitter->runAllFitter();
      
// 	// add new branches
// 	std::cout << " add new branches to output file " << std::endl;
// 	beam_default = myalgo->Fit_d0phi();
// 	file_->cd();
// 	TTree *newtree = new TTree("mytreecorr","mytreecorr");
// 	newtree->Branch("d0phi_chi2",&fd0phi_chi2,"fd0phi_chi2/D");
// 	newtree->Branch("d0phi_d0",&fd0phi_d0,"fd0phi_d0/D");
// 	newtree->SetBranchAddress("d0phi_chi2",&fd0phi_chi2);
// 	newtree->SetBranchAddress("d0phi_d0",&fd0phi_d0);
// 	std::vector<BSTrkParameters>  tmpvector = myalgo->GetData();
	
// 	std::vector<BSTrkParameters>::iterator iparam = tmpvector.begin();
// 	for( iparam = tmpvector.begin() ;
// 		 iparam != tmpvector.end() ; ++iparam) {
// 		fd0phi_chi2 = iparam->d0phi_chi2();
// 		fd0phi_d0   = iparam->d0phi_d0();
// 		newtree->Fill();
// 	}
// 	newtree->Write();
    }

	// let's close everything
// 	file_->cd();
// 	ftree_->Write();
// 	file_->Close();

  }
  else std::cout << "[BeamSpotAnalyzer] beamfit fails !!!" << std::endl;

  std::cout << "[BeamSpotAnalyzer] endJob done \n" << std::endl;
}

//define this as a plug-in
DEFINE_ANOTHER_FWK_MODULE(BeamSpotAnalyzer);
