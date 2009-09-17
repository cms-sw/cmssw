/**_________________________________________________________________
   class:   BeamSpotAnalyzer.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)

 version $Id: BeamSpotAnalyzer.cc,v 1.9 2009/08/25 18:54:40 jengbou Exp $

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

  outputfilename_ = iConfig.getUntrackedParameter<std::string>("OutputFileName");
  
//   file_ = TFile::Open(outputfilename_.c_str(),"RECREATE");

//   ftree_ = new TTree("mytree","mytree");
//   ftree_->AutoSave();
  
//   ftree_->Branch("pt",&fpt,"fpt/D");
//   ftree_->Branch("d0",&fd0,"fd0/D");
//   ftree_->Branch("sigmad0",&fsigmad0,"fsigmad0/D");
//   ftree_->Branch("phi0",&fphi0,"fphi0/D");
//   ftree_->Branch("z0",&fz0,"fz0/D");
//   ftree_->Branch("sigmaz0",&fsigmaz0,"fsigmaz0/D");
//   ftree_->Branch("theta",&ftheta,"ftheta/D");
//   ftree_->Branch("eta",&feta,"feta/D");
//   ftree_->Branch("charge",&fcharge,"fcharge/I");
//   ftree_->Branch("chi2",&fchi2,"fchi2/D");
//   ftree_->Branch("ndof",&fndof,"fndof/D");
//   ftree_->Branch("nHit",&fnHit,"fnHit/i");
//   ftree_->Branch("nStripHit",&fnStripHit,"fnStripHit/i");
//   ftree_->Branch("nPixelHit",&fnPixelHit,"fnPixelHit/i");
//   ftree_->Branch("nTIBHit",&fnTIBHit,"fnTIBHit/i");
//   ftree_->Branch("nTOBHit",&fnTOBHit,"fnTOBHit/i");
//   ftree_->Branch("nTIDHit",&fnTIDHit,"fnTIDHit/i");
//   ftree_->Branch("nTECHit",&fnTECHit,"fnTECHit/i");
//   ftree_->Branch("nPXBHit",&fnPXBHit,"fnPXBHit/i");
//   ftree_->Branch("nPXFHit",&fnPXFHit,"fnPXFHit/i");
//   ftree_->Branch("cov",&fcov,"fcov[7][7]/D");
  
   
//   fBSvector.clear();

  
  // get parameter
  write2DB_ = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("WriteToDB");
  runallfitters_ = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("RunAllFitters");
  
  theBeamFitter = new BeamFitter(iConfig);
  theBeamFitter->resetTrkVector();

  ftotal_tracks = 0;
  ftotalevents = 0;
  
}


BeamSpotAnalyzer::~BeamSpotAnalyzer()
{
  delete theBeamFitter;
}


void
BeamSpotAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	
// 	ftree_->SetBranchAddress("theta",&ftheta);
// 	ftree_->SetBranchAddress("pt",&fpt);
// 	ftree_->SetBranchAddress("eta",&feta);
// 	ftree_->SetBranchAddress("charge",&fcharge);
// 	ftree_->SetBranchAddress("chi2",&fchi2);
// 	ftree_->SetBranchAddress("ndof",&fndof);
//      ftree_->SetBranchAddress("d0",&fd0);
// 	ftree_->SetBranchAddress("sigmad0",&fsigmad0);
// 	ftree_->SetBranchAddress("phi0",&fphi0);
// 	ftree_->SetBranchAddress("z0",&fz0);
// 	ftree_->SetBranchAddress("sigmaz0",&fsigmaz0);
// 	ftree_->SetBranchAddress("nHit",&fnHit);
// 	ftree_->SetBranchAddress("nStripHit",&fnStripHit);
// 	ftree_->SetBranchAddress("nPixelHit",&fnPixelHit);
// 	ftree_->SetBranchAddress("nTIBHit",&fnTIBHit);
// 	ftree_->SetBranchAddress("nTOBHit",&fnTOBHit);
// 	ftree_->SetBranchAddress("nTIDHit",&fnTIDHit);
// 	ftree_->SetBranchAddress("nTECHit",&fnTECHit);
// 	ftree_->SetBranchAddress("nPXBHit",&fnPXBHit);
// 	ftree_->SetBranchAddress("nPXFHit",&fnPXFHit);
// 	ftree_->SetBranchAddress("cov",&fcov);

	
// 	  ftree_->Fill();

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
