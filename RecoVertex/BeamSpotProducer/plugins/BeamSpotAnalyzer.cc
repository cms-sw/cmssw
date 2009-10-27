/**_________________________________________________________________
   class:   BeamSpotAnalyzer.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)

 version $Id: BeamSpotAnalyzer.cc,v 1.11 2009/09/18 20:47:54 jengbou Exp $

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
  write2DB_       = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("WriteToDB");
  runallfitters_  = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("RunAllFitters");
  fitNLumi_       = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getUntrackedParameter<int>("fitEveryNLumi",-1);
  resetFitNLumi_  = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getUntrackedParameter<int>("resetEveryNLumi",-1);

  theBeamFitter = new BeamFitter(iConfig);
  theBeamFitter->resetTrkVector();

  ftotalevents = 0;
  ftmprun0 = ftmprun = -1;
  ftmplumi0 = ftmplumi = -1;
  countLumi_ = 0;
}


BeamSpotAnalyzer::~BeamSpotAnalyzer()
{
  delete theBeamFitter;
}


void
BeamSpotAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	ftotalevents++;
	theBeamFitter->readEvent(iEvent);
	ftmprun = iEvent.id().run();
	ftmplumi = iEvent.luminosityBlock();
}



void 
BeamSpotAnalyzer::beginJob(const edm::EventSetup&)
{
}

//--------------------------------------------------------
void
BeamSpotAnalyzer::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
									   const edm::EventSetup& context) {

	if ( countLumi_ == 0 || (resetFitNLumi_ > 0 && countLumi_%resetFitNLumi_ == 0) ) {
		ftmprun0 = lumiSeg.run();
		ftmplumi0 = lumiSeg.id().luminosityBlock();
	}
	countLumi_++;
	//std::cout << "Lumi # " << countLumi_ << std::endl;
	
}

//--------------------------------------------------------
void
BeamSpotAnalyzer::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
									 const edm::EventSetup& iSetup) {

	if ( fitNLumi_ == -1 && resetFitNLumi_ == -1 ) return;
	
	if (fitNLumi_ > 0 && countLumi_%fitNLumi_!=0) return;

	if (theBeamFitter->runFitter()){
		reco::BeamSpot bs = theBeamFitter->getBeamSpot();
		std::cout << "\n RESULTS OF DEFAULT FIT " << std::endl;
		std::cout << " for runs: " << ftmprun0 << " - " << ftmprun << std::endl;
		std::cout << " for lumi blocks : " << ftmplumi0 << " - " << ftmplumi << std::endl;
		std::cout << " lumi counter # " << countLumi_ << std::endl;
		std::cout << bs << std::endl;
		std::cout << "[BeamFitter] fit done. \n" << std::endl;	
	}
	else { // Fill in empty beam spot if beamfit fails
		reco::BeamSpot bs;
		bs.setType(reco::BeamSpot::Fake);
		std::cout << "\n Empty Beam spot fit" << std::endl;
		std::cout << " for runs: " << ftmprun0 << " - " << ftmprun << std::endl;
		std::cout << " for lumi blocks : " << ftmplumi0 << " - " << ftmplumi << std::endl;
		std::cout << " lumi counter # " << countLumi_ << std::endl;
		std::cout << bs << std::endl;
		std::cout << "[BeamFitter] fit failed \n" << std::endl;
	}

	
	if (resetFitNLumi_ > 0 && countLumi_%resetFitNLumi_ == 0) {
		std::vector<BSTrkParameters> theBSvector = theBeamFitter->getBSvector();
		std::cout << "Total number of tracks accumulated = " << theBSvector.size() << std::endl;
		std::cout << "Reset track collection for beam fit" <<std::endl;
		theBeamFitter->resetTrkVector();
	}

}


void 
BeamSpotAnalyzer::endJob() {
  std::cout << "\n-------------------------------------\n" << std::endl;
  std::cout << "\n Total number of events processed: "<< ftotalevents << std::endl;
  std::cout << "\n-------------------------------------\n\n" << std::endl;

  if ( fitNLumi_ == -1 && resetFitNLumi_ == -1 ) {
	  
	  if(theBeamFitter->runFitter()){
		  reco::BeamSpot beam_default = theBeamFitter->getBeamSpot();
    
		  std::cout << "\n RESULTS OF DEFAULT FIT:" << std::endl;
		  std::cout << " for runs: " << ftmprun0 << " - " << ftmprun << std::endl;
		  std::cout << " for lumi blocks : " << ftmplumi0 << " - " << ftmplumi << std::endl;
		  std::cout << " lumi counter # " << countLumi_ << std::endl;
		  std::cout << beam_default << std::endl;
    
		  if (write2DB_) {
			  std::cout << "\n-------------------------------------\n\n" << std::endl;
			  std::cout << " write results to DB..." << std::endl;
			  theBeamFitter->write2DB();
		  }
    
		  if (runallfitters_) {
			  theBeamFitter->runAllFitter();     
		  }

	  }
	  else std::cout << "[BeamSpotAnalyzer] beamfit fails !!!" << std::endl;
  }
  
  std::cout << "[BeamSpotAnalyzer] endJob done \n" << std::endl;
}

//define this as a plug-in
DEFINE_ANOTHER_FWK_MODULE(BeamSpotAnalyzer);
