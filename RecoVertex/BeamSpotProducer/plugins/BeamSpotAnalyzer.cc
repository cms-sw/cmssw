/**_________________________________________________________________
   class:   BeamSpotAnalyzer.cc
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)

 version $Id: BeamSpotAnalyzer.cc,v 1.32 2011/02/22 17:52:25 friis Exp $

________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamSpotAnalyzer.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TMath.h"

BeamSpotAnalyzer::BeamSpotAnalyzer(const edm::ParameterSet& iConfig)
{
  // get parameter
  write2DB_       = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("WriteToDB");
  runallfitters_  = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("RunAllFitters");
  fitNLumi_       = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getUntrackedParameter<int>("fitEveryNLumi",-1);
  resetFitNLumi_  = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getUntrackedParameter<int>("resetEveryNLumi",-1);
  runbeamwidthfit_  = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("RunBeamWidthFit");

  theBeamFitter = new BeamFitter(iConfig);
  theBeamFitter->resetTrkVector();
  theBeamFitter->resetLSRange();
  theBeamFitter->resetCutFlow();
  theBeamFitter->resetRefTime();
  theBeamFitter->resetPVFitter();

  ftotalevents = 0;
  ftmprun0 = ftmprun = -1;
  countLumi_ = 0;
  beginLumiOfBSFit_ = endLumiOfBSFit_ = -1;
  previousLumi_ = previousRun_ = 0;
  Org_resetFitNLumi_ = resetFitNLumi_;
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

}



void
BeamSpotAnalyzer::beginJob()
{
}

//--------------------------------------------------------
void
BeamSpotAnalyzer::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
									   const edm::EventSetup& context) {

    const edm::TimeValue_t fbegintimestamp = lumiSeg.beginTime().value();
    const std::time_t ftmptime = fbegintimestamp >> 32;

    if ( countLumi_ == 0 || (resetFitNLumi_ > 0 && countLumi_%resetFitNLumi_ == 0) ) {
        ftmprun0 = lumiSeg.run();
        ftmprun = ftmprun0;
        beginLumiOfBSFit_ = lumiSeg.luminosityBlock();
        refBStime[0] = ftmptime;
    }

    countLumi_++;
    //std::cout << "Lumi # " << countLumi_ << std::endl;
    if ( ftmprun == previousRun_ ) {
      if ( (previousLumi_ + 1) != int(lumiSeg.luminosityBlock()) )
	edm::LogWarning("BeamSpotAnalyzer") << "LUMI SECTIONS ARE NOT SORTED!";
    }
}

//--------------------------------------------------------
void
BeamSpotAnalyzer::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
									 const edm::EventSetup& iSetup) {

  //LogDebug("BeamSpotAnalyzer") <<
  std::cout <<
    "for lumis "<< beginLumiOfBSFit_ << " - " << endLumiOfBSFit_ << std::endl <<
    "number of selected tracks = " << theBeamFitter->getNTracks() << std::endl;
  std::cout << "number of selected PVs = " << theBeamFitter->getNPVs() << std::endl;
  //std::cout << "number of selected PVs per bx: " << std::endl;
  //theBeamFitter->getNPVsperBX();

    const edm::TimeValue_t fendtimestamp = lumiSeg.endTime().value();
    const std::time_t fendtime = fendtimestamp >> 32;
    refBStime[1] = fendtime;

    endLumiOfBSFit_ = lumiSeg.luminosityBlock();
    previousLumi_ = endLumiOfBSFit_;

    if ( fitNLumi_ == -1 && resetFitNLumi_ == -1 ) return;

    if (fitNLumi_ > 0 && countLumi_%fitNLumi_!=0) return;

    theBeamFitter->setFitLSRange(beginLumiOfBSFit_,endLumiOfBSFit_);
    theBeamFitter->setRefTime(refBStime[0],refBStime[1]);
    theBeamFitter->setRun(ftmprun0);

    std::pair<int,int> LSRange = theBeamFitter->getFitLSRange();

    if (theBeamFitter->runPVandTrkFitter()){
      reco::BeamSpot bs = theBeamFitter->getBeamSpot();
      std::cout << "\n RESULTS OF DEFAULT FIT " << std::endl;
      std::cout << " for runs: " << ftmprun0 << " - " << ftmprun << std::endl;
      std::cout << " for lumi blocks : " << LSRange.first << " - " << LSRange.second << std::endl;
      std::cout << " lumi counter # " << countLumi_ << std::endl;
      std::cout << bs << std::endl;
      std::cout << "[BeamFitter] fit done. \n" << std::endl;
    }
    else { // Fill in empty beam spot if beamfit fails
      reco::BeamSpot bs;
      bs.setType(reco::BeamSpot::Fake);
      std::cout << "\n Empty Beam spot fit" << std::endl;
      std::cout << " for runs: " << ftmprun0 << " - " << ftmprun << std::endl;
      std::cout << " for lumi blocks : " << LSRange.first << " - " << LSRange.second << std::endl;
      std::cout << " lumi counter # " << countLumi_ << std::endl;
      std::cout << bs << std::endl;
      std::cout << "[BeamFitter] fit failed \n" << std::endl;
      //accumulate more events
      // dissable this for the moment
      //resetFitNLumi_ += 1;
      //std::cout << "reset fitNLumi " << resetFitNLumi_ << std::endl;
    }

    if (resetFitNLumi_ > 0 && countLumi_%resetFitNLumi_ == 0) {
      std::vector<BSTrkParameters> theBSvector = theBeamFitter->getBSvector();
      std::cout << "Total number of tracks accumulated = " << theBSvector.size() << std::endl;
      std::cout << "Reset track collection for beam fit" <<std::endl;
      theBeamFitter->resetTrkVector();
      theBeamFitter->resetLSRange();
      theBeamFitter->resetCutFlow();
      theBeamFitter->resetRefTime();
      theBeamFitter->resetPVFitter();
      countLumi_=0;
      // reset counter to orginal
      resetFitNLumi_ = Org_resetFitNLumi_;
    }

}


void
BeamSpotAnalyzer::endJob() {
  std::cout << "\n-------------------------------------\n" << std::endl;
  std::cout << "\n Total number of events processed: "<< ftotalevents << std::endl;
  std::cout << "\n-------------------------------------\n\n" << std::endl;

  if ( fitNLumi_ == -1 && resetFitNLumi_ == -1 ) {

	  if(theBeamFitter->runPVandTrkFitter()){
		  reco::BeamSpot beam_default = theBeamFitter->getBeamSpot();
                  std::pair<int,int> LSRange = theBeamFitter->getFitLSRange();

		  std::cout << "\n RESULTS OF DEFAULT FIT:" << std::endl;
		  std::cout << " for runs: " << ftmprun0 << " - " << ftmprun << std::endl;
		  std::cout << " for lumi blocks : " << LSRange.first << " - " << LSRange.second << std::endl;
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
      if ((runbeamwidthfit_)){
                 theBeamFitter->runBeamWidthFitter();
                 reco::BeamSpot beam_width = theBeamFitter->getBeamWidth();
                 std::cout <<beam_width<< std::endl;
      }

	  else std::cout << "[BeamSpotAnalyzer] beamfit fails !!!" << std::endl;
  }

  std::cout << "[BeamSpotAnalyzer] endJob done \n" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotAnalyzer);
