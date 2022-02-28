/**_________________________________________________________________
   class:   BeamSpotAnalyzer.cc
   package: RecoVertex/BeamSpotProducer

 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)

________________________________________________________________**/

// C++ standard
#include <string>

// CMS
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"

#include "TMath.h"

class BeamSpotAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  explicit BeamSpotAnalyzer(const edm::ParameterSet&);
  ~BeamSpotAnalyzer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) override;
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) override;

  int ftotalevents;
  int fitNLumi_;
  int resetFitNLumi_;
  int countLumi_;  //counter
  int org_resetFitNLumi_;
  int previousLumi_;
  int previousRun_;
  int ftmprun0, ftmprun;
  int beginLumiOfBSFit_;
  int endLumiOfBSFit_;
  std::time_t refBStime[2];

  bool write2DB_;
  bool runbeamwidthfit_;
  bool runallfitters_;

  BeamFitter* theBeamFitter;
};

BeamSpotAnalyzer::BeamSpotAnalyzer(const edm::ParameterSet& iConfig) {
  // get parameter
  write2DB_ = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("WriteToDB");
  runallfitters_ = iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("RunAllFitters");
  fitNLumi_ =
      iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getUntrackedParameter<int>("fitEveryNLumi", -1);
  resetFitNLumi_ =
      iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getUntrackedParameter<int>("resetEveryNLumi", -1);
  runbeamwidthfit_ =
      iConfig.getParameter<edm::ParameterSet>("BSAnalyzerParameters").getParameter<bool>("RunBeamWidthFit");

  theBeamFitter = new BeamFitter(iConfig, consumesCollector());
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
  org_resetFitNLumi_ = resetFitNLumi_;
}

BeamSpotAnalyzer::~BeamSpotAnalyzer() { delete theBeamFitter; }

void BeamSpotAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ftotalevents++;
  theBeamFitter->readEvent(iEvent);
  ftmprun = iEvent.id().run();
}

//--------------------------------------------------------
void BeamSpotAnalyzer::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) {
  const edm::TimeValue_t fbegintimestamp = lumiSeg.beginTime().value();
  const std::time_t ftmptime = fbegintimestamp >> 32;

  if (countLumi_ == 0 || (resetFitNLumi_ > 0 && countLumi_ % resetFitNLumi_ == 0)) {
    ftmprun0 = lumiSeg.run();
    ftmprun = ftmprun0;
    beginLumiOfBSFit_ = lumiSeg.luminosityBlock();
    refBStime[0] = ftmptime;
  }

  countLumi_++;
  if (ftmprun == previousRun_) {
    if ((previousLumi_ + 1) != int(lumiSeg.luminosityBlock()))
      edm::LogWarning("BeamSpotAnalyzer") << "LUMI SECTIONS ARE NOT SORTED!";
  }
}

//--------------------------------------------------------
void BeamSpotAnalyzer::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup) {
  edm::LogPrint("BeamSpotAnalyzer") << "for lumis " << beginLumiOfBSFit_ << " - " << endLumiOfBSFit_ << std::endl
                                    << "number of selected tracks = " << theBeamFitter->getNTracks();
  edm::LogPrint("BeamSpotAnalyzer") << "number of selected PVs = " << theBeamFitter->getNPVs();
  //edm::LogPrint("BeamSpotAnalyzer") << "number of selected PVs per bx: " << theBeamFitter->getNPVsperBX() std::endl;

  const edm::TimeValue_t fendtimestamp = lumiSeg.endTime().value();
  const std::time_t fendtime = fendtimestamp >> 32;
  refBStime[1] = fendtime;

  endLumiOfBSFit_ = lumiSeg.luminosityBlock();
  previousLumi_ = endLumiOfBSFit_;

  if (fitNLumi_ == -1 && resetFitNLumi_ == -1)
    return;

  if (fitNLumi_ > 0 && countLumi_ % fitNLumi_ != 0)
    return;

  theBeamFitter->setFitLSRange(beginLumiOfBSFit_, endLumiOfBSFit_);
  theBeamFitter->setRefTime(refBStime[0], refBStime[1]);
  theBeamFitter->setRun(ftmprun0);

  std::pair<int, int> LSRange = theBeamFitter->getFitLSRange();

  if (theBeamFitter->runPVandTrkFitter()) {
    reco::BeamSpot bs = theBeamFitter->getBeamSpot();
    edm::LogPrint("BeamSpotAnalyzer") << "\n RESULTS OF DEFAULT FIT ";
    edm::LogPrint("BeamSpotAnalyzer") << " for runs: " << ftmprun0 << " - " << ftmprun;
    edm::LogPrint("BeamSpotAnalyzer") << " for lumi blocks : " << LSRange.first << " - " << LSRange.second;
    edm::LogPrint("BeamSpotAnalyzer") << " lumi counter # " << countLumi_;
    edm::LogPrint("BeamSpotAnalyzer") << bs;
    edm::LogPrint("BeamSpotAnalyzer") << "[BeamFitter] fit done. \n";
  } else {  // Fill in empty beam spot if beamfit fails
    reco::BeamSpot bs;
    bs.setType(reco::BeamSpot::Fake);
    edm::LogPrint("BeamSpotAnalyzer") << "\n Empty Beam spot fit";
    edm::LogPrint("BeamSpotAnalyzer") << " for runs: " << ftmprun0 << " - " << ftmprun;
    edm::LogPrint("BeamSpotAnalyzer") << " for lumi blocks : " << LSRange.first << " - " << LSRange.second;
    edm::LogPrint("BeamSpotAnalyzer") << " lumi counter # " << countLumi_;
    edm::LogPrint("BeamSpotAnalyzer") << bs;
    edm::LogPrint("BeamSpotAnalyzer") << "[BeamFitter] fit failed \n";
    // accumulate more events
    // disable this for the moment
    //resetFitNLumi_ += 1;
    //edm::LogPrint("BeamSpotAnalyzer") << "reset fitNLumi " << resetFitNLumi_ ;
  }

  if (resetFitNLumi_ > 0 && countLumi_ % resetFitNLumi_ == 0) {
    std::vector<BSTrkParameters> theBSvector = theBeamFitter->getBSvector();
    edm::LogPrint("BeamSpotAnalyzer") << "Total number of tracks accumulated = " << theBSvector.size();
    edm::LogPrint("BeamSpotAnalyzer") << "Reset track collection for beam fit";
    theBeamFitter->resetTrkVector();
    theBeamFitter->resetLSRange();
    theBeamFitter->resetCutFlow();
    theBeamFitter->resetRefTime();
    theBeamFitter->resetPVFitter();
    countLumi_ = 0;
    // reset counter to orginal
    resetFitNLumi_ = org_resetFitNLumi_;
  }
}

void BeamSpotAnalyzer::endJob() {
  edm::LogPrint("BeamSpotAnalyzer") << "\n-------------------------------------\n";
  edm::LogPrint("BeamSpotAnalyzer") << "\n Total number of events processed: " << ftotalevents;
  edm::LogPrint("BeamSpotAnalyzer") << "\n-------------------------------------\n\n";

  if (fitNLumi_ == -1 && resetFitNLumi_ == -1) {
    if (theBeamFitter->runPVandTrkFitter()) {
      reco::BeamSpot beam_default = theBeamFitter->getBeamSpot();
      std::pair<int, int> LSRange = theBeamFitter->getFitLSRange();

      edm::LogPrint("BeamSpotAnalyzer") << "\n RESULTS OF DEFAULT FIT:";
      edm::LogPrint("BeamSpotAnalyzer") << " for runs: " << ftmprun0 << " - " << ftmprun;
      edm::LogPrint("BeamSpotAnalyzer") << " for lumi blocks : " << LSRange.first << " - " << LSRange.second;
      edm::LogPrint("BeamSpotAnalyzer") << " lumi counter # " << countLumi_;
      edm::LogPrint("BeamSpotAnalyzer") << beam_default;

      if (write2DB_) {
        edm::LogPrint("BeamSpotAnalyzer") << "\n-------------------------------------\n\n";
        edm::LogPrint("BeamSpotAnalyzer") << " write results to DB...";
        theBeamFitter->write2DB();
      }

      if (runallfitters_) {
        theBeamFitter->runAllFitter();
      }
    }
    if ((runbeamwidthfit_)) {
      theBeamFitter->runBeamWidthFitter();
      reco::BeamSpot beam_width = theBeamFitter->getBeamWidth();
      edm::LogPrint("BeamSpotAnalyzer") << beam_width;
    } else {
      edm::LogPrint("BeamSpotAnalyzer") << "[BeamSpotAnalyzer] beamfit fails !!!";
    }
  }

  edm::LogPrint("BeamSpotAnalyzer") << "[BeamSpotAnalyzer] endJob done \n";
}

void BeamSpotAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Analyzer of BeamSpot Objects");

  edm::ParameterSetDescription bsAnalyzerParamsDesc;
  bsAnalyzerParamsDesc.add("WriteToDB", false);
  bsAnalyzerParamsDesc.add("RunAllFitters", false);
  bsAnalyzerParamsDesc.addUntracked("fitEveryNLumi", -1);
  bsAnalyzerParamsDesc.addUntracked("resetEveryNLumi", -1);
  bsAnalyzerParamsDesc.add("RunBeamWidthFit", false);
  desc.add<edm::ParameterSetDescription>("BSAnalyzerParameters", bsAnalyzerParamsDesc);

  BeamFitter::fillDescription(desc);
  PVFitter::fillDescription(desc);

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotAnalyzer);
