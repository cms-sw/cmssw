#ifndef HypothesisAnalyzer_h
#define HypothesisAnalyzer_h

#include "TH1F.h"
#include "TH2F.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"

class HypothesisAnalyzer : public edm::EDAnalyzer {

 public:

  explicit HypothesisAnalyzer(const edm::ParameterSet&);
  ~HypothesisAnalyzer(){};
  
 private:

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  const edm::InputTag semiLepEvt_;
  const std::string hypoClassKey_;

  TH1F* neutrinoEta_;
  TH1F* neutrinoPullEta_;

  TH1F* hadWPt_;
  TH1F* hadWEta_;
  TH1F* hadWMass_;

  TH1F* hadWPullPt_;
  TH1F* hadWPullEta_;
  TH1F* hadWPullMass_;

  TH1F* hadTopPt_;
  TH1F* hadTopEta_;
  TH1F* hadTopMass_;

  TH1F* hadTopPullPt_;
  TH1F* hadTopPullEta_;
  TH1F* hadTopPullMass_;

  TH1F* lepWPt_;
  TH1F* lepWEta_;
  TH1F* lepWMass_;

  TH1F* lepWPullPt_;
  TH1F* lepWPullEta_;
  TH1F* lepWPullMass_;

  TH1F* lepTopPt_;
  TH1F* lepTopEta_;
  TH1F* lepTopMass_;

  TH1F* topPairMass_;
  TH1F* topPairPullMass_;

  TH1F* lepTopPullPt_;
  TH1F* lepTopPullEta_;
  TH1F* lepTopPullMass_;

  TH1F* genMatchDr_;
  TH1F* kinFitProb_;

  TH2F* genMatchDrVsHadTopPullMass_;
  TH2F* kinFitProbVsHadTopPullMass_;

};

#endif
