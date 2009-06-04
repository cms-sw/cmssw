#ifndef HypothesisAnalyzer_h
#define HypothesisAnalyzer_h

#include "TH1F.h"
#include "TH2F.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

class HypothesisAnalyzer : public edm::EDAnalyzer {

 public:

  explicit HypothesisAnalyzer(const edm::ParameterSet&);
  ~HypothesisAnalyzer(){};
  
 private:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  edm::InputTag semiLepEvt_;
  edm::InputTag hypoClassKey_;

  TH1F* hadWPt_;
  TH1F* hadWMass_;
  TH1F* hadTopPt_;
  TH1F* hadTopMass_;

  TH1F* lepWPt_;
  TH1F* lepWMass_;
  TH1F* lepTopPt_;
  TH1F* lepTopMass_;
};

#endif
