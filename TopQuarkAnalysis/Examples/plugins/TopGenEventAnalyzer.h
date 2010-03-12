#ifndef TopGenEventAnalyzer_h  
#define TopGenEventAnalyzer_h

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


class TopGenEventAnalyzer : public edm::EDAnalyzer {

 public:
  
  explicit TopGenEventAnalyzer(const edm::ParameterSet&);
  ~TopGenEventAnalyzer();

 private:

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
           	
  edm::InputTag inputGenEvent_;

  TH1F *nLep_;
  TH1F *topPt_;
  TH1F *topEta_;
  TH1F *topPhi_;
  TH1F *topBarPt_;
  TH1F *topBarEta_;
  TH1F *topBarPhi_;
  TH1F *ttbarPt_;
  TH1F *ttbarEta_;
  TH1F *ttbarPhi_;
};  

#endif  
