#ifndef TopElecAnalyzer_h  
#define TopElecAnalyzer_h

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

class TopElecAnalyzer : public edm::EDAnalyzer {

 public:
  
  explicit TopElecAnalyzer(const edm::ParameterSet&);
  ~TopElecAnalyzer();
  
 private:
  
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  edm::InputTag elecs_;
  
  TH1I *nrElec_;
  TH1F *ptElec_;
  TH1F *enElec_;
  TH1F *etaElec_;
  TH1F *phiElec_;
  TH1F *dptElec_;
  TH1F *denElec_;
  TH1F *genElec_;
  TH1F *trgElec_;
};  

#endif  
