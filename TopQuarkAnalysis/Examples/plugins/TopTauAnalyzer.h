#ifndef TopTauAnalyzer_h  
#define TopTauAnalyzer_h

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

class TopTauAnalyzer : public edm::EDAnalyzer {

 public:
  
  explicit TopTauAnalyzer(const edm::ParameterSet&);
  ~TopTauAnalyzer();
  
 private:
  
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  edm::InputTag taus_;
  
  TH1I *NrTau_;
  TH1F *ptTau_;
  TH1F *enTau_;
  TH1F *etaTau_;
  TH1F *phiTau_;
};  

#endif  
