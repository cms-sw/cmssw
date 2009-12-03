#ifndef TopJetAnalyzer_h  
#define TopJetAnalyzer_h

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

class TopJetAnalyzer : public edm::EDAnalyzer {

 public:
  
  explicit TopJetAnalyzer(const edm::ParameterSet&);
  ~TopJetAnalyzer();
  
 private:
  
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  edm::InputTag input_;
  
  TH1I *Num_Jets;
  TH1F *pt_Jets;
  TH1F *energy_Jets;
  TH1F *eta_Jets;
  TH1F *phi_Jets;

  TH1F *btag_Jets;
};  

#endif  
