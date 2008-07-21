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

  edm::InputTag input_;
  
  TH1I *Num_Elecs;
  TH1F *pt_Elecs;
  TH1F *energy_Elecs;
  TH1F *eta_Elecs;
  TH1F *phi_Elecs;
};  

#endif  
