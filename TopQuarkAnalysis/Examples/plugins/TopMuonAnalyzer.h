#ifndef TopMuonAnalyzer_h  
#define TopMuonAnalyzer_h

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

class TopMuonAnalyzer : public edm::EDAnalyzer {

 public:
  
  explicit TopMuonAnalyzer(const edm::ParameterSet&);
  ~TopMuonAnalyzer();

 private:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
           	
  edm::InputTag inputElec_;
  edm::InputTag inputMuon_;

  TH1I *Num_Leptons;
  TH1I *Num_Muons;
  TH1F *pt_Muons;
  TH1F *energy_Muons;
  TH1F *eta_Muons;
  TH1F *phi_Muons;
};  

#endif  
