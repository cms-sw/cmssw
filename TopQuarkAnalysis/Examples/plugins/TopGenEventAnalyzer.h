#ifndef TopGenEventAnalyzer_h  
#define TopGenEventAnalyzer_h

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"


class TopGenEventAnalyzer : public edm::EDAnalyzer {

 public:
  
  explicit TopGenEventAnalyzer(const edm::ParameterSet&);
  ~TopGenEventAnalyzer();

 private:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
           	
  edm::InputTag inputGenEvent_;


  TH1F *semilep;
  TH1F *fulllep;
  TH1F *fullhad;
  TH1F *Num_Leptons;
  TH1F *Summe;
  TH1F *number_of_Daughters;
  TH1F *Daughters;
  TH1F *pdg;
  TH1F *leptype;
  TH1F *Mothers;
  TH1F *Daughters_of_Tau;
  TH1F *Mothers_of_Mu;
};  

#endif  
