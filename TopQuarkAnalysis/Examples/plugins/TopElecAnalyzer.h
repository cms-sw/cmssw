#ifndef TopElecAnalyzer_h  
#define TopElecAnalyzer_h

#include "TH1F.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

class TopElecAnalyzer : public edm::EDAnalyzer {

 public:
  
  explicit TopElecAnalyzer(const edm::ParameterSet&);
  ~TopElecAnalyzer();
  
 private:
  
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  edm::InputTag input_;
  bool verbose_;
  
  TH1F *mult_;
  TH1F *en_;
  TH1F *pt_;
  TH1F *eta_;
  TH1F *phi_;

};  

#endif  
