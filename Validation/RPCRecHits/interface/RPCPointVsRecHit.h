#ifndef Validaiton_RPCRecHits_RPCPointVsRecHit_h
#define Validation_RPCRecHits_RPCPointVsRecHit_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>

class RPCPointVsRecHit : public edm::EDAnalyzer
{
public:
  RPCPointVsRecHit(const edm::ParameterSet& pset);
  ~RPCPointVsRecHit();

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);
  void beginJob();
  void endJob();

private:
  edm::InputTag refHitLabel_, recHitLabel_;

  DQMStore* dbe_;
  std::string rootFileName_;
  bool isStandAloneMode_;

  typedef MonitorElement* MEP;
  
  std::map<int, MEP> h_;
};

#endif
