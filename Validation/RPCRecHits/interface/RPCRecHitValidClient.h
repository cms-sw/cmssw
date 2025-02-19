#ifndef Validation_RPCRecHits_RPCRecHitValidClient_h
#define Validaiton_RPCRecHits_RPCRecHitValidClient_h

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

class RPCRecHitValidClient : public edm::EDAnalyzer
{
public:
  RPCRecHitValidClient(const edm::ParameterSet& pset);
  ~RPCRecHitValidClient() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
  void endRun(const edm::Run& run, const edm::EventSetup& eventSetup);
  //void beginJob() {};
  //void endJob() {};

private:
  std::string subDir_;
};

#endif
