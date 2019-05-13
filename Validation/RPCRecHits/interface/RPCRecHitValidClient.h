#ifndef Validation_RPCRecHits_RPCRecHitValidClient_h
#define Validation_RPCRecHits_RPCRecHitValidClient_h

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <string>

class RPCRecHitValidClient : public DQMEDHarvester {
public:
  RPCRecHitValidClient(const edm::ParameterSet &pset);
  ~RPCRecHitValidClient() override{};

  void dqmEndJob(DQMStore::IBooker &booker, DQMStore::IGetter &getter) override;

private:
  std::string subDir_;
};

#endif  // Validation_RPCRecHits_RPCRecHitValidClient_h
