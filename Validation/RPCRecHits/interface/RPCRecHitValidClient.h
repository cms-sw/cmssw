#ifndef Validation_RPCRecHits_RPCRecHitValidClient_h
#define Validaiton_RPCRecHits_RPCRecHitValidClient_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>

class RPCRecHitValidClient : public DQMEDHarvester
{
public:
  RPCRecHitValidClient(const edm::ParameterSet& pset);
  ~RPCRecHitValidClient() {};

  void dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter);

private:
  std::string subDir_;
};

#endif
