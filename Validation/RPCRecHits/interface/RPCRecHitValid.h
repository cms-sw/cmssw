#ifndef Validation_RPCRecHits_RPCRecHitValid_h
#define Validaiton_RPCRecHits_RPCRecHitValid_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Validation/RPCRecHits/interface/RPCValidHistograms.h"

#include <string>

class RPCRecHitValid : public edm::EDAnalyzer
{
public:
  RPCRecHitValid(const edm::ParameterSet& pset);
  ~RPCRecHitValid();

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);
  void beginJob();
  void endJob();

private:
  edm::InputTag simHitLabel_, recHitLabel_;

  DQMStore* dbe_;
  std::string rootFileName_;
  bool isStandAloneMode_;

  RPCValidHistograms h_;
};

#endif
