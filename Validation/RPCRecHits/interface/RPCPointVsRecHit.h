#ifndef Validation_RPCRecHits_RPCPointVsRecHit_h
#define Validation_RPCRecHits_RPCPointVsRecHit_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "Validation/RPCRecHits/interface/RPCValidHistograms.h"

#include <string>

class RPCPointVsRecHit : public DQMEDAnalyzer
{
public:
  RPCPointVsRecHit(const edm::ParameterSet& pset);
  ~RPCPointVsRecHit() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  edm::EDGetTokenT<RPCRecHitCollection> refHitToken_, recHitToken_;

  std::string subDir_;
  RPCValidHistograms h_;
};

#endif // Validation_RPCRecHits_RPCPointVsRecHit_h
