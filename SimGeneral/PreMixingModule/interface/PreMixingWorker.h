#ifndef SimGeneral_PreMixingModule_PreMixingWorker_h
#define SimGeneral_PreMixingModule_PreMixingWorker_h

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <vector>

namespace edm {
  class ModuleCallingContext;
}
class PileupSummaryInfo;
class PileUpEventPrincipal;

class PreMixingWorker {
public:
  PreMixingWorker() = default;
  virtual ~PreMixingWorker() = default;

  virtual void beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}
  virtual void endRun() {}
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const& iSetup) {}
  virtual void initializeBunchCrossing(edm::Event const& iEvent, edm::EventSetup const& iSetup, int bunchCrossing) {}
  virtual void finalizeBunchCrossing(edm::Event& iEvent, edm::EventSetup const& iSetup, int bunchCrossing) {}

  virtual void initializeEvent(edm::Event const& iEvent, edm::EventSetup const& iSetup) = 0;
  virtual void addSignals(edm::Event const& iEvent, edm::EventSetup const& iSetup) = 0;
  virtual void addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) = 0;
  virtual void put(edm::Event& iEvent, edm::EventSetup const& iSetup, std::vector<PileupSummaryInfo> const& ps, int bunchSpacing) = 0;
};

#endif
