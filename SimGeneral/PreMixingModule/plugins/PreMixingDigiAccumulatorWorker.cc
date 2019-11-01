#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

class PreMixingDigiAccumulatorWorker : public PreMixingWorker {
public:
  PreMixingDigiAccumulatorWorker(const edm::ParameterSet& ps,
                                 edm::ProducesCollector producesCollector,
                                 edm::ConsumesCollector&& iC)
      : accumulator_(edm::DigiAccumulatorMixModFactory::get()->makeDigiAccumulator(
            ps.getParameter<edm::ParameterSet>("accumulator"), producesCollector, iC)) {}
  ~PreMixingDigiAccumulatorWorker() override = default;

  void initializeEvent(const edm::Event& e, const edm::EventSetup& ES) override {
    accumulator_->initializeEvent(e, ES);
  }

  void initializeBunchCrossing(edm::Event const& e, edm::EventSetup const& ES, int bunchCrossing) override {
    accumulator_->initializeBunchCrossing(e, ES, bunchCrossing);
  }
  void finalizeBunchCrossing(edm::Event& e, edm::EventSetup const& ES, int bunchCrossing) override {
    accumulator_->finalizeBunchCrossing(e, ES, bunchCrossing);
  }

  void addSignals(const edm::Event& e, const edm::EventSetup& ES) override { accumulator_->accumulate(e, ES); }
  void addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& ES) override {
    accumulator_->accumulate(pep, ES, pep.principal().streamID());
  }
  void put(edm::Event& e, const edm::EventSetup& ES, std::vector<PileupSummaryInfo> const& ps, int bs) override {
    accumulator_->finalizeEvent(e, ES);
  }

private:
  std::unique_ptr<DigiAccumulatorMixMod> accumulator_;
};

DEFINE_PREMIXING_WORKER(PreMixingDigiAccumulatorWorker);
