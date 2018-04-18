#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducerBase.h"

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "PreMixingWorker.h"

namespace edm {
  class PreMixingDigiAccumulatorWorker: public PreMixingWorker {
  public:
    PreMixingDigiAccumulatorWorker(const edm::ParameterSet& ps, edm::ProducerBase& producer, edm::ConsumesCollector&& iC):
      accumulator_(DigiAccumulatorMixModFactory::get()->makeDigiAccumulator(ps.getParameter<edm::ParameterSet>("accumulator"), producer, iC))
    {}
    ~PreMixingDigiAccumulatorWorker() override = default;

    void initializeEvent(const edm::Event &e, const edm::EventSetup& ES) override {
      accumulator_->initializeEvent(e, ES);
    }

    void initializeBunchCrossing(edm::Event const& e, edm::EventSetup const& ES, int bunchCrossing) override {
      accumulator_->initializeBunchCrossing(e, ES, bunchCrossing);
    }
    void finalizeBunchCrossing(edm::Event& e, edm::EventSetup const& ES, int bunchCrossing) override {
      accumulator_->finalizeBunchCrossing(e, ES, bunchCrossing);
    }
    
    void addSignals(const edm::Event &e, const edm::EventSetup& ES) override {
      accumulator_->accumulate(e, ES);
    }
    void addPileups(int bcr, const edm::EventPrincipal& ep, int EventId,
                    const edm::EventSetup& ES, edm::ModuleCallingContext const *mcc) override {
      PileUpEventPrincipal pep(ep, mcc, bcr);
      accumulator_->accumulate(pep, ES, ep.streamID());
    }
    void put(edm::Event &e,const edm::EventSetup& ES, std::vector<PileupSummaryInfo> const& ps, int bs) override {
      accumulator_->finalizeEvent(e, ES);
    }

  private:
    std::unique_ptr<DigiAccumulatorMixMod> accumulator_;
  };
}

#include "PreMixingWorkerFactory.h"
DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, edm::PreMixingDigiAccumulatorWorker, "PreMixingDigiAccumulatorWorker");
