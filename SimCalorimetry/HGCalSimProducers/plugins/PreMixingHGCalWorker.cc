#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HGCDigi/interface/PHGCSimAccumulator.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizer.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

class PreMixingHGCalWorker : public PreMixingWorker {
public:
  PreMixingHGCalWorker(const edm::ParameterSet& ps, edm::ProducesCollector, edm::ConsumesCollector&& iC);
  ~PreMixingHGCalWorker() override = default;

  PreMixingHGCalWorker(const PreMixingHGCalWorker&) = delete;
  PreMixingHGCalWorker& operator=(const PreMixingHGCalWorker&) = delete;

  void beginRun(const edm::Run& run, const edm::EventSetup& ES) override;
  void endRun() override;
  void initializeEvent(const edm::Event& e, const edm::EventSetup& ES) override {}
  void addSignals(const edm::Event& e, const edm::EventSetup& ES) override;
  void addPileups(const PileUpEventPrincipal&, const edm::EventSetup& ES) override;
  void put(edm::Event& e, const edm::EventSetup& ES, std::vector<PileupSummaryInfo> const& ps, int bs) override;

private:
  edm::EDGetTokenT<PHGCSimAccumulator> signalToken_;

  edm::InputTag pileInputTag_;

  HGCDigitizer digitizer_;
};

PreMixingHGCalWorker::PreMixingHGCalWorker(const edm::ParameterSet& ps,
                                           edm::ProducesCollector producesCollector,
                                           edm::ConsumesCollector&& iC)
    : signalToken_(iC.consumes<PHGCSimAccumulator>(ps.getParameter<edm::InputTag>("digiTagSig"))),
      pileInputTag_(ps.getParameter<edm::InputTag>("pileInputTag")),
      digitizer_(ps, iC) {
  producesCollector.produces<HGCalDigiCollection>(digitizer_.digiCollection());
}

void PreMixingHGCalWorker::beginRun(const edm::Run& run, const edm::EventSetup& ES) { digitizer_.beginRun(ES); }

void PreMixingHGCalWorker::endRun() { digitizer_.endRun(); }

void PreMixingHGCalWorker::addSignals(const edm::Event& e, const edm::EventSetup& ES) {
  edm::Handle<PHGCSimAccumulator> handle;
  e.getByToken(signalToken_, handle);
  digitizer_.accumulate(*handle);
}

void PreMixingHGCalWorker::addPileups(const PileUpEventPrincipal& pep, const edm::EventSetup& ES) {
  edm::Handle<PHGCSimAccumulator> handle;
  pep.getByLabel(pileInputTag_, handle);
  digitizer_.accumulate(*handle);
}

void PreMixingHGCalWorker::put(edm::Event& e,
                               const edm::EventSetup& ES,
                               std::vector<PileupSummaryInfo> const& ps,
                               int bs) {
  edm::Service<edm::RandomNumberGenerator> rng;
  digitizer_.finalizeEvent(e, ES, &rng->getEngine(e.streamID()));
}

DEFINE_PREMIXING_WORKER(PreMixingHGCalWorker);
