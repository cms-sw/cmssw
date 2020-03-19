#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

#include "Phase2TrackerDigitizer.h"
#include "Phase2TrackerDigitizerAlgorithm.h"

class PreMixingPhase2TrackerWorker : public PreMixingWorker {
public:
  PreMixingPhase2TrackerWorker(const edm::ParameterSet& ps, edm::ProducesCollector, edm::ConsumesCollector&& iC);
  ~PreMixingPhase2TrackerWorker() override = default;

  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es) override;

  void initializeEvent(edm::Event const& e, edm::EventSetup const& es) override;
  void addSignals(edm::Event const& e, edm::EventSetup const& es) override;
  void addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& es) override;
  void put(edm::Event& e, edm::EventSetup const& iSetup, std::vector<PileupSummaryInfo> const& ps, int bs) override;

private:
  void accumulate(const edm::DetSetVector<PixelDigi>& digis);

  cms::Phase2TrackerDigitizer digitizer_;

  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> pixelSignalToken_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> trackerSignalToken_;
  edm::InputTag pixelPileupLabel_;
  edm::InputTag trackerPileupLabel_;
  float electronPerAdc_;

  // Maybe map of maps is not that bad for this add once, update once,
  // read once workflow?
  using SignalMap = std::map<uint32_t, std::map<int, float>>;  // (channel, charge)
  SignalMap accumulator_;
};

PreMixingPhase2TrackerWorker::PreMixingPhase2TrackerWorker(const edm::ParameterSet& ps,
                                                           edm::ProducesCollector producesCollector,
                                                           edm::ConsumesCollector&& iC)
    : digitizer_(ps, producesCollector, iC),
      pixelSignalToken_(iC.consumes<edm::DetSetVector<PixelDigi>>(ps.getParameter<edm::InputTag>("pixelLabelSig"))),
      trackerSignalToken_(iC.consumes<edm::DetSetVector<PixelDigi>>(ps.getParameter<edm::InputTag>("trackerLabelSig"))),
      pixelPileupLabel_(ps.getParameter<edm::InputTag>("pixelPileInputTag")),
      trackerPileupLabel_(ps.getParameter<edm::InputTag>("trackerPileInputTag")),
      electronPerAdc_(ps.getParameter<double>("premixStage1ElectronPerAdc")) {}

void PreMixingPhase2TrackerWorker::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& es) {
  digitizer_.beginLuminosityBlock(lumi, es);
}

void PreMixingPhase2TrackerWorker::initializeEvent(edm::Event const& e, edm::EventSetup const& es) {
  digitizer_.initializeEvent(e, es);
}

void PreMixingPhase2TrackerWorker::addSignals(edm::Event const& e, edm::EventSetup const& es) {
  edm::Handle<edm::DetSetVector<PixelDigi>> hdigis;
  e.getByToken(pixelSignalToken_, hdigis);
  accumulate(*hdigis);

  e.getByToken(trackerSignalToken_, hdigis);
  accumulate(*hdigis);
}

void PreMixingPhase2TrackerWorker::addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& es) {
  edm::Handle<edm::DetSetVector<PixelDigi>> hdigis;
  pep.getByLabel(pixelPileupLabel_, hdigis);
  accumulate(*hdigis);

  pep.getByLabel(trackerPileupLabel_, hdigis);
  accumulate(*hdigis);
}

void PreMixingPhase2TrackerWorker::accumulate(const edm::DetSetVector<PixelDigi>& digis) {
  for (const auto& detset : digis) {
    auto& accDet = accumulator_[detset.detId()];
    for (const auto& digi : detset) {
      // note: according to C++ standard operator[] does
      // value-initializiation, which for float means initial value of 0
      auto& acc = accDet[digi.channel()];
      acc += digi.adc() * electronPerAdc_;
    }
  }
}

void PreMixingPhase2TrackerWorker::put(edm::Event& e,
                                       edm::EventSetup const& iSetup,
                                       std::vector<PileupSummaryInfo> const& ps,
                                       int bs) {
  digitizer_.loadAccumulator(accumulator_);
  digitizer_.finalizeEvent(e, iSetup);
  decltype(accumulator_){}.swap(accumulator_);  // release memory
}

DEFINE_PREMIXING_WORKER(PreMixingPhase2TrackerWorker);
