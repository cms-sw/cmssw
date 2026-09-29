#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSignalGeneratorPh2.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer_Ph2.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

class PreMixingEcalPh2Worker : public PreMixingWorker {
public:
  PreMixingEcalPh2Worker(const edm::ParameterSet &ps, edm::ProducesCollector, edm::ConsumesCollector &&iC);
  ~PreMixingEcalPh2Worker() override = default;

  PreMixingEcalPh2Worker(const PreMixingEcalPh2Worker &) = delete;
  PreMixingEcalPh2Worker &operator=(const PreMixingEcalPh2Worker &) = delete;

  void beginLuminosityBlock(edm::LuminosityBlock const &lumi, edm::EventSetup const &setup) override;

  void initializeEvent(edm::Event const &e, edm::EventSetup const &ES) override;
  void addSignals(edm::Event const &e, edm::EventSetup const &ES) override;
  void addPileups(PileUpEventPrincipal const &pep, edm::EventSetup const &es) override;
  void put(edm::Event &e, edm::EventSetup const &iSetup, std::vector<PileupSummaryInfo> const &ps, int bs) override;

private:
  EBSignalGeneratorPh2 theEBSignalGenerator_;
  EcalDigiProducer_Ph2 myEcalDigitizer_;
};

// Constructor
PreMixingEcalPh2Worker::PreMixingEcalPh2Worker(const edm::ParameterSet &ps,
                                               edm::ProducesCollector producesCollector,
                                               edm::ConsumesCollector &&iC)
    : theEBSignalGenerator_(iC,
                            ps.getParameter<edm::InputTag>("EBPileInputTag"),
                            ps.getParameter<double>("EBs25notContainment"),
                            ps.getParameter<double>("photoelectronsToAnalogBarrel"),
                            ps.getParameter<bool>("timeDependent")),
      myEcalDigitizer_(ps, iC) {
  producesCollector.produces<EBDigiCollectionPh2>(ps.getParameter<std::string>("EBDigiCollectionDM"));

  myEcalDigitizer_.setEBNoiseSignalGenerator(&theEBSignalGenerator_);
}

void PreMixingEcalPh2Worker::initializeEvent(const edm::Event &e, const edm::EventSetup &ES) {
  myEcalDigitizer_.initializeEvent(e, ES);
}

void PreMixingEcalPh2Worker::addSignals(const edm::Event &e, const edm::EventSetup &ES) {
  myEcalDigitizer_.accumulate(e, ES);
}

void PreMixingEcalPh2Worker::addPileups(const PileUpEventPrincipal &pep, const edm::EventSetup &ES) {
  LogDebug("PreMixingEcalPh2Worker") << "\n===============> adding pileups from event  " << pep.principal().id()
                                     << " for bunchcrossing " << pep.bunchCrossing();

  theEBSignalGenerator_.initializeEvent(&pep.principal(), &ES);

  // add noise signals using incoming digis
  theEBSignalGenerator_.fill(pep.moduleCallingContext());
}

void PreMixingEcalPh2Worker::put(edm::Event &e,
                                 const edm::EventSetup &ES,
                                 std::vector<PileupSummaryInfo> const &ps,
                                 int bs) {
  myEcalDigitizer_.finalizeEvent(e, ES);
}

void PreMixingEcalPh2Worker::beginLuminosityBlock(edm::LuminosityBlock const &lumi, edm::EventSetup const &setup) {
  myEcalDigitizer_.beginLuminosityBlock(lumi, setup);
}

DEFINE_PREMIXING_WORKER(PreMixingEcalPh2Worker);
