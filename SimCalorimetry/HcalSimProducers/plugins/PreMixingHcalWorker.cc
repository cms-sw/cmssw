#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSignalGenerator.h"
#include "SimCalorimetry/HcalSimProducers/interface/HcalDigiProducer.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

class PreMixingHcalWorker : public PreMixingWorker {
public:
  PreMixingHcalWorker(const edm::ParameterSet &ps, edm::ProducesCollector, edm::ConsumesCollector &&iC);
  ~PreMixingHcalWorker() override = default;

  PreMixingHcalWorker(const PreMixingHcalWorker &) = delete;
  PreMixingHcalWorker &operator=(const PreMixingHcalWorker &) = delete;

  void beginRun(const edm::Run &run, const edm::EventSetup &ES) override;
  void initializeEvent(const edm::Event &e, const edm::EventSetup &ES) override;
  void addSignals(const edm::Event &e, const edm::EventSetup &ES) override;
  void addPileups(const PileUpEventPrincipal &, const edm::EventSetup &ES) override;
  void put(edm::Event &e, const edm::EventSetup &ES, std::vector<PileupSummaryInfo> const &ps, int bs) override;

private:
  edm::InputTag HBHEPileInputTag_;     // InputTag for Pileup Digis collection
  edm::InputTag HOPileInputTag_;       // InputTag for Pileup Digis collection
  edm::InputTag HFPileInputTag_;       // InputTag for Pileup Digis collection
  edm::InputTag ZDCPileInputTag_;      // InputTag for Pileup Digis collection
  edm::InputTag QIE10PileInputTag_;    // InputTag for Pileup Digis collection
  edm::InputTag QIE11PileInputTag_;    // InputTag for Pileup Digis collection
  std::string HBHEDigiCollectionDM_;   // secondary name to be given to collection
                                       // of digis
  std::string HODigiCollectionDM_;     // secondary name to be given to collection of digis
  std::string HFDigiCollectionDM_;     // secondary name to be given to collection of digis
  std::string ZDCDigiCollectionDM_;    // secondary name to be given to collection of digis
  std::string QIE10DigiCollectionDM_;  // secondary name to be given to
                                       // collection of digis
  std::string QIE11DigiCollectionDM_;  // secondary name to be given to
                                       // collection of digis

  edm::EDGetTokenT<HBHEDigitizerTraits::DigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HODigitizerTraits::DigiCollection> tok_ho_;
  edm::EDGetTokenT<HFDigitizerTraits::DigiCollection> tok_hf_;
  edm::EDGetTokenT<ZDCDigitizerTraits::DigiCollection> tok_zdc_;
  edm::EDGetTokenT<HcalQIE10DigitizerTraits::DigiCollection> tok_qie10_;
  edm::EDGetTokenT<HcalQIE11DigitizerTraits::DigiCollection> tok_qie11_;

  HcalDigiProducer myHcalDigitizer_;
  HBHESignalGenerator theHBHESignalGenerator;
  HOSignalGenerator theHOSignalGenerator;
  HFSignalGenerator theHFSignalGenerator;
  ZDCSignalGenerator theZDCSignalGenerator;
  QIE10SignalGenerator theQIE10SignalGenerator;
  QIE11SignalGenerator theQIE11SignalGenerator;
};

// Constructor
PreMixingHcalWorker::PreMixingHcalWorker(const edm::ParameterSet &ps,
                                         edm::ProducesCollector producesCollector,
                                         edm::ConsumesCollector &&iC)
    : HBHEPileInputTag_(ps.getParameter<edm::InputTag>("HBHEPileInputTag")),
      HOPileInputTag_(ps.getParameter<edm::InputTag>("HOPileInputTag")),
      HFPileInputTag_(ps.getParameter<edm::InputTag>("HFPileInputTag")),
      ZDCPileInputTag_(ps.getParameter<edm::InputTag>("ZDCPileInputTag")),
      QIE10PileInputTag_(ps.getParameter<edm::InputTag>("QIE10PileInputTag")),
      QIE11PileInputTag_(ps.getParameter<edm::InputTag>("QIE11PileInputTag")),
      myHcalDigitizer_(ps, iC) {
  tok_hbhe_ = iC.consumes<HBHEDigitizerTraits::DigiCollection>(HBHEPileInputTag_);
  tok_ho_ = iC.consumes<HODigitizerTraits::DigiCollection>(HOPileInputTag_);
  tok_hf_ = iC.consumes<HFDigitizerTraits::DigiCollection>(HFPileInputTag_);
  tok_zdc_ = iC.consumes<ZDCDigitizerTraits::DigiCollection>(ZDCPileInputTag_);
  tok_qie10_ = iC.consumes<HcalQIE10DigitizerTraits::DigiCollection>(QIE10PileInputTag_);
  tok_qie11_ = iC.consumes<HcalQIE11DigitizerTraits::DigiCollection>(QIE11PileInputTag_);

  theHBHESignalGenerator = HBHESignalGenerator(HBHEPileInputTag_, tok_hbhe_);
  theHOSignalGenerator = HOSignalGenerator(HOPileInputTag_, tok_ho_);
  theHFSignalGenerator = HFSignalGenerator(HFPileInputTag_, tok_hf_);
  theZDCSignalGenerator = ZDCSignalGenerator(ZDCPileInputTag_, tok_zdc_);
  theQIE10SignalGenerator = QIE10SignalGenerator(QIE10PileInputTag_, tok_qie10_);
  theQIE11SignalGenerator = QIE11SignalGenerator(QIE11PileInputTag_, tok_qie11_);

  // Hcal
  // Signal inputs now handled by HcalDigitizer - gets pSimHits directly

  HBHEDigiCollectionDM_ = ps.getParameter<std::string>("HBHEDigiCollectionDM");
  HODigiCollectionDM_ = ps.getParameter<std::string>("HODigiCollectionDM");
  HFDigiCollectionDM_ = ps.getParameter<std::string>("HFDigiCollectionDM");
  ZDCDigiCollectionDM_ = ps.getParameter<std::string>("ZDCDigiCollectionDM");
  QIE10DigiCollectionDM_ = ps.getParameter<std::string>("QIE10DigiCollectionDM");
  QIE11DigiCollectionDM_ = ps.getParameter<std::string>("QIE11DigiCollectionDM");

  producesCollector.produces<HBHEDigiCollection>();
  producesCollector.produces<HODigiCollection>();
  producesCollector.produces<HFDigiCollection>();
  producesCollector.produces<ZDCDigiCollection>();

  producesCollector.produces<QIE10DigiCollection>("HFQIE10DigiCollection");
  producesCollector.produces<QIE11DigiCollection>("HBHEQIE11DigiCollection");

  // initialize HcalDigitizer here...
  myHcalDigitizer_.setHBHENoiseSignalGenerator(&theHBHESignalGenerator);
  myHcalDigitizer_.setHFNoiseSignalGenerator(&theHFSignalGenerator);
  myHcalDigitizer_.setHONoiseSignalGenerator(&theHOSignalGenerator);
  myHcalDigitizer_.setZDCNoiseSignalGenerator(&theZDCSignalGenerator);
  myHcalDigitizer_.setQIE10NoiseSignalGenerator(&theQIE10SignalGenerator);
  myHcalDigitizer_.setQIE11NoiseSignalGenerator(&theQIE11SignalGenerator);
}

void PreMixingHcalWorker::beginRun(const edm::Run &run, const edm::EventSetup &ES) {
  myHcalDigitizer_.beginRun(run, ES);
}

void PreMixingHcalWorker::initializeEvent(const edm::Event &e, const edm::EventSetup &ES) {
  myHcalDigitizer_.initializeEvent(e, ES);
}

void PreMixingHcalWorker::addSignals(const edm::Event &e, const edm::EventSetup &ES) {
  myHcalDigitizer_.accumulate(e, ES);
}

void PreMixingHcalWorker::addPileups(const PileUpEventPrincipal &pep, const edm::EventSetup &ES) {
  const auto &ep = pep.principal();
  LogDebug("PreMixingHcalWorker") << "\n===============> adding pileups from event  " << ep.id()
                                  << " for bunchcrossing " << pep.bunchCrossing();

  theHBHESignalGenerator.initializeEvent(&ep, &ES);
  theHOSignalGenerator.initializeEvent(&ep, &ES);
  theHFSignalGenerator.initializeEvent(&ep, &ES);
  theZDCSignalGenerator.initializeEvent(&ep, &ES);
  theQIE10SignalGenerator.initializeEvent(&ep, &ES);
  theQIE11SignalGenerator.initializeEvent(&ep, &ES);

  // put digis from pileup event into digitizer

  const auto *mcc = pep.moduleCallingContext();
  theHBHESignalGenerator.fill(mcc);
  theHOSignalGenerator.fill(mcc);
  theHFSignalGenerator.fill(mcc);
  theQIE10SignalGenerator.fill(mcc);
  theQIE11SignalGenerator.fill(mcc);
}

void PreMixingHcalWorker::put(edm::Event &e,
                              const edm::EventSetup &ES,
                              std::vector<PileupSummaryInfo> const &ps,
                              int bs) {
  myHcalDigitizer_.finalizeEvent(e, ES);
}

DEFINE_PREMIXING_WORKER(PreMixingHcalWorker);
