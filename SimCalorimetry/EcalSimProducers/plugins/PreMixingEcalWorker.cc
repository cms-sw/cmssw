#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSignalGenerator.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

class PreMixingEcalWorker : public PreMixingWorker {
public:
  PreMixingEcalWorker(const edm::ParameterSet &ps, edm::ProducesCollector, edm::ConsumesCollector &&iC);
  ~PreMixingEcalWorker() override = default;

  PreMixingEcalWorker(const PreMixingEcalWorker &) = delete;
  PreMixingEcalWorker &operator=(const PreMixingEcalWorker &) = delete;

  void beginLuminosityBlock(edm::LuminosityBlock const &lumi, edm::EventSetup const &setup) override;

  void initializeEvent(edm::Event const &e, edm::EventSetup const &ES) override;
  void addSignals(edm::Event const &e, edm::EventSetup const &ES) override;
  void addPileups(PileUpEventPrincipal const &pep, edm::EventSetup const &es) override;
  void put(edm::Event &e, edm::EventSetup const &iSetup, std::vector<PileupSummaryInfo> const &ps, int bs) override;

private:
  edm::InputTag EBPileInputTag_;  // InputTag for Pileup Digis collection
  edm::InputTag EEPileInputTag_;  // InputTag for Pileup Digis collection
  edm::InputTag ESPileInputTag_;  // InputTag for Pileup Digis collection

  std::string EBDigiCollectionDM_;  // secondary name to be given to collection of digis
  std::string EEDigiCollectionDM_;  // secondary name to be given to collection of digis
  std::string ESDigiCollectionDM_;  // secondary name to be given to collection of digis

  edm::EDGetTokenT<EBDigitizerTraits::DigiCollection> tok_eb_;
  edm::EDGetTokenT<EEDigitizerTraits::DigiCollection> tok_ee_;
  edm::EDGetTokenT<ESDigitizerTraits::DigiCollection> tok_es_;

  const double m_EBs25notCont;
  const double m_EEs25notCont;
  const double m_peToABarrel;
  const double m_peToAEndcap;
  const bool m_timeDependent;

  EBSignalGenerator theEBSignalGenerator;
  EESignalGenerator theEESignalGenerator;
  ESSignalGenerator theESSignalGenerator;
  EcalDigiProducer myEcalDigitizer_;
};

// Constructor
PreMixingEcalWorker::PreMixingEcalWorker(const edm::ParameterSet &ps,
                                         edm::ProducesCollector producesCollector,
                                         edm::ConsumesCollector &&iC)
    : EBPileInputTag_(ps.getParameter<edm::InputTag>("EBPileInputTag")),
      EEPileInputTag_(ps.getParameter<edm::InputTag>("EEPileInputTag")),
      ESPileInputTag_(ps.getParameter<edm::InputTag>("ESPileInputTag")),
      tok_eb_(iC.consumes<EBDigitizerTraits::DigiCollection>(EBPileInputTag_)),
      tok_ee_(iC.consumes<EEDigitizerTraits::DigiCollection>(EEPileInputTag_)),
      tok_es_(iC.consumes<ESDigitizerTraits::DigiCollection>(ESPileInputTag_)),
      m_EBs25notCont(ps.getParameter<double>("EBs25notContainment")),
      m_EEs25notCont(ps.getParameter<double>("EEs25notContainment")),
      m_peToABarrel(ps.getParameter<double>("photoelectronsToAnalogBarrel")),
      m_peToAEndcap(ps.getParameter<double>("photoelectronsToAnalogEndcap")),
      m_timeDependent(ps.getParameter<bool>("timeDependent")),
      theEBSignalGenerator(
          EBPileInputTag_, tok_eb_, m_EBs25notCont, m_EEs25notCont, m_peToABarrel, m_peToAEndcap, m_timeDependent),
      theEESignalGenerator(
          EEPileInputTag_, tok_ee_, m_EBs25notCont, m_EEs25notCont, m_peToABarrel, m_peToAEndcap, m_timeDependent),
      theESSignalGenerator(ESPileInputTag_, tok_es_, m_EBs25notCont, m_EEs25notCont, m_peToABarrel, m_peToAEndcap),
      myEcalDigitizer_(ps, iC) {
  EBDigiCollectionDM_ = ps.getParameter<std::string>("EBDigiCollectionDM");
  EEDigiCollectionDM_ = ps.getParameter<std::string>("EEDigiCollectionDM");
  ESDigiCollectionDM_ = ps.getParameter<std::string>("ESDigiCollectionDM");

  producesCollector.produces<EBDigiCollection>(EBDigiCollectionDM_);
  producesCollector.produces<EEDigiCollection>(EEDigiCollectionDM_);
  producesCollector.produces<ESDigiCollection>(ESDigiCollectionDM_);

  myEcalDigitizer_.setEBNoiseSignalGenerator(&theEBSignalGenerator);
  myEcalDigitizer_.setEENoiseSignalGenerator(&theEESignalGenerator);
  myEcalDigitizer_.setESNoiseSignalGenerator(&theESSignalGenerator);
}

void PreMixingEcalWorker::initializeEvent(const edm::Event &e, const edm::EventSetup &ES) {
  myEcalDigitizer_.initializeEvent(e, ES);
}

void PreMixingEcalWorker::addSignals(const edm::Event &e, const edm::EventSetup &ES) {
  myEcalDigitizer_.accumulate(e, ES);
}

void PreMixingEcalWorker::addPileups(const PileUpEventPrincipal &pep, const edm::EventSetup &ES) {
  LogDebug("PreMixingEcalWorker") << "\n===============> adding pileups from event  " << pep.principal().id()
                                  << " for bunchcrossing " << pep.bunchCrossing();

  theEBSignalGenerator.initializeEvent(&pep.principal(), &ES);
  theEESignalGenerator.initializeEvent(&pep.principal(), &ES);
  theESSignalGenerator.initializeEvent(&pep.principal(), &ES);

  // add noise signals using incoming digis
  theEBSignalGenerator.fill(pep.moduleCallingContext());
  theEESignalGenerator.fill(pep.moduleCallingContext());
  theESSignalGenerator.fill(pep.moduleCallingContext());
}

void PreMixingEcalWorker::put(edm::Event &e,
                              const edm::EventSetup &ES,
                              std::vector<PileupSummaryInfo> const &ps,
                              int bs) {
  myEcalDigitizer_.finalizeEvent(e, ES);
}

void PreMixingEcalWorker::beginLuminosityBlock(edm::LuminosityBlock const &lumi, edm::EventSetup const &setup) {
  myEcalDigitizer_.beginLuminosityBlock(lumi, setup);
}

DEFINE_PREMIXING_WORKER(PreMixingEcalWorker);
