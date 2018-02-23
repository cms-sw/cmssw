#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSignalGenerator.h"

#include "PreMixingWorker.h"

namespace edm {
  class PreMixingEcalWorker: public PreMixingWorker {
  public:
    PreMixingEcalWorker(const edm::ParameterSet& ps, edm::ProducerBase& producer, edm::ConsumesCollector&& iC);
    ~PreMixingEcalWorker() override = default;

    PreMixingEcalWorker(const PreMixingEcalWorker&) = delete;
    PreMixingEcalWorker& operator=(const PreMixingEcalWorker&) = delete;

    void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) override;

    void initializeEvent(edm::Event const& e, edm::EventSetup const& ES) override;
    void addSignals(edm::Event const& e, edm::EventSetup const& ES) override;
    void addPileups(int bcr, edm::EventPrincipal const& ep, int EventId, edm::EventSetup const& es, ModuleCallingContext const*) override;
    void put(edm::Event &e, edm::EventSetup const& iSetup, std::vector<PileupSummaryInfo> const& ps, int bs) override;

  private:
    edm::InputTag EBPileInputTag_; // InputTag for Pileup Digis collection
    edm::InputTag EEPileInputTag_  ; // InputTag for Pileup Digis collection
    edm::InputTag ESPileInputTag_  ; // InputTag for Pileup Digis collection

    std::string EBDigiCollectionDM_; // secondary name to be given to collection of digis
    std::string EEDigiCollectionDM_  ; // secondary name to be given to collection of digis
    std::string ESDigiCollectionDM_  ; // secondary name to be given to collection of digis

    edm::EDGetTokenT<EBDigitizerTraits::DigiCollection> tok_eb_;
    edm::EDGetTokenT<EEDigitizerTraits::DigiCollection> tok_ee_;
    edm::EDGetTokenT<ESDigitizerTraits::DigiCollection> tok_es_;

    const double m_EBs25notCont;
    const double m_EEs25notCont;
    const double m_peToABarrel;
    const double m_peToAEndcap;

    EBSignalGenerator theEBSignalGenerator;
    EESignalGenerator theEESignalGenerator;
    ESSignalGenerator theESSignalGenerator;
    EcalDigiProducer myEcalDigitizer_;
  };


  // Constructor
  PreMixingEcalWorker::PreMixingEcalWorker(const edm::ParameterSet& ps, edm::ProducerBase& producer, edm::ConsumesCollector&& iC) : 
    EBPileInputTag_(ps.getParameter<edm::InputTag>("EBPileInputTag")),
    EEPileInputTag_(ps.getParameter<edm::InputTag>("EEPileInputTag")),
    ESPileInputTag_(ps.getParameter<edm::InputTag>("ESPileInputTag")),
    tok_eb_(iC.consumes<EBDigitizerTraits::DigiCollection>(EBPileInputTag_)),
    tok_ee_(iC.consumes<EEDigitizerTraits::DigiCollection>(EEPileInputTag_)),
    tok_es_(iC.consumes<ESDigitizerTraits::DigiCollection>(ESPileInputTag_)),
    m_EBs25notCont(ps.getParameter<double>("EBs25notContainment") ) ,
    m_EEs25notCont(ps.getParameter<double>("EEs25notContainment") ) ,
    m_peToABarrel(ps.getParameter<double>("photoelectronsToAnalogBarrel") ) ,
    m_peToAEndcap(ps.getParameter<double>("photoelectronsToAnalogEndcap") ),
    theEBSignalGenerator(EBPileInputTag_,tok_eb_, m_EBs25notCont, m_EEs25notCont, m_peToABarrel, m_peToAEndcap),
    theEESignalGenerator(EEPileInputTag_,tok_ee_, m_EBs25notCont, m_EEs25notCont, m_peToABarrel, m_peToAEndcap),
    theESSignalGenerator(ESPileInputTag_,tok_es_, m_EBs25notCont, m_EEs25notCont, m_peToABarrel, m_peToAEndcap),
    myEcalDigitizer_(ps, iC)
  {
    EBDigiCollectionDM_   = ps.getParameter<std::string>("EBDigiCollectionDM");
    EEDigiCollectionDM_   = ps.getParameter<std::string>("EEDigiCollectionDM");
    ESDigiCollectionDM_   = ps.getParameter<std::string>("ESDigiCollectionDM");

    producer.produces< EBDigiCollection >(EBDigiCollectionDM_);
    producer.produces< EEDigiCollection >(EEDigiCollectionDM_);
    producer.produces< ESDigiCollection >(ESDigiCollectionDM_);

    myEcalDigitizer_.setEBNoiseSignalGenerator( & theEBSignalGenerator );
    myEcalDigitizer_.setEENoiseSignalGenerator( & theEESignalGenerator );
    myEcalDigitizer_.setESNoiseSignalGenerator( & theESSignalGenerator );
  }

  void PreMixingEcalWorker::initializeEvent(const edm::Event &e, const edm::EventSetup& ES) {
    myEcalDigitizer_.initializeEvent(e, ES);
  }

  void PreMixingEcalWorker::addSignals(const edm::Event &e,const edm::EventSetup& ES) { 
    myEcalDigitizer_.accumulate(e, ES);
  }

  void PreMixingEcalWorker::addPileups(int bcr, const EventPrincipal& ep, int eventNr, const edm::EventSetup& ES,
                                       edm::ModuleCallingContext const* mcc) {

    LogDebug("PreMixingEcalWorker") <<"\n===============> adding pileups from event  "<<ep.id()<<" for bunchcrossing "<<bcr;

    theEBSignalGenerator.initializeEvent(&ep, &ES);
    theEESignalGenerator.initializeEvent(&ep, &ES);
    theESSignalGenerator.initializeEvent(&ep, &ES);

    // add noise signals using incoming digis
    theEBSignalGenerator.fill(mcc);
    theEESignalGenerator.fill(mcc);
    theESSignalGenerator.fill(mcc);
  }

  void PreMixingEcalWorker::put(edm::Event &e,const edm::EventSetup& ES, std::vector<PileupSummaryInfo> const& ps, int bs) {
    myEcalDigitizer_.finalizeEvent( e, ES );
  }

  void PreMixingEcalWorker::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
    myEcalDigitizer_.beginLuminosityBlock(lumi,setup);
  }
} //edm

#include "PreMixingWorkerFactory.h"
DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, edm::PreMixingEcalWorker, "PreMixingEcalWorker");
