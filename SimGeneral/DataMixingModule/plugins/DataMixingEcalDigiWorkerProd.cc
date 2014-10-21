// File: DataMixingEcalDigiWorkerProd.cc
// Description:  see DataMixingEcalDigiWorkerProd.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataMixingEcalDigiWorkerProd.h"


using namespace std;
namespace edm {
  // Constructor 
  DataMixingEcalDigiWorkerProd::DataMixingEcalDigiWorkerProd(const edm::ParameterSet& ps, edm::ConsumesCollector& iC) : 
    EBPileInputTag_(ps.getParameter<edm::InputTag>("EBPileInputTag")),
    EEPileInputTag_(ps.getParameter<edm::InputTag>("EEPileInputTag")),
    ESPileInputTag_(ps.getParameter<edm::InputTag>("ESPileInputTag")),
    m_EBs25notCont(ps.getParameter<double>("EBs25notContainment") ) ,
    m_EEs25notCont(ps.getParameter<double>("EEs25notContainment") ) ,
    m_peToABarrel(ps.getParameter<double>("photoelectronsToAnalogBarrel") ) ,
    m_peToAEndcap(ps.getParameter<double>("photoelectronsToAnalogEndcap") ) ,
    label_(ps.getParameter<std::string>("Label"))
  {  

    theEBSignalGenerator = EBSignalGenerator(EBPileInputTag_,tok_eb_, m_EBs25notCont, m_EEs25notCont, m_peToABarrel, m_peToAEndcap);
    theEESignalGenerator = EESignalGenerator(EEPileInputTag_,tok_ee_, m_EBs25notCont, m_EEs25notCont, m_peToABarrel, m_peToAEndcap);
    theESSignalGenerator = ESSignalGenerator(ESPileInputTag_,tok_es_, m_EBs25notCont, m_EEs25notCont, m_peToABarrel, m_peToAEndcap);

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    // Ecal 
    // Signal inputs now handled by EcalDigitizer - gets pSimHits directly

    EBDigiCollectionDM_   = ps.getParameter<std::string>("EBDigiCollectionDM");
    EEDigiCollectionDM_   = ps.getParameter<std::string>("EEDigiCollectionDM");
    ESDigiCollectionDM_   = ps.getParameter<std::string>("ESDigiCollectionDM");

    // initialize EcalDigitizer here...

    myEcalDigitizer_ = new EcalDigiProducer( ps , iC);

    myEcalDigitizer_->setEBNoiseSignalGenerator( & theEBSignalGenerator );
    myEcalDigitizer_->setEENoiseSignalGenerator( & theEESignalGenerator );
    myEcalDigitizer_->setESNoiseSignalGenerator( & theESSignalGenerator );

  }
	       
  // Virtual destructor needed.
  DataMixingEcalDigiWorkerProd::~DataMixingEcalDigiWorkerProd() { 
    delete myEcalDigitizer_;
  }  

  void DataMixingEcalDigiWorkerProd::beginRun(const edm::EventSetup& ES) {

    // myEcalDigitizer_->beginRun(ES); 
  }

  void DataMixingEcalDigiWorkerProd::initializeEvent(const edm::Event &e, const edm::EventSetup& ES) {
    myEcalDigitizer_->initializeEvent(e, ES); 
  }

  void DataMixingEcalDigiWorkerProd::addEcalSignals(const edm::Event &e,const edm::EventSetup& ES) { 
    
    myEcalDigitizer_->accumulate(e, ES);  // puts SimHits into Ecal digitizer

  } // end of addEcalSignals

  void DataMixingEcalDigiWorkerProd::addEcalPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,const edm::EventSetup& ES,
                                                    edm::ModuleCallingContext const* mcc) {
  
    LogDebug("DataMixingEcalDigiWorkerProd") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    theEBSignalGenerator.initializeEvent(ep, &ES);
    theEESignalGenerator.initializeEvent(ep, &ES);
    theESSignalGenerator.initializeEvent(ep, &ES);

    // add noise signals using incoming digis

    theEBSignalGenerator.fill(mcc);
    theEESignalGenerator.fill(mcc);
    theESSignalGenerator.fill(mcc);
  }

  void DataMixingEcalDigiWorkerProd::putEcal(edm::Event &e,const edm::EventSetup& ES) {

    // Digitize

    myEcalDigitizer_->finalizeEvent( e, ES );

  }

} //edm

