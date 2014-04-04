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
    label_(ps.getParameter<std::string>("Label"))
  {  

    theEBSignalGenerator = EBSignalGenerator(EBPileInputTag_,tok_eb_);
    theEESignalGenerator = EESignalGenerator(EBPileInputTag_,tok_ee_);
    //    theESSignalGenerator = ESSignalGenerator(ESPileInputTag_,tok_es_);

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
    //myEcalDigitizer_->setESNoiseSignalGenerator( & theESSignalGenerator );


    std::cout << " IN ECAL constructor " << std::endl;

  }
	       
  // Virtual destructor needed.
  DataMixingEcalDigiWorkerProd::~DataMixingEcalDigiWorkerProd() { 
    delete myEcalDigitizer_;
  }  

  void DataMixingEcalDigiWorkerProd::beginRun(const edm::EventSetup& ES) {

    //    std::cout << " IN DM Ecal BeginRun " << std::endl;

    // myEcalDigitizer_->beginRun(ES); 
  }

  void DataMixingEcalDigiWorkerProd::initializeEvent(const edm::Event &e, const edm::EventSetup& ES) {
    myEcalDigitizer_->initializeEvent(e, ES); 
  }

  void DataMixingEcalDigiWorkerProd::addEcalSignals(const edm::Event &e,const edm::EventSetup& ES) { 
    
    std::cout << " In Ecal Add Signals, something to do " << std::endl;

    myEcalDigitizer_->accumulate(e, ES);

  } // end of addEcalSignals

  void DataMixingEcalDigiWorkerProd::addEcalPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,const edm::EventSetup& ES,
                                                    edm::ModuleCallingContext const* mcc) {
  
    LogDebug("DataMixingEcalDigiWorkerProd") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;


    std::cout << " In Ecal Add Pileup! " << std::endl;

    theEBSignalGenerator.initializeEvent(ep, &ES);
    theEESignalGenerator.initializeEvent(ep, &ES);
    //theESSignalGenerator.initializeEvent(ep, &ES);

    theEBSignalGenerator.fill(mcc);
    theEESignalGenerator.fill(mcc);
    //theESSignalGenerator.fill(mcc);
  }

  void DataMixingEcalDigiWorkerProd::putEcal(edm::Event &e,const edm::EventSetup& ES) {

    // Digitize

    myEcalDigitizer_->initializeEvent( e, ES );
    myEcalDigitizer_->finalizeEvent( e, ES );
  }

} //edm

