// File: DataMixingHcalDigiWorkerProd.cc
// Description:  see DataMixingHcalDigiWorkerProd.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataMixingHcalDigiWorkerProd.h"


using namespace std;
namespace edm {
  // Constructor 
  DataMixingHcalDigiWorkerProd::DataMixingHcalDigiWorkerProd(const edm::ParameterSet& ps) : 
    HBHEPileInputTag_(ps.getParameter<edm::InputTag>("HBHEPileInputTag")),
    HOPileInputTag_(ps.getParameter<edm::InputTag>("HOPileInputTag")),
    HFPileInputTag_(ps.getParameter<edm::InputTag>("HFPileInputTag")),
    ZDCPileInputTag_(ps.getParameter<edm::InputTag>("ZDCPileInputTag")),
    theHBHESignalGenerator(HBHEPileInputTag_),
    theHOSignalGenerator(HOPileInputTag_),
    theHFSignalGenerator(HFPileInputTag_),
    theZDCSignalGenerator(ZDCPileInputTag_),
    label_(ps.getParameter<std::string>("Label"))
  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    // Hcal 
    // Signal inputs now handled by HcalDigitizer - gets pSimHits directly

    HBHEDigiCollectionDM_ = ps.getParameter<std::string>("HBHEDigiCollectionDM");
    HODigiCollectionDM_   = ps.getParameter<std::string>("HODigiCollectionDM");
    HFDigiCollectionDM_   = ps.getParameter<std::string>("HFDigiCollectionDM");
    ZDCDigiCollectionDM_  = ps.getParameter<std::string>("ZDCDigiCollectionDM");

    // initialize HcalDigitizer here...

    myHcalDigitizer_ = new HcalDigitizer( ps );

    myHcalDigitizer_->setHBHENoiseSignalGenerator( & theHBHESignalGenerator );
    myHcalDigitizer_->setHFNoiseSignalGenerator( & theHFSignalGenerator );
    myHcalDigitizer_->setHONoiseSignalGenerator( & theHOSignalGenerator );
    myHcalDigitizer_->setZDCNoiseSignalGenerator( & theZDCSignalGenerator );

  }
	       
  // Virtual destructor needed.
  DataMixingHcalDigiWorkerProd::~DataMixingHcalDigiWorkerProd() { 
    delete myHcalDigitizer_;
  }  

  void DataMixingHcalDigiWorkerProd::addHcalSignals(const edm::Event &e,const edm::EventSetup& ES) { 
    
    // nothing to do

  } // end of addHcalSignals

  void DataMixingHcalDigiWorkerProd::addHcalPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,const edm::EventSetup& ES) {
  
    LogDebug("DataMixingHcalDigiWorkerProd") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    theHBHESignalGenerator.initializeEvent(ep, &ES);
    theHOSignalGenerator.initializeEvent(ep, &ES);
    theHFSignalGenerator.initializeEvent(ep, &ES);
    theZDCSignalGenerator.initializeEvent(ep, &ES);

    theHBHESignalGenerator.fill();
    theHOSignalGenerator.fill();
    theHFSignalGenerator.fill();
  }

  void DataMixingHcalDigiWorkerProd::putHcal(edm::Event &e,const edm::EventSetup& ES) {

    // Digitize

    myHcalDigitizer_->initializeEvent( e, ES );
    myHcalDigitizer_->finalizeEvent( e, ES );
  }

} //edm

