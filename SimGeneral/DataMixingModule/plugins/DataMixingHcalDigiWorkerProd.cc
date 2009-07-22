// File: DataMixingHcalDigiWorkerProd.cc
// Description:  see DataMixingHcalDigiWorkerProd.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/Selector.h"
#include "DataMixingHcalDigiWorkerProd.h"


using namespace std;
namespace edm {
  // Constructor 
  DataMixingHcalDigiWorkerProd::DataMixingHcalDigiWorkerProd(const edm::ParameterSet& ps) : 
    HBHEdigiCollectionPile_(ps.getParameter<edm::InputTag>("HBHEdigiCollectionPile")),
    HOdigiCollectionPile_(ps.getParameter<edm::InputTag>("HOdigiCollectionPile")),
    HFdigiCollectionPile_(ps.getParameter<edm::InputTag>("HFdigiCollectionPile")),
    ZDCdigiCollectionPile_(ps.getParameter<edm::InputTag>("ZDCdigiCollectionPile")),
    theHBHESignalGenerator(HBHEdigiCollectionPile_),
    theHOSignalGenerator(HOdigiCollectionPile_),
    theHFSignalGenerator(HFdigiCollectionPile_),
    theZDCSignalGenerator(ZDCdigiCollectionPile_),
    label_(ps.getParameter<std::string>("Label"))
  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // create input selector
    //    if (label_.size()>0){
    //  sel_=new Selector( ModuleLabelSelector(label_));
    // }
    //else {
    //  sel_=new Selector( MatchAllSelector());
    //}

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
    //delete sel_;
    //sel_=0;
  }  

  void DataMixingHcalDigiWorkerProd::addHcalSignals(const edm::Event &e,const edm::EventSetup& ES) { 
    
    // nothing to do

  } // end of addHcalSignals

  void DataMixingHcalDigiWorkerProd::addHcalPileups(const int bcr, Event *e, unsigned int eventNr,const edm::EventSetup& ES) {
  
    LogDebug("DataMixingHcalDigiWorkerProd") <<"\n===============> adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    theHBHESignalGenerator.initializeEvent(e, &ES);
    theHOSignalGenerator.initializeEvent(e, &ES);
    theHFSignalGenerator.initializeEvent(e, &ES);
    theZDCSignalGenerator.initializeEvent(e, &ES);

    theHBHESignalGenerator.fill();
    theHOSignalGenerator.fill();
    theHFSignalGenerator.fill();
  }

  void DataMixingHcalDigiWorkerProd::putHcal(edm::Event &e,const edm::EventSetup& ES) {

    // Digitize

    myHcalDigitizer_->produce( e, ES );
  }

} //edm

