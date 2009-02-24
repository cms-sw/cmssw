// File: DataMixingHcalDigiWorkerProd.cc
// Description:  see DataMixingHcalDigiWorkerProd.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
// calibration headers, for future reference 
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"   
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h" 
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"  
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"   
//
//
#include "DataMixingHcalDigiWorkerProd.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingHcalDigiWorkerProd::DataMixingHcalDigiWorkerProd() {sel_=0;} 

  // Constructor 
  DataMixingHcalDigiWorkerProd::DataMixingHcalDigiWorkerProd(const edm::ParameterSet& ps) : 
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

    //HBHEdigiCollectionSig_  = ps.getParameter<edm::InputTag>("HBHEdigiCollectionSig");
    //HOdigiCollectionSig_    = ps.getParameter<edm::InputTag>("HOdigiCollectionSig");
    //HFdigiCollectionSig_    = ps.getParameter<edm::InputTag>("HFdigiCollectionSig");
    //ZDCdigiCollectionSig_   = ps.getParameter<edm::InputTag>("ZDCdigiCollectionSig");
    HBHEdigiCollectionPile_  = ps.getParameter<edm::InputTag>("HBHEdigiCollectionPile");
    HOdigiCollectionPile_    = ps.getParameter<edm::InputTag>("HOdigiCollectionPile");
    HFdigiCollectionPile_    = ps.getParameter<edm::InputTag>("HFdigiCollectionPile");
    ZDCdigiCollectionPile_   = ps.getParameter<edm::InputTag>("ZDCdigiCollectionPile");

    HBHEDigiCollectionDM_ = ps.getParameter<std::string>("HBHEDigiCollectionDM");
    HODigiCollectionDM_   = ps.getParameter<std::string>("HODigiCollectionDM");
    HFDigiCollectionDM_   = ps.getParameter<std::string>("HFDigiCollectionDM");
    ZDCDigiCollectionDM_  = ps.getParameter<std::string>("ZDCDigiCollectionDM");

    // initialize HcalDigitizer here...

    myHcalDigitizer_ = new HcalDigitizer( ps );

    myHcalDigitizer_->setHBHENoiseSignalGenerator( & myHBHENoise_ );
    myHcalDigitizer_->setHFNoiseSignalGenerator( & myHFNoise_ );
    myHcalDigitizer_->setHONoiseSignalGenerator( & myHONoise_ );
    myHcalDigitizer_->setZDCNoiseSignalGenerator( & myZDCNoise_ );

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

    // get conditions                                                                                                             
    edm::ESHandle<HcalDbService> conditions;
    ES.get<HcalDbRecord>().get(conditions);

    const HcalQIEShape* shape = conditions->getHcalShape (); // this one is generic         

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // HBHE first

   Handle< HBHEDigiCollection > pHBHEDigis;
   const HBHEDigiCollection*  HBHEDigis = 0;

   if( e->getByLabel( HBHEdigiCollectionPile_.label(), pHBHEDigis) ) {
     HBHEDigis = pHBHEDigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalDigiWorkerProd") << "total # HEHB digis: " << HBHEDigis->size();
#endif
   } 
   
 
   if (HBHEDigis)
     {
       // loop over digis, adding these to the existing maps
       for(HBHEDigiCollection::const_iterator it  = HBHEDigis->begin();
	   it != HBHEDigis->end(); ++it) {

         // calibration, for future reference:  (same block for all Hcal types)                                
         HcalDetId cell = it->id();
         //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
         const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
         HcalCoderDb coder (*channelCoder, *shape);

         CaloSamples tool;
         coder.adc2fC((*it),tool);

	 HBHEDigiStore_.push_back( tool );
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalDigiWorkerProd") << "processed HBHEDigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
     }
    // HO Next

   Handle< HODigiCollection > pHODigis;
   const HODigiCollection*  HODigis = 0;

   if( e->getByLabel( HOdigiCollectionPile_.label(), pHODigis) ) {
     HODigis = pHODigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalDigiWorkerProd") << "total # HO digis: " << HODigis->size();
#endif
   } 
   
 
   if (HODigis)
     {
       // loop over digis, adding these to the existing maps
       for(HODigiCollection::const_iterator it  = HODigis->begin();
	   it != HODigis->end(); ++it) {

         // calibration, for future reference:  (same block for all Hcal types)                                
         HcalDetId cell = it->id();
         //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
         const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
         HcalCoderDb coder (*channelCoder, *shape);

         CaloSamples tool;
         coder.adc2fC((*it),tool);

	 HODigiStore_.push_back( tool );
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalDigiWorkerProd") << "processed HODigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
     }

    // HF Next

   Handle< HFDigiCollection > pHFDigis;
   const HFDigiCollection*  HFDigis = 0;

   if( e->getByLabel( HFdigiCollectionPile_.label(), pHFDigis) ) {
     HFDigis = pHFDigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalDigiWorkerProd") << "total # HF digis: " << HFDigis->size();
#endif
   } 
   
 
   if (HFDigis)
     {
       // loop over digis, adding these to the existing maps
       for(HFDigiCollection::const_iterator it  = HFDigis->begin();
	   it != HFDigis->end(); ++it) {

         // calibration, for future reference:  (same block for all Hcal types)                                
         HcalDetId cell = it->id();
         //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
         const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
         HcalCoderDb coder (*channelCoder, *shape);

         CaloSamples tool;
         coder.adc2fC((*it),tool);

	 HFDigiStore_.push_back( tool );
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalDigiWorkerProd") << "processed HFDigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
     }

    // ZDC Next

   Handle< ZDCDigiCollection > pZDCDigis;
   const ZDCDigiCollection*  ZDCDigis = 0;

   if( e->getByLabel( ZDCdigiCollectionPile_.label(), pZDCDigis) ) {
     ZDCDigis = pZDCDigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalDigiWorkerProd") << "total # ZDC digis: " << ZDCDigis->size();
#endif
   } 
   
 
   if (ZDCDigis)
     {
       // loop over digis, adding these to the existing maps
       for(ZDCDigiCollection::const_iterator it  = ZDCDigis->begin();
	   it != ZDCDigis->end(); ++it) {

         // calibration, for future reference:  (same block for all Hcal types)                                
         HcalDetId cell = it->id();
         //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
         const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
         HcalCoderDb coder (*channelCoder, *shape);

         CaloSamples tool;
         coder.adc2fC((*it),tool);

	 ZDCDigiStore_.push_back( tool );
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalDigiWorkerProd") << "processed ZDCDigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
     }

  }
 
  void DataMixingHcalDigiWorkerProd::putHcal(edm::Event &e,const edm::EventSetup& ES) {

    // set the noise signals

    myHBHENoise_.setNoiseSignals( HBHEDigiStore_ );
    myHONoise_.setNoiseSignals( HODigiStore_ );
    myHFNoise_.setNoiseSignals( HFDigiStore_ );
    myZDCNoise_.setNoiseSignals( ZDCDigiStore_ );
  
    // Digitize

    myHcalDigitizer_->produce( e, ES );

    // clear local storage after this event
    HBHEDigiStore_.clear();
    HODigiStore_.clear();
    HFDigiStore_.clear();
    ZDCDigiStore_.clear();

    //myHBHENoise_.doClear();
    //myHFNoise_.doClear();
    //myHONoise_.doClear();
    //myZDCNoise_.doClear();


  }

} //edm
