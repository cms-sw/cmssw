// File: DataMixingEMDigiWorker.cc
// Description:  see DataMixingEMDigiWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
//
//
#include "DataMixingEMDigiWorker.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingEMDigiWorker::DataMixingEMDigiWorker() { sel_=0;}

  // Constructor 
  DataMixingEMDigiWorker::DataMixingEMDigiWorker(const edm::ParameterSet& ps) : 
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // create input selector
    if (label_.size()>0){
      sel_=new Selector( ModuleLabelSelector(label_));
    }
    else {
      sel_=new Selector( MatchAllSelector());
    }

    // declare the products to produce, retrieve

    EBProducerSig_ = ps.getParameter<edm::InputTag>("EBdigiProducerSig");
    EEProducerSig_ = ps.getParameter<edm::InputTag>("EEdigiProducerSig");
    ESProducerSig_ = ps.getParameter<edm::InputTag>("ESdigiProducerSig");
    EBProducerPile_ = ps.getParameter<edm::InputTag>("EBdigiProducerPile");
    EEProducerPile_ = ps.getParameter<edm::InputTag>("EEdigiProducerPile");
    ESProducerPile_ = ps.getParameter<edm::InputTag>("ESdigiProducerPile");
    EBdigiCollectionSig_ = ps.getParameter<edm::InputTag>("EBdigiCollectionSig");
    EEdigiCollectionSig_ = ps.getParameter<edm::InputTag>("EEdigiCollectionSig");
    ESdigiCollectionSig_ = ps.getParameter<edm::InputTag>("ESdigiCollectionSig");
    EBdigiCollectionPile_ = ps.getParameter<edm::InputTag>("EBdigiCollectionPile");
    EEdigiCollectionPile_ = ps.getParameter<edm::InputTag>("EEdigiCollectionPile");
    ESdigiCollectionPile_ = ps.getParameter<edm::InputTag>("ESdigiCollectionPile");

    EBDigiCollectionDM_        = ps.getParameter<std::string>("EBDigiCollectionDM");
    EEDigiCollectionDM_        = ps.getParameter<std::string>("EEDigiCollectionDM");
    ESDigiCollectionDM_        = ps.getParameter<std::string>("ESDigiCollectionDM");
   //   nMaxPrintout_            = ps.getUntrackedParameter<int>("nMaxPrintout",10);

   //EBalgo_ = new EcalDigiSimpleAlgo();
   //EEalgo_ = new EcalDigiSimpleAlgo();

   // don't think I can "produce" in a sub-class...

   //produces< EBDigiCollection >(EBDigiCollectionDM_);
   //produces< EEDigiCollection >(EEDigiCollectionDM_);

  }
	       

  // Virtual destructor needed.
  DataMixingEMDigiWorker::~DataMixingEMDigiWorker() { 
    delete sel_;
    sel_=0;
  }  

  void DataMixingEMDigiWorker::addEMSignals(const edm::Event &e,const edm::EventSetup& ES) { 
    // fill in maps of hits

    LogInfo("DataMixingEMDigiWorker")<<"===============> adding MC signals for "<<e.id();

    // EB first

   Handle< EBDigiCollection > pEBDigis;

   const EBDigiCollection*  EBDigis = 0;

   if(e.getByLabel(EBProducerSig_.label(),EBdigiCollectionSig_.label(), pEBDigis) ){
     EBDigis = pEBDigis.product(); // get a ptr to the product
     LogDebug("DataMixingEMDigiWorker") << "total # EB digis: " << EBDigis->size();
   }
   else { cout << "NO EBDigis! " << EBProducerSig_.label() << " " << EBdigiCollectionSig_.label() << endl;}
 
   if (EBDigis)
     {
       // loop over digis, storing them in a map so we can add pileup later


       for(EBDigiCollection::const_iterator it  = EBDigis->begin();	
	   it != EBDigis->end(); ++it) {

	 EBDigiStorage_.insert(EBDigiMap::value_type( ( it->id() ), *it ));
#ifdef DEBUG	 
         LogDebug("DataMixingEMDigiWorker") << "processed EBDigi with rawId: "
				      << it->id().rawId() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
     }

   // EE next

   Handle< EEDigiCollection > pEEDigis;

   const EEDigiCollection*  EEDigis = 0;

   
   if(e.getByLabel(EEProducerSig_.label(),EEdigiCollectionSig_.label(), pEEDigis) ){
     EEDigis = pEEDigis.product(); // get a ptr to the product
     LogDebug("DataMixingEMDigiWorker") << "total # EE digis: " << EEDigis->size();
   } 
   
 
   if (EEDigis)
     {
       // loop over digis, storing them in a map so we can add pileup later
       for(EEDigiCollection::const_iterator it  = EEDigis->begin();	
	   it != EEDigis->end(); ++it) {

	 EEDigiStorage_.insert(EEDigiMap::value_type( ( it->id() ), *it ));
#ifdef DEBUG	 
	 LogDebug("DataMixingEMDigiWorker") << "processed EEDigi with rawId: "
				      << it->id().rawId() << "\n"
				      << " digi energy: " << it->energy();
#endif

       }
     }
   // ES next

   Handle< ESDigiCollection > pESDigis;

   const ESDigiCollection*  ESDigis = 0;

   
   if(e.getByLabel( ESProducerSig_.label(),ESdigiCollectionSig_.label(), pESDigis) ){
     ESDigis = pESDigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingEMDigiWorker") << "total # ES digis: " << ESDigis->size();
#endif
   } 

 
   if (ESDigis)
     {

       // loop over digis, storing them in a map so we can add pileup later
       for(ESDigiCollection::const_iterator it  = ESDigis->begin();	
	   it != ESDigis->end(); ++it) {

	 ESDigiStorage_.insert(ESDigiMap::value_type( ( it->id() ), *it ));
	 
#ifdef DEBUG	 
         LogDebug("DataMixingEMDigiWorker") << "processed ESDigi with rawId: "
				      << it->id().rawId() << "\n"
				      << " digi energy: " << it->energy();
#endif

       }
     }
    
  } // end of addEMSignals

  void DataMixingEMDigiWorker::addEMPileups(const int bcr, Event *e, unsigned int eventNr, const edm::EventSetup& ES) {
  
    LogInfo("DataMixingEMDigiWorker") <<"\n===============> adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // EB first

   Handle< EBDigiCollection > pEBDigis;
   const EBDigiCollection*  EBDigis = 0;

  
   if( e->getByLabel(EBProducerPile_.label(),EBdigiCollectionPile_.label(), pEBDigis) ){
     EBDigis = pEBDigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingEMDigiWorker") << "total # EB digis: " << EBDigis->size();
#endif
   } 
   
 
   if (EBDigis)
     {
       // loop over digis, adding these to the existing maps
       for(EBDigiCollection::const_iterator it  = EBDigis->begin();
	   it != EBDigis->end(); ++it) {

	 EBDigiStorage_.insert(EBDigiMap::value_type( (it->id()), *it ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingEMDigiWorker") << "processed EBDigi with rawId: "
				      << it->id().rawId() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
     }
    // EE Next

   Handle< EEDigiCollection > pEEDigis;
   const EEDigiCollection*  EEDigis = 0;

   
   if( e->getByLabel( EEProducerPile_.label(),EEdigiCollectionPile_.label(), pEEDigis) ){
     EEDigis = pEEDigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingEMDigiWorker") << "total # EE digis: " << EEDigis->size();
#endif
   }
   
 
   if (EEDigis)
     {
       // loop over digis, adding these to the existing maps
       for(EEDigiCollection::const_iterator it  = EEDigis->begin();
	   it != EEDigis->end(); ++it) {

	 EEDigiStorage_.insert(EEDigiMap::value_type( (it->id()), *it ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingEMDigiWorker") << "processed EEDigi with rawId: "
				      << it->id().rawId() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
     }
    // ES Next

   Handle< ESDigiCollection > pESDigis;
   const ESDigiCollection*  ESDigis = 0;

   
   if( e->getByLabel( ESProducerPile_.label(),ESdigiCollectionPile_.label(), pESDigis) ){
     ESDigis = pESDigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingEMDigiWorker") << "total # ES digis: " << ESDigis->size();
#endif
   } 
   
 
   if (ESDigis)
     {
       // loop over digis, adding these to the existing maps
       for(ESDigiCollection::const_iterator it  = ESDigis->begin();
	   it != ESDigis->end(); ++it) {

	 ESDigiStorage_.insert(ESDigiMap::value_type( (it->id()), *it ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingEMDigiWorker") << "processed ESDigi with rawId: "
				      << it->id().rawId() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
     }

  }
 
  void DataMixingEMDigiWorker::putEM(edm::Event &e, const edm::EventSetup& ES) {

    // collection of digis to put in the event
    std::auto_ptr< EBDigiCollection > EBdigis( new EBDigiCollection );
    std::auto_ptr< EEDigiCollection > EEdigis( new EEDigiCollection );
    std::auto_ptr< ESDigiCollection > ESdigis( new ESDigiCollection );

    // loop over the maps we have, re-making individual hits or digis if necessary.
    DetId formerID = 0;
    DetId currentID;

    EBDataFrame EB_old;

    int gain_new = 0;
    int gain_old = 0;
    int gain_consensus = 0;
    int adc_new;
    int adc_old;
    int adc_sum;
    uint16_t data;

    // EB first...

    EBDigiMap::const_iterator iEBchk;

    

    for(EBDigiMap::const_iterator iEB  = EBDigiStorage_.begin();
	iEB != EBDigiStorage_.end(); iEB++) {

      currentID = iEB->first; 

      if (currentID == formerID) { // we have to add these digis together

	//loop over digi samples in each DataFrame
	uint sizenew = (iEB->second).size();
	uint sizeold = EB_old.size();

	uint max_samp = max(sizenew, sizeold);

	// samples from different events can be of different lengths - sum all
	// that overlap.
	// check to see if gains match - if not, scale smaller cell down.

	for(uint isamp = 0; isamp<max_samp; isamp++) {
	  if(isamp < sizenew) {
	    gain_new = (iEB->second)[isamp].gainId();
	    adc_new = (iEB->second)[isamp].adc();
	  }
	  else { adc_new = 0;}

	  if(isamp < sizeold) {
	      gain_old = EB_old[isamp].gainId();
	      adc_old = EB_old[isamp].adc();
	  }
	  else { adc_old = 0;}

	  if(adc_new>0 && adc_old>0) {
	    if(gain_old == gain_new) {  // we're happy - easy case
	      gain_consensus = gain_old;
	    }
	    else {  // lower gain sample has more energy
	      if(gain_old > gain_new) {
		int gain_diff = (gain_old-gain_new)*6;
		adc_old = adc_old/gain_diff;  // scale energy to new gain
		gain_consensus = gain_new;
	      }
	      else {
		int gain_diff = (gain_new-gain_old)*6;
		adc_new = adc_new/gain_diff;  // scale energy to new gain
		gain_consensus = gain_old;
	      }
	    }
	  }

	  // add values
	  adc_sum = adc_new + adc_old;
	  // make data word of gain, rawdata
	  adc_sum = min(adc_sum,4096); //first 12 bits of (uint)
	  data = adc_sum+gain_consensus<<12; // data is 14 bit word with gain as MSBs
	  EB_old.setSample(isamp,data);  // overwrite old sample, adding new info
	}

      }
      else {
	  if(formerID>0) {
	    EBdigis->push_back( formerID, EB_old.frame().begin() );
	  }
	  //save pointers for next iteration
	  formerID = currentID;
	  EB_old = iEB->second;
      }

      iEBchk = iEB;
      if((++iEBchk) == EBDigiStorage_.end()) {  //make sure not to lose the last one
	    EBdigis->push_back( currentID, (iEB->second).frame().begin()) ;
      }
    }

    // EE next...

    formerID = 0;
    EEDataFrame EE_old;

    EEDigiMap::const_iterator iEEchk;

    for(EEDigiMap::const_iterator iEE  = EEDigiStorage_.begin();
	iEE != EEDigiStorage_.end(); iEE++) {

      currentID = iEE->first; 

      if (currentID == formerID) { // we have to add these digis together

	//loop over digi samples in each DataFrame
	uint sizenew = (iEE->second).size();
	uint sizeold = EE_old.size();

	uint max_samp = max(sizenew, sizeold);

	// samples from different events can be of different lengths - sum all
	// that overlap.
	// check to see if gains match - if not, scale smaller cell down.

	for(uint isamp = 0; isamp<max_samp; isamp++) {
	  if(isamp < sizenew) {
	    gain_new = (iEE->second)[isamp].gainId();
	    adc_new = (iEE->second)[isamp].adc();
	  }
	  else { adc_new = 0;}

	  if(isamp < sizeold) {
	      gain_old = EE_old[isamp].gainId();
	      adc_old = EE_old[isamp].adc();
	  }
	  else { adc_old = 0;}

	  if(adc_new>0 && adc_old>0) {
	    if(gain_old == gain_new) {  // we're happy - easy case
	      gain_consensus = gain_old;
	    }
	    else {  // lower gain sample has more energy
	      if(gain_old > gain_new) {
		int gain_diff = (gain_old-gain_new)*6;
		adc_old = adc_old/gain_diff;  // scale energy to new gain
		gain_consensus = gain_new;
	      }
	      else {
		int gain_diff = (gain_new-gain_old)*6;
		adc_new = adc_new/gain_diff;  // scale energy to new gain
		gain_consensus = gain_old;
	      }
	    }
	  }

	  // add values
	  adc_sum = adc_new + adc_old;
	  // make data word of gain, rawdata
	  adc_sum = min(adc_sum,4096); //first 12 bits of (uint)
	  data = adc_sum+gain_consensus<<12; // data is 14 bit word with gain as MSBs
	  EE_old.setSample(isamp,data);
	}

      }
      else {
	  if(formerID>0) {
	    EEdigis->push_back(formerID, EE_old.frame().begin() );
	  }
	  //save pointers for next iteration
	  formerID = currentID;
	  EE_old = iEE->second;
      }

      iEEchk = iEE;
      if((++iEEchk) == EEDigiStorage_.end()) {  //make sure not to lose the last one
	    EEdigis->push_back(currentID, (iEE->second).frame().begin());
      }
    }
   

    // ES next...

    formerID = 0;
    ESDataFrame ES_old;

    ESDigiMap::const_iterator iESchk;

    for(ESDigiMap::const_iterator iES  = ESDigiStorage_.begin();
	iES != ESDigiStorage_.end(); iES++) {

      currentID = iES->first; 

      if (currentID == formerID) { // we have to add these digis together

	//loop over digi samples in each DataFrame
	uint sizenew = (iES->second).size();
	uint sizeold = ES_old.size();
	uint16_t rawdat = 0;
	uint max_samp = max(sizenew, sizeold);

	// samples from different events can be of different lengths - sum all
	// that overlap.
	// check to see if gains match - if not, scale smaller cell down.

	for(uint isamp = 0; isamp<max_samp; isamp++) {
	  if(isamp < sizenew) {
	    adc_new = (iES->second)[isamp].adc();
	    rawdat = (iES->second)[isamp].raw();
	  }
	  else { adc_new = 0;}

	  if(isamp < sizeold) {
	      adc_old = ES_old[isamp].adc();
	      rawdat = ES_old[isamp].raw();
	  }
	  else { adc_old = 0;}

	  // add values
	  adc_sum = adc_new + adc_old;
	  // make data word of gain, rawdata
	  adc_sum = min(adc_sum,4095); //first 12 bits of (uint)
	  data = adc_sum+(rawdat&0xF000); // data is 14 bit word with gain as MSBs
	  ES_old.setSample(isamp,data);
	}

      }
      else {
	  if(formerID>0) {
	    ESdigis->push_back(ES_old);
	  }
	  //save pointers for next iteration
	  formerID = currentID;
	  ES_old = iES->second;
      }

      iESchk = iES;
      if((++iESchk) == ESDigiStorage_.end()) {  //make sure not to lose the last one
	ESdigis->push_back(iES->second);
	//	ESDataFrame df( (*ESdigis)->back() );
	//for(int isamp=0; isamp<(iES->second).size(); isamp++) {
	//  df.setSample(isamp,(iES->second).data[isamp]);
	//	}
      }
    }


    // done merging

    // put the collection of reconstructed hits in the event   
    LogInfo("DataMixingEMDigiWorker") << "total # EB Merged digis: " << EBdigis->size() ;
    LogInfo("DataMixingEMDigiWorker") << "total # EE Merged digis: " << EEdigis->size() ;
    LogInfo("DataMixingEMDigiWorker") << "total # ES Merged digis: " << ESdigis->size() ;

    e.put( EBdigis, EBDigiCollectionDM_ );
    e.put( EEdigis, EEDigiCollectionDM_ );
    e.put( ESdigis, ESDigiCollectionDM_ );
    
    // clear local storage after this event

    EBDigiStorage_.clear();
    EEDigiStorage_.clear();
    ESDigiStorage_.clear();

  }

} //edm
