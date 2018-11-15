// File: DataMixingEMWorker.cc
// Description:  see DataMixingEMWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include <memory>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
//
//
#include "DataMixingEMWorker.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingEMWorker::DataMixingEMWorker() { }

  // Constructor 
  DataMixingEMWorker::DataMixingEMWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC) : 
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce, retrieve

    EBProducerSig_ = ps.getParameter<edm::InputTag>("EBProducerSig");
    EEProducerSig_ = ps.getParameter<edm::InputTag>("EEProducerSig");
    ESProducerSig_ = ps.getParameter<edm::InputTag>("ESProducerSig");

    EBRecHitToken_ = iC.consumes<EBRecHitCollection>(EBProducerSig_);
    EERecHitToken_ = iC.consumes<EERecHitCollection>(EEProducerSig_);
    ESRecHitToken_ = iC.consumes<ESRecHitCollection>(ESProducerSig_);


    EBrechitCollectionSig_ = ps.getParameter<edm::InputTag>("EBrechitCollectionSig");
    EErechitCollectionSig_ = ps.getParameter<edm::InputTag>("EErechitCollectionSig");
    ESrechitCollectionSig_ = ps.getParameter<edm::InputTag>("ESrechitCollectionSig");

    EBPileRecHitInputTag_ = ps.getParameter<edm::InputTag>("EBPileRecHitInputTag");
    EEPileRecHitInputTag_ = ps.getParameter<edm::InputTag>("EEPileRecHitInputTag");
    ESPileRecHitInputTag_ = ps.getParameter<edm::InputTag>("ESPileRecHitInputTag");

    EBPileRecHitToken_ = iC.consumes<EBRecHitCollection>(EBPileRecHitInputTag_);
    EEPileRecHitToken_ = iC.consumes<EERecHitCollection>(EEPileRecHitInputTag_);
    ESPileRecHitToken_ = iC.consumes<ESRecHitCollection>(ESPileRecHitInputTag_);


    EBRecHitCollectionDM_        = ps.getParameter<std::string>("EBRecHitCollectionDM");
    EERecHitCollectionDM_        = ps.getParameter<std::string>("EERecHitCollectionDM");
    ESRecHitCollectionDM_        = ps.getParameter<std::string>("ESRecHitCollectionDM");

  }
	       

  // Virtual destructor needed.
  DataMixingEMWorker::~DataMixingEMWorker() { 
  }  

  void DataMixingEMWorker::addEMSignals(const edm::Event &e) { 
    // fill in maps of hits

    LogInfo("DataMixingEMWorker")<<"===============> adding MC signals for "<<e.id();

    // EB first

   Handle< EBRecHitCollection > pEBRecHits;

   const EBRecHitCollection*  EBRecHits = nullptr;

   if(e.getByToken(EBRecHitToken_, pEBRecHits) ){
     EBRecHits = pEBRecHits.product(); // get a ptr to the product
     LogDebug("DataMixingEMWorker") << "total # EB rechits: " << EBRecHits->size();
   }
   
 
   if (EBRecHits)
     {
       // loop over rechits, storing them in a map so we can add pileup later
       for(EBRecHitCollection::const_iterator it  = EBRecHits->begin();	
	   it != EBRecHits->end(); ++it) {

	 EBRecHitStorage_.insert(EBRecHitMap::value_type( ( it->id() ), *it ));
	 
         LogDebug("DataMixingEMWorker") << "processed EBRecHit with rawId: "
				      << it->id().rawId() << "\n"
				      << " rechit energy: " << it->energy();

       }
     }

   // EE next

   Handle< EERecHitCollection > pEERecHits;

   const EERecHitCollection*  EERecHits = nullptr;

   
   if(e.getByToken(EERecHitToken_, pEERecHits) ){
     EERecHits = pEERecHits.product(); // get a ptr to the product
     LogDebug("DataMixingEMWorker") << "total # EE rechits: " << EERecHits->size();
   } 
   
 
   if (EERecHits)
     {
       // loop over rechits, storing them in a map so we can add pileup later
       for(EERecHitCollection::const_iterator it  = EERecHits->begin();	
	   it != EERecHits->end(); ++it) {

	 EERecHitStorage_.insert(EERecHitMap::value_type( ( it->id() ), *it ));
#ifdef DEBUG	 
	 LogDebug("DataMixingEMWorker") << "processed EERecHit with rawId: "
				      << it->id().rawId() << "\n"
				      << " rechit energy: " << it->energy();
#endif

       }
     }
   // ES next

   Handle< ESRecHitCollection > pESRecHits;

   const ESRecHitCollection*  ESRecHits = nullptr;

   
   if(e.getByToken( ESRecHitToken_, pESRecHits) ){
     ESRecHits = pESRecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingEMWorker") << "total # ES rechits: " << ESRecHits->size();
#endif
   } 
   
 
   if (ESRecHits)
     {
       // loop over rechits, storing them in a map so we can add pileup later
       for(ESRecHitCollection::const_iterator it  = ESRecHits->begin();	
	   it != ESRecHits->end(); ++it) {

	 ESRecHitStorage_.insert(ESRecHitMap::value_type( ( it->id() ), *it ));
	 
#ifdef DEBUG	 
         LogDebug("DataMixingEMWorker") << "processed ESRecHit with rawId: "
				      << it->id().rawId() << "\n"
				      << " rechit energy: " << it->energy();
#endif

       }
     }
    
  } // end of addEMSignals

  void DataMixingEMWorker::addEMPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,
                                        ModuleCallingContext const* mcc) {
  
    LogInfo("DataMixingEMWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // EB first

    std::shared_ptr<Wrapper<EBRecHitCollection>  const> EBRecHitsPTR =
      getProductByTag<EBRecHitCollection>(*ep, EBPileRecHitInputTag_, mcc);

    if(EBRecHitsPTR ) {

      const EBRecHitCollection*  EBRecHits = const_cast< EBRecHitCollection * >(EBRecHitsPTR->product());

      LogDebug("DataMixingEMWorker") << "total # EB rechits: " << EBRecHits->size();

      // loop over digis, adding these to the existing maps                                         
      for(EBRecHitCollection::const_iterator it  = EBRecHits->begin();
	  it != EBRecHits->end(); ++it) {

	EBRecHitStorage_.insert(EBRecHitMap::value_type( (it->id()), *it ));

#ifdef DEBUG
	LogDebug("DataMixingEMWorker") << "processed EBRecHit with rawId: "
					   << it->id().rawId() << "\n"
					   << " rechit energy: " << it->energy();
#endif
      }
    }

    // EE Next

    std::shared_ptr<Wrapper<EERecHitCollection>  const> EERecHitsPTR =
      getProductByTag<EERecHitCollection>(*ep, EEPileRecHitInputTag_, mcc);

    if(EERecHitsPTR ) {

      const EERecHitCollection*  EERecHits = const_cast< EERecHitCollection * >(EERecHitsPTR->product());

      LogDebug("DataMixingEMWorker") << "total # EE rechits: " << EERecHits->size();

      // loop over digis, adding these to the existing maps                                         
      for(EERecHitCollection::const_iterator it  = EERecHits->begin();
          it != EERecHits->end(); ++it) {

        EERecHitStorage_.insert(EERecHitMap::value_type( (it->id()), *it ));

#ifdef DEBUG
        LogDebug("DataMixingEMWorker") << "processed EERecHit with rawId: "
				       << it->id().rawId() << "\n"
				       << " rechit energy: " << it->energy();
#endif
      }
    }

    // ES Next

    std::shared_ptr<Wrapper<ESRecHitCollection>  const> ESRecHitsPTR =
      getProductByTag<ESRecHitCollection>(*ep, ESPileRecHitInputTag_, mcc);

    if(ESRecHitsPTR ) {

      const ESRecHitCollection*  ESRecHits = const_cast< ESRecHitCollection * >(ESRecHitsPTR->product());

      LogDebug("DataMixingEMWorker") << "total # ES rechits: " << ESRecHits->size();

      // loop over digis, adding these to the existing maps                                         
      for(ESRecHitCollection::const_iterator it  = ESRecHits->begin();
          it != ESRecHits->end(); ++it) {

        ESRecHitStorage_.insert(ESRecHitMap::value_type( (it->id()), *it ));

#ifdef DEBUG
        LogDebug("DataMixingEMWorker") << "processed ESRecHit with rawId: "
				       << it->id().rawId() << "\n"
				       << " rechit energy: " << it->energy();
#endif
      }
    }


  }
 
  void DataMixingEMWorker::putEM(edm::Event &e) {

    // collection of rechits to put in the event
    std::unique_ptr< EBRecHitCollection > EBrechits( new EBRecHitCollection );
    std::unique_ptr< EERecHitCollection > EErechits( new EERecHitCollection );
    std::unique_ptr< ESRecHitCollection > ESrechits( new ESRecHitCollection );

    // loop over the maps we have, re-making individual hits or digis if necessary.
    DetId formerID = 0;
    DetId currentID;
    float ESum = 0.;
    float EBTime = 0.;

    // EB first...

    EBRecHitMap::const_iterator iEBchk;

    for(EBRecHitMap::const_iterator iEB  = EBRecHitStorage_.begin();
	iEB != EBRecHitStorage_.end(); ++iEB) {

      currentID = iEB->first; 

      if (currentID == formerID) { // we have to add these rechits together

	ESum+=(iEB->second).energy(); 
      }
      else {
	  if(formerID>0) {
	    // cutoff for ESum?
	    EcalRecHit aHit(formerID, ESum, EBTime);
	    EBrechits->push_back( aHit );
	  }
	  //save pointers for next iteration
	  formerID = currentID;
	  ESum = (iEB->second).energy();
	  EBTime = (iEB->second).time();  // take time of first hit in sequence - is this ok?
      }

      iEBchk = iEB;
      if((++iEBchk) == EBRecHitStorage_.end()) {  //make sure not to lose the last one
	EcalRecHit aHit(formerID, ESum, EBTime);
	EBrechits->push_back( aHit );	  
      }
    }

    // EE next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    ESum = 0.;
    float EETime = 0.;
    
    EERecHitMap::const_iterator iEEchk;

    for(EERecHitMap::const_iterator iEE  = EERecHitStorage_.begin();
	iEE != EERecHitStorage_.end(); ++iEE) {

      currentID = iEE->first; 

      if (currentID == formerID) { // we have to add these rechits together

	ESum+=(iEE->second).energy(); 
      }
      else {
	  if(formerID>0) {
	    // cutoff for ESum?
	    EcalRecHit aHit(formerID, ESum, EETime);
	    EErechits->push_back( aHit );
	  }
	  //save pointers for next iteration
	  formerID = currentID;
	  ESum = (iEE->second).energy();
	  EETime = (iEE->second).time();  // take time of first hit in sequence - is this ok?
      }

      iEEchk = iEE;
      if((++iEEchk) == EERecHitStorage_.end()) {  //make sure not to lose the last one
	EcalRecHit aHit(formerID, ESum, EETime);
	EErechits->push_back( aHit );	  
      }
    }

    // ES next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    ESum = 0.;
    float ESTime = 0.;

    ESRecHitMap::const_iterator iESchk;

    for(ESRecHitMap::const_iterator iES  = ESRecHitStorage_.begin();
	iES != ESRecHitStorage_.end(); ++iES) {

      currentID = iES->first; 

      if (currentID == formerID) { // we have to add these rechits together

	ESum+=(iES->second).energy(); 
      }
      else {
	  if(formerID>0) {
	    // cutoff for ESum?
	    EcalRecHit aHit(formerID, ESum, ESTime);
	    ESrechits->push_back( aHit );
	  }
	  //save pointers for next iteration
	  formerID = currentID;
	  ESum = (iES->second).energy();
	  ESTime = (iES->second).time();  // take time of first hit in sequence - is this ok?
      }

      iESchk = iES;
      if((++iESchk) == ESRecHitStorage_.end()) {  //make sure not to lose the last one
	EcalRecHit aHit(formerID, ESum, ESTime);
	ESrechits->push_back( aHit );	  
      }
    }

    // done merging

    // put the collection of reconstructed hits in the event   
    LogInfo("DataMixingEMWorker") << "total # EB Merged rechits: " << EBrechits->size() ;
    LogInfo("DataMixingEMWorker") << "total # EE Merged rechits: " << EErechits->size() ;
    LogInfo("DataMixingEMWorker") << "total # ES Merged rechits: " << ESrechits->size() ;

    e.put(std::move(EBrechits), EBRecHitCollectionDM_ );
    e.put(std::move(EErechits), EERecHitCollectionDM_ );
    e.put(std::move(ESrechits), ESRecHitCollectionDM_ );
    
    // clear local storage after this event

    EBRecHitStorage_.clear();
    EERecHitStorage_.clear();
    ESRecHitStorage_.clear();

  }

} //edm

//  LocalWords:  ESProducerSig
