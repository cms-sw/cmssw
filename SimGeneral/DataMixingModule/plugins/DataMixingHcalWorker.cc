// File: DataMixingHcalWorker.cc
// Description:  see DataMixingHcalWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include <memory>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
//
//
#include "DataMixingHcalWorker.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingHcalWorker::DataMixingHcalWorker() { } 

  // Constructor 
  DataMixingHcalWorker::DataMixingHcalWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC) : 
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    // Hcal 

    HBHErechitCollectionSig_  = ps.getParameter<edm::InputTag>("HBHEProducerSig");
    HOrechitCollectionSig_    = ps.getParameter<edm::InputTag>("HOProducerSig");
    HFrechitCollectionSig_    = ps.getParameter<edm::InputTag>("HFProducerSig");
    ZDCrechitCollectionSig_   = ps.getParameter<edm::InputTag>("ZDCrechitCollectionSig");

    HBHEPileRecHitInputTag_ = ps.getParameter<edm::InputTag>("HBHEPileRecHitInputTag");
    HOPileRecHitInputTag_ = ps.getParameter<edm::InputTag>("HOPileRecHitInputTag");
    HFPileRecHitInputTag_ = ps.getParameter<edm::InputTag>("HFPileRecHitInputTag");
    ZDCPileRecHitInputTag_ = ps.getParameter<edm::InputTag>("ZDCPileRecHitInputTag");

    HBHERecHitCollectionDM_ = ps.getParameter<std::string>("HBHERecHitCollectionDM");
    HORecHitCollectionDM_   = ps.getParameter<std::string>("HORecHitCollectionDM");
    HFRecHitCollectionDM_   = ps.getParameter<std::string>("HFRecHitCollectionDM");
    ZDCRecHitCollectionDM_  = ps.getParameter<std::string>("ZDCRecHitCollectionDM");

    HBHERecHitToken_ = iC.consumes<HBHERecHitCollection>(HBHErechitCollectionSig_);
    HORecHitToken_ = iC.consumes<HORecHitCollection>(HOrechitCollectionSig_);
    HFRecHitToken_ = iC.consumes<HFRecHitCollection>(HFrechitCollectionSig_);

    HBHERecHitPToken_ = iC.consumes<HBHERecHitCollection>(HBHEPileRecHitInputTag_);
    HORecHitPToken_ = iC.consumes<HORecHitCollection>(HOPileRecHitInputTag_);
    HFRecHitPToken_ = iC.consumes<HFRecHitCollection>(HFPileRecHitInputTag_);


    ZDCRecHitToken_ = iC.consumes<ZDCRecHitCollection>(ZDCrechitCollectionSig_);
    ZDCRecHitPToken_ = iC.consumes<ZDCRecHitCollection>(ZDCPileRecHitInputTag_);

  }
	       
  // Virtual destructor needed.
  DataMixingHcalWorker::~DataMixingHcalWorker() { 
  }  

  void DataMixingHcalWorker::addHcalSignals(const edm::Event &e) { 
    // fill in maps of hits

    LogInfo("DataMixingHcalWorker")<<"===============> adding MC signals for "<<e.id();

    // HBHE first

   Handle< HBHERecHitCollection > pHBHERecHits;

   const HBHERecHitCollection*  HBHERecHits = nullptr;

   if( e.getByToken( HBHERecHitToken_, pHBHERecHits) ) {
     HBHERecHits = pHBHERecHits.product(); // get a ptr to the product
     LogDebug("DataMixingHcalWorker") << "total # HBHE rechits: " << HBHERecHits->size();
   } 
   
 
   if (HBHERecHits)
     {
       // loop over rechits, storing them in a map so we can add pileup later
       for(HBHERecHitCollection::const_iterator it  = HBHERecHits->begin();	
	   it != HBHERecHits->end(); ++it) {

	 HBHERecHitStorage_.insert(HBHERecHitMap::value_type( ( it->id() ), *it ));
	 
#ifdef DEBUG	 
         LogDebug("DataMixingHcalWorker") << "processed HBHERecHit with rawId: "
				      << it->id() << "\n"
				      << " rechit energy: " << it->energy();
#endif

       }
     }

   // HO next

   Handle< HORecHitCollection > pHORecHits;

   const HORecHitCollection*  HORecHits = nullptr;

   if( e.getByToken( HORecHitToken_, pHORecHits) ){
     HORecHits = pHORecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalWorker") << "total # HO rechits: " << HORecHits->size();
#endif
   } 
   
 
   if (HORecHits)
     {
       // loop over rechits, storing them in a map so we can add pileup later
       for(HORecHitCollection::const_iterator it  = HORecHits->begin();	
	   it != HORecHits->end(); ++it) {

	 HORecHitStorage_.insert(HORecHitMap::value_type( ( it->id() ), *it ));
	 
#ifdef DEBUG	 
         LogDebug("DataMixingHcalWorker") << "processed HORecHit with rawId: "
				      << it->id() << "\n"
				      << " rechit energy: " << it->energy();
#endif

       }
     }

   // HF next

   Handle< HFRecHitCollection > pHFRecHits;

   const HFRecHitCollection*  HFRecHits = nullptr;

   if( e.getByToken( HFRecHitToken_, pHFRecHits) ) {
     HFRecHits = pHFRecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalWorker") << "total # HF rechits: " << HFRecHits->size();
#endif
   } 
   
 
   if (HFRecHits)
     {
       // loop over rechits, storing them in a map so we can add pileup later
       for(HFRecHitCollection::const_iterator it  = HFRecHits->begin();	
	   it != HFRecHits->end(); ++it) {

	 HFRecHitStorage_.insert(HFRecHitMap::value_type( ( it->id() ), *it ));
	 
#ifdef DEBUG	 
         LogDebug("DataMixingHcalWorker") << "processed HFRecHit with rawId: "
				      << it->id() << "\n"
				      << " rechit energy: " << it->energy();
#endif

       }
     }

   // ZDC next

   Handle< ZDCRecHitCollection > pZDCRecHits;

   const ZDCRecHitCollection*  ZDCRecHits = nullptr;

   if( e.getByToken( ZDCRecHitToken_, pZDCRecHits) ) {
     ZDCRecHits = pZDCRecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalWorker") << "total # ZDC rechits: " << ZDCRecHits->size();
#endif
   } 
   
 
   if (ZDCRecHits)
     {
       // loop over rechits, storing them in a map so we can add pileup later
       for(ZDCRecHitCollection::const_iterator it  = ZDCRecHits->begin();	
	   it != ZDCRecHits->end(); ++it) {

	 ZDCRecHitStorage_.insert(ZDCRecHitMap::value_type( ( it->id() ), *it ));
	 
#ifdef DEBUG	 
         LogDebug("DataMixingHcalWorker") << "processed ZDCRecHit with rawId: "
				      << it->id() << "\n"
				      << " rechit energy: " << it->energy();
#endif

       }
     }
    
  } // end of addEMSignals

  void DataMixingHcalWorker::addHcalPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,
                                            ModuleCallingContext const* mcc) {
  
    LogDebug("DataMixingHcalWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // HBHE first

    std::shared_ptr<Wrapper<HBHERecHitCollection>  const> HBHERecHitsPTR =
      getProductByTag<HBHERecHitCollection>(*ep, HBHEPileRecHitInputTag_, mcc);

    if(HBHERecHitsPTR ) {

      const HBHERecHitCollection*  HBHERecHits = const_cast< HBHERecHitCollection * >(HBHERecHitsPTR->product());

      LogDebug("DataMixingEMWorker") << "total # HBHE rechits: " << HBHERecHits->size();

      // loop over rechits, adding these to the existing maps                                                     
      for(HBHERecHitCollection::const_iterator it  = HBHERecHits->begin();
          it != HBHERecHits->end(); ++it) {

        HBHERecHitStorage_.insert(HBHERecHitMap::value_type( (it->id()), *it ));

#ifdef DEBUG
        LogDebug("DataMixingEMWorker") << "processed HBHERecHit with rawId: "
				       << it->id().rawId() << "\n"
				       << " rechit energy: " << it->energy();
#endif
      }
    }

    // HO Next

    std::shared_ptr<Wrapper<HORecHitCollection>  const> HORecHitsPTR =
      getProductByTag<HORecHitCollection>(*ep, HOPileRecHitInputTag_, mcc);

    if(HORecHitsPTR ) {

      const HORecHitCollection*  HORecHits = const_cast< HORecHitCollection * >(HORecHitsPTR->product());

      LogDebug("DataMixingEMWorker") << "total # HO rechits: " << HORecHits->size();

      // loop over rechits, adding these to the existing maps                                                     
      for(HORecHitCollection::const_iterator it  = HORecHits->begin();
          it != HORecHits->end(); ++it) {

        HORecHitStorage_.insert(HORecHitMap::value_type( (it->id()), *it ));

#ifdef DEBUG
        LogDebug("DataMixingEMWorker") << "processed HORecHit with rawId: "
				       << it->id().rawId() << "\n"
				       << " rechit energy: " << it->energy();
#endif
      }
    }

    // HF Next

    std::shared_ptr<Wrapper<HFRecHitCollection>  const> HFRecHitsPTR =
      getProductByTag<HFRecHitCollection>(*ep, HFPileRecHitInputTag_, mcc);

    if(HFRecHitsPTR ) {

      const HFRecHitCollection*  HFRecHits = const_cast< HFRecHitCollection * >(HFRecHitsPTR->product());

      LogDebug("DataMixingEMWorker") << "total # HF rechits: " << HFRecHits->size();

      // loop over rechits, adding these to the existing maps                                                     
      for(HFRecHitCollection::const_iterator it  = HFRecHits->begin();
          it != HFRecHits->end(); ++it) {

        HFRecHitStorage_.insert(HFRecHitMap::value_type( (it->id()), *it ));

#ifdef DEBUG
        LogDebug("DataMixingEMWorker") << "processed HFRecHit with rawId: "
				       << it->id().rawId() << "\n"
				       << " rechit energy: " << it->energy();
#endif
      }
    }

    // ZDC Next

    std::shared_ptr<Wrapper<ZDCRecHitCollection>  const> ZDCRecHitsPTR =
      getProductByTag<ZDCRecHitCollection>(*ep, ZDCPileRecHitInputTag_, mcc);

    if(ZDCRecHitsPTR ) {

      const ZDCRecHitCollection*  ZDCRecHits = const_cast< ZDCRecHitCollection * >(ZDCRecHitsPTR->product());

      LogDebug("DataMixingEMWorker") << "total # ZDC rechits: " << ZDCRecHits->size();

      // loop over rechits, adding these to the existing maps                                                     
      for(ZDCRecHitCollection::const_iterator it  = ZDCRecHits->begin();
          it != ZDCRecHits->end(); ++it) {

        ZDCRecHitStorage_.insert(ZDCRecHitMap::value_type( (it->id()), *it ));

#ifdef DEBUG
        LogDebug("DataMixingEMWorker") << "processed ZDCRecHit with rawId: "
				       << it->id().rawId() << "\n"
				       << " rechit energy: " << it->energy();
#endif
      }
    }


  }
 
  void DataMixingHcalWorker::putHcal(edm::Event &e) {

    // collection of rechits to put in the event
    std::unique_ptr< HBHERecHitCollection > HBHErechits( new HBHERecHitCollection );
    std::unique_ptr< HORecHitCollection > HOrechits( new HORecHitCollection );
    std::unique_ptr< HFRecHitCollection > HFrechits( new HFRecHitCollection );
    std::unique_ptr< ZDCRecHitCollection > ZDCrechits( new ZDCRecHitCollection );

    // loop over the maps we have, re-making individual hits or rechits if necessary.
    DetId formerID = 0;
    DetId currentID;
    float ESum = 0.;
    float HBTime = 0.;

    // HB first...

    HBHERecHitMap::const_iterator iHBchk;

    for(HBHERecHitMap::const_iterator iHB  = HBHERecHitStorage_.begin();
	iHB != HBHERecHitStorage_.end(); ++iHB) {

      currentID = iHB->first; 

      if (currentID == formerID) { // we have to add these rechits together

	ESum+=(iHB->second).energy();  

      }
      else {
	if(formerID>0) {
	  // cutoff for ESum?                                                                                 
	  HBHERecHit aHit(formerID, ESum, HBTime);
	  HBHErechits->push_back( aHit );
	}
	//save pointers for next iteration                                                                    
	formerID = currentID;
	ESum = (iHB->second).energy();
	HBTime = (iHB->second).time();  // take time of first hit in sequence - is this ok?
      }

      iHBchk = iHB;
      if((++iHBchk) == HBHERecHitStorage_.end()) {  //make sure not to lose the last one  
        HBHERecHit aHit(formerID, ESum, HBTime);
        HBHErechits->push_back( aHit );
      }
    }

    // HO next...

    // loop over the maps we have, re-making individual hits or rechits if necessary.
    formerID = 0;
    ESum = 0.;
    float HOTime = 0.;

    HORecHitMap::const_iterator iHOchk;

    for(HORecHitMap::const_iterator iHO  = HORecHitStorage_.begin();
	iHO != HORecHitStorage_.end(); ++iHO) {

      currentID = iHO->first; 

      if (currentID == formerID) { // we have to add these rechits together

	ESum+=(iHO->second).energy();  

      }
      else {
	if(formerID>0) {
	  // cutoff for ESum?                                                                                 
	  HORecHit aHit(formerID, ESum, HOTime);
	  HOrechits->push_back( aHit );
	}
	//save pointers for next iteration                                                                    
	formerID = currentID;
	ESum = (iHO->second).energy();
	HOTime = (iHO->second).time();  // take time of first hit in sequence - is this ok?
      }

      iHOchk = iHO;
      if((++iHOchk) == HORecHitStorage_.end()) {  //make sure not to lose the last one  
        HORecHit aHit(formerID, ESum, HOTime);
        HOrechits->push_back( aHit );
      }
    }


    // HF next...

    // loop over the maps we have, re-making individual hits or rechits if necessary.
    formerID = 0;
    ESum = 0.;
    float HFTime = 0.;

    HFRecHitMap::const_iterator iHFchk;

    for(HFRecHitMap::const_iterator iHF  = HFRecHitStorage_.begin();
	iHF != HFRecHitStorage_.end(); ++iHF) {

      currentID = iHF->first; 

      if (currentID == formerID) { // we have to add these rechits together

	ESum+=(iHF->second).energy();  

      }
      else {
	if(formerID>0) {
	  // cutoff for ESum?                                                                                 
	  HFRecHit aHit(formerID, ESum, HFTime);
	  HFrechits->push_back( aHit );
	}
	//save pointers for next iteration                                                                    
	formerID = currentID;
	ESum = (iHF->second).energy();
	HFTime = (iHF->second).time();  // take time of first hit in sequence - is this ok?
      }

      iHFchk = iHF;
      if((++iHFchk) == HFRecHitStorage_.end()) {  //make sure not to lose the last one  
        HFRecHit aHit(formerID, ESum, HBTime);
        HFrechits->push_back( aHit );
      }
    }

    // ZDC next...

    // loop over the maps we have, re-making individual hits or rechits if necessary.
    formerID = 0;
    ESum = 0.;
    float ZDCTime = 0.;
    float lowGainEnergy = 0;
    ZDCRecHit ZOldHit;

    ZDCRecHitMap::const_iterator iZDCchk;

    for(ZDCRecHitMap::const_iterator iZDC  = ZDCRecHitStorage_.begin();
	iZDC != ZDCRecHitStorage_.end(); ++iZDC) {

      currentID = iZDC->first; 

      if (currentID == formerID) { // we have to add these rechits together

	ESum+=(iZDC->second).energy();  
	
      }
      else {
	if(formerID>0) {
	  // cutoff for ESum?                                                                                 
	  ZDCRecHit aHit(formerID, ESum, ZDCTime, lowGainEnergy);
	  ZDCrechits->push_back( aHit );
	}
	//save pointers for next iteration                                                                    
	formerID = currentID;
	ESum = (iZDC->second).energy();
	lowGainEnergy = (iZDC->second).lowGainEnergy();
	ZDCTime = (iZDC->second).time();  // take time of first hit in sequence - is this ok?
      }
      
      iZDCchk = iZDC;
      if((++iZDCchk) == ZDCRecHitStorage_.end()) {  //make sure not to lose the last one  
	ZDCRecHit aHit(formerID, ESum, HBTime, lowGainEnergy);
	ZDCrechits->push_back( aHit );
      }
    } 
  
   //done merging

    // put the collection of recunstructed hits in the event   
    LogInfo("DataMixingHcalWorker") << "total # HBHE Merged rechits: " << HBHErechits->size() ;
    LogInfo("DataMixingHcalWorker") << "total # HO Merged rechits: " << HOrechits->size() ;
    LogInfo("DataMixingHcalWorker") << "total # HF Merged rechits: " << HFrechits->size() ;
    LogInfo("DataMixingHcalWorker") << "total # ZDC Merged rechits: " << ZDCrechits->size() ;

    e.put(std::move(HBHErechits), HBHERecHitCollectionDM_ );
    e.put(std::move(HOrechits), HORecHitCollectionDM_ );
    e.put(std::move(HFrechits), HFRecHitCollectionDM_ );
    e.put(std::move(ZDCrechits), ZDCRecHitCollectionDM_ );

    // clear local storage after this event
    HBHERecHitStorage_.clear();
    HORecHitStorage_.clear();
    HFRecHitStorage_.clear();
    ZDCRecHitStorage_.clear();

  }

} //edm
