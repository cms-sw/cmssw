// File: DataMixingHcalWorker.cc
// Description:  see DataMixingHcalWorker.h
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
#include "DataMixingHcalWorker.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingHcalWorker::DataMixingHcalWorker() { }

  // Constructor 
  DataMixingHcalWorker::DataMixingHcalWorker(const edm::ParameterSet& ps) : 
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

    // declare the products to produce

    // Hcal 

    HBHErechitCollection_  = ps.getParameter<edm::InputTag>("HBHErechitCollection");
    HOrechitCollection_    = ps.getParameter<edm::InputTag>("HOrechitCollection");
    HFrechitCollection_    = ps.getParameter<edm::InputTag>("HFrechitCollection");
    ZDCrechitCollection_   = ps.getParameter<edm::InputTag>("ZDCrechitCollection");

    HBHERecHitCollectionDM_ = ps.getParameter<std::string>("HBHERecHitCollectionDM");
    HORecHitCollectionDM_   = ps.getParameter<std::string>("HORecHitCollectionDM");
    HFRecHitCollectionDM_   = ps.getParameter<std::string>("HFRecHitCollectionDM");
    ZDCRecHitCollectionDM_  = ps.getParameter<std::string>("ZDCRecHitCollectionDM");


  }
	       
  // Virtual destructor needed.
  DataMixingHcalWorker::~DataMixingHcalWorker() { 
    delete sel_;

  }  

  void DataMixingHcalWorker::addHcalSignals(const edm::Event &e) { 
    // fill in maps of hits

    LogDebug("DataMixingHcalWorker")<<"===============> adding MC signals for "<<e.id();

    // HBHE first

   Handle< HBHERecHitCollection > pHBHERecHits;

   const HBHERecHitCollection*  HBHERecHits = 0;

   try {
     e.getByLabel( HBHErechitCollection_, pHBHERecHits);
     HBHERecHits = pHBHERecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalWorker") << "total # HBHE rechits: " << HBHERecHits->size();
#endif
   } catch (...) {
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

   const HORecHitCollection*  HORecHits = 0;

   try {
     e.getByLabel( HOrechitCollection_, pHORecHits);
     HORecHits = pHORecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalWorker") << "total # HO rechits: " << HORecHits->size();
#endif
   } catch (...) {
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

   const HFRecHitCollection*  HFRecHits = 0;

   try {
     e.getByLabel( HFrechitCollection_, pHFRecHits);
     HFRecHits = pHFRecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalWorker") << "total # HF rechits: " << HFRecHits->size();
#endif
   } catch (...) {
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

   const ZDCRecHitCollection*  ZDCRecHits = 0;

   try {
     e.getByLabel( ZDCrechitCollection_, pZDCRecHits);
     ZDCRecHits = pZDCRecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalWorker") << "total # ZDC rechits: " << ZDCRecHits->size();
#endif
   } catch (...) {
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

  void DataMixingHcalWorker::addHcalPileups(const int bcr, Event *e, unsigned int eventNr) {
  
    LogDebug("DataMixingHcalWorker") <<"\n===============> adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // HBHE first

   Handle< HBHERecHitCollection > pHBHERecHits;
   const HBHERecHitCollection*  HBHERecHits = 0;

   try {
     e->getByLabel( HBHErechitCollection_, pHBHERecHits);
     HBHERecHits = pHBHERecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalWorker") << "total # HEHB rechits: " << HBHERecHits->size();
#endif
   } catch (...) {
   }
 
   if (HBHERecHits)
     {
       // loop over rechits, adding these to the existing maps
       for(HBHERecHitCollection::const_iterator it  = HBHERecHits->begin();
	   it != HBHERecHits->end(); ++it) {

	 HBHERecHitStorage_.insert(HBHERecHitMap::value_type( (it->id()), *it ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalWorker") << "processed HBHERecHit with rawId: "
				      << it->id() << "\n"
				      << " rechit energy: " << it->energy();
#endif
       }
     }
    // HO Next

   Handle< HORecHitCollection > pHORecHits;
   const HORecHitCollection*  HORecHits = 0;

   try {
     e->getByLabel( HOrechitCollection_, pHORecHits);
     HORecHits = pHORecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalWorker") << "total # HO rechits: " << HORecHits->size();
#endif
   } catch (...) {
   }
 
   if (HORecHits)
     {
       // loop over rechits, adding these to the existing maps
       for(HORecHitCollection::const_iterator it  = HORecHits->begin();
	   it != HORecHits->end(); ++it) {

	 HORecHitStorage_.insert(HORecHitMap::value_type( (it->id()), *it ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalWorker") << "processed HORecHit with rawId: "
				      << it->id() << "\n"
				      << " rechit energy: " << it->energy();
#endif
       }
     }

    // HF Next

   Handle< HFRecHitCollection > pHFRecHits;
   const HFRecHitCollection*  HFRecHits = 0;

   try {
     e->getByLabel( HFrechitCollection_, pHFRecHits);
     HFRecHits = pHFRecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalWorker") << "total # HF rechits: " << HFRecHits->size();
#endif
   } catch (...) {
   }
 
   if (HFRecHits)
     {
       // loop over rechits, adding these to the existing maps
       for(HFRecHitCollection::const_iterator it  = HFRecHits->begin();
	   it != HFRecHits->end(); ++it) {

	 HFRecHitStorage_.insert(HFRecHitMap::value_type( (it->id()), *it ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalWorker") << "processed HFRecHit with rawId: "
				      << it->id() << "\n"
				      << " rechit energy: " << it->energy();
#endif
       }
     }

    // ZDC Next

   Handle< ZDCRecHitCollection > pZDCRecHits;
   const ZDCRecHitCollection*  ZDCRecHits = 0;

   try {
     e->getByLabel( ZDCrechitCollection_, pZDCRecHits);
     ZDCRecHits = pZDCRecHits.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalWorker") << "total # ZDC rechits: " << ZDCRecHits->size();
#endif
   } catch (...) {
   }
 
   if (ZDCRecHits)
     {
       // loop over rechits, adding these to the existing maps
       for(ZDCRecHitCollection::const_iterator it  = ZDCRecHits->begin();
	   it != ZDCRecHits->end(); ++it) {

	 ZDCRecHitStorage_.insert(ZDCRecHitMap::value_type( (it->id()), *it ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalWorker") << "processed ZDCRecHit with rawId: "
				      << it->id() << "\n"
				      << " rechit energy: " << it->energy();
#endif
       }
     }

  }
 
  void DataMixingHcalWorker::putHcal(edm::Event &e) {

    // collection of rechits to put in the event
    std::auto_ptr< HBHERecHitCollection > HBHErechits( new HBHERecHitCollection );
    std::auto_ptr< HORecHitCollection > HOrechits( new HORecHitCollection );
    std::auto_ptr< HFRecHitCollection > HFrechits( new HFRecHitCollection );
    std::auto_ptr< ZDCRecHitCollection > ZDCrechits( new ZDCRecHitCollection );

    // loop over the maps we have, re-making individual hits or digis if necessary.
    DetId formerID = 0;
    DetId currentID;
    float ESum = 0.;
    float HBTime = 0.;
    HBHERecHit OldHit;
    int nmatch=0;

    // HB first...

    HBHERecHitMap::const_iterator iHBchk;

    for(HBHERecHitMap::const_iterator iHB  = HBHERecHitStorage_.begin();
	iHB != HBHERecHitStorage_.end(); ++iHB) {

      currentID = iHB->first; 

      if (currentID == formerID) { // we have to add these rechits together
	nmatch++;                  // use this to avoid using the "count" function
	ESum+=(iHB->second).energy();          // on every element...

	iHBchk = iHB;
	if((iHBchk++) == HBHERecHitStorage_.end()) {  //make sure not to lose the last one
	  HBHERecHit aHit(formerID, ESum, HBTime);
	  HBHErechits->push_back( aHit );	  
	  // reset energy sum, nmatch
	  ESum = 0 ;
	  nmatch=0;	  
	}
      }
      else {
	if(nmatch>0) {
	  HBHERecHit aHit(formerID, ESum, HBTime);
	  HBHErechits->push_back( aHit );	  
	  // reset energy sum, nmatch
	  ESum = 0 ;
	  nmatch=0;	  
	}
	else {
	  HBHErechits->push_back( OldHit );
	}
	
	iHBchk = iHB;
	if((iHBchk++) == HBHERecHitStorage_.end()) {  //make sure not to lose the last one
	  HBHErechits->push_back( iHB->second );
	}

	// save pointers for next iteration
	OldHit = iHB->second;
	formerID = currentID;
	ESum = (iHB->second).energy();
	HBTime = (iHB->second).time();  // take time of first hit in sequence - is this ok?
      }
    }

    // HO next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    ESum = 0.;
    float HOTime = 0.;
    HORecHit HOOldHit;
    nmatch=0;

    HORecHitMap::const_iterator iHOchk;

    for(HORecHitMap::const_iterator iHO  = HORecHitStorage_.begin();
	iHO != HORecHitStorage_.end(); ++iHO) {

      currentID = iHO->first; 

      if (currentID == formerID) { // we have to add these rechits together
	nmatch++;                  // use this to avoid using the "count" function
	ESum+=(iHO->second).energy();          // on every element...

	iHOchk = iHO;
	if((iHOchk++) == HORecHitStorage_.end()) {  //make sure not to lose the last one
	  HORecHit aHit(formerID, ESum, HOTime);
	  HOrechits->push_back( aHit );	  
	  // reset energy sum, nmatch
	  ESum = 0 ;
	  nmatch=0;	  
	}
      }
      else {
	if(nmatch>0) {
	  HORecHit aHit(formerID, ESum, HOTime);
	  HOrechits->push_back( aHit );	  
	  // reset energy sum, nmatch
	  ESum = 0 ;
	  nmatch=0;	  
	}
	else {
	  HOrechits->push_back( HOOldHit );
	}
	
	iHOchk = iHO;
	if((iHOchk++) == HORecHitStorage_.end()) {  //make sure not to lose the last one
	  HOrechits->push_back( iHO->second );
	}

	// save pointers for next iteration
	HOOldHit = iHO->second;
	formerID = currentID;
	ESum = (iHO->second).energy();
	HOTime = (iHO->second).time();  // take time of first hit in sequence - is this ok?
      }
    }

    // HF next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    ESum = 0.;
    float HFTime = 0.;
    HFRecHit HFOldHit;
    nmatch=0;

    HFRecHitMap::const_iterator iHFchk;

    for(HFRecHitMap::const_iterator iHF  = HFRecHitStorage_.begin();
	iHF != HFRecHitStorage_.end(); ++iHF) {

      currentID = iHF->first; 

      if (currentID == formerID) { // we have to add these rechits together
	nmatch++;                  // use this to avoid using the "count" function
	ESum+=(iHF->second).energy();          // on every element...

	iHFchk = iHF;
	if((iHFchk++) == HFRecHitStorage_.end()) {  //make sure not to lose the last one
	  HFRecHit aHit(formerID, ESum, HFTime);
	  HFrechits->push_back( aHit );	  
	  // reset energy sum, nmatch
	  ESum = 0 ;
	  nmatch=0;	  
	}
      }
      else {
	if(nmatch>0) {
	  HFRecHit aHit(formerID, ESum, HFTime);
	  HFrechits->push_back( aHit );	  
	  // reset energy sum, nmatch
	  ESum = 0 ;
	  nmatch=0;	  
	}
	else {
	  HFrechits->push_back( HFOldHit );
	}
	
	iHFchk = iHF;
	if((iHFchk++) == HFRecHitStorage_.end()) {  //make sure not to lose the last one
	  HFrechits->push_back( iHF->second );
	}

	// save pointers for next iteration
	HFOldHit = iHF->second;
	formerID = currentID;
	ESum = (iHF->second).energy();
	HFTime = (iHF->second).time();  // take time of first hit in sequence - is this ok?
      }
    }

    // ZDC next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    ESum = 0.;
    float ZDCTime = 0.;
    ZDCRecHit ZOldHit;
    nmatch=0;

    ZDCRecHitMap::const_iterator iZDCchk;

    for(ZDCRecHitMap::const_iterator iZDC  = ZDCRecHitStorage_.begin();
	iZDC != ZDCRecHitStorage_.end(); ++iZDC) {

      currentID = iZDC->first; 

      if (currentID == formerID) { // we have to add these rechits together
	nmatch++;                  // use this to avoid using the "count" function
	ESum+=(iZDC->second).energy();          // on every element...

	iZDCchk = iZDC;
	if((iZDCchk++) == ZDCRecHitStorage_.end()) {  //make sure not to lose the last one
	  ZDCRecHit aHit(formerID, ESum, ZDCTime);
	  ZDCrechits->push_back( aHit );	  
	  // reset energy sum, nmatch
	  ESum = 0 ;
	  nmatch=0;	  
	}
      }
      else {
	if(nmatch>0) {
	  ZDCRecHit aHit(formerID, ESum, ZDCTime);
	  ZDCrechits->push_back( aHit );	  
	  // reset energy sum, nmatch
	  ESum = 0 ;
	  nmatch=0;	  
	}
	else {
	  ZDCrechits->push_back( ZOldHit );
	}
	
	iZDCchk = iZDC;
	if((iZDCchk++) == ZDCRecHitStorage_.end()) {  //make sure not to lose the last one
	  ZDCrechits->push_back( iZDC->second );
	}

	// save pointers for next iteration
	ZOldHit = iZDC->second;
	formerID = currentID;
	ESum = (iZDC->second).energy();
	ZDCTime = (iZDC->second).time();  // take time of first hit in sequence - is this ok?
      }
    }

    // put the collection of recunstructed hits in the event   
    LogInfo("DataMixingHcalWorker") << "total # HBHE Merged rechits: " << HBHErechits->size() ;
    LogInfo("DataMixingHcalWorker") << "total # HO Merged rechits: " << HOrechits->size() ;
    LogInfo("DataMixingHcalWorker") << "total # HF Merged rechits: " << HFrechits->size() ;
    LogInfo("DataMixingHcalWorker") << "total # ZDC Merged rechits: " << ZDCrechits->size() ;

    e.put( HBHErechits, HBHERecHitCollectionDM_ );
    e.put( HOrechits, HORecHitCollectionDM_ );
    e.put( HFrechits, HFRecHitCollectionDM_ );
    e.put( ZDCrechits, ZDCRecHitCollectionDM_ );

    // clear local storage after this event
    HBHERecHitStorage_.clear();
    HORecHitStorage_.clear();
    HFRecHitStorage_.clear();
    ZDCRecHitStorage_.clear();

  }

} //edm
