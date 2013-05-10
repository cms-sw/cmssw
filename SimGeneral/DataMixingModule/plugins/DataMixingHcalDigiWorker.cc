// File: DataMixingHcalDigiWorker.cc
// Description:  see DataMixingHcalDigiWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
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
#include "DataMixingHcalDigiWorker.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingHcalDigiWorker::DataMixingHcalDigiWorker() { }

  // Constructor 
  DataMixingHcalDigiWorker::DataMixingHcalDigiWorker(const edm::ParameterSet& ps) : 
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    // Hcal 

    HBHEdigiCollectionSig_  = ps.getParameter<edm::InputTag>("HBHEdigiCollectionSig");
    HOdigiCollectionSig_    = ps.getParameter<edm::InputTag>("HOdigiCollectionSig");
    HFdigiCollectionSig_    = ps.getParameter<edm::InputTag>("HFdigiCollectionSig");
    ZDCdigiCollectionSig_   = ps.getParameter<edm::InputTag>("ZDCdigiCollectionSig");

    HBHEPileInputTag_ = ps.getParameter<edm::InputTag>("HBHEPileInputTag");
    HOPileInputTag_ = ps.getParameter<edm::InputTag>("HOPileInputTag");
    HFPileInputTag_ = ps.getParameter<edm::InputTag>("HFPileInputTag");
    ZDCPileInputTag_ = ps.getParameter<edm::InputTag>("ZDCPileInputTag");

    DoZDC_ = false;
    if(ZDCPileInputTag_.label() != "") DoZDC_ = true;

    HBHEDigiCollectionDM_ = ps.getParameter<std::string>("HBHEDigiCollectionDM");
    HODigiCollectionDM_   = ps.getParameter<std::string>("HODigiCollectionDM");
    HFDigiCollectionDM_   = ps.getParameter<std::string>("HFDigiCollectionDM");
    ZDCDigiCollectionDM_  = ps.getParameter<std::string>("ZDCDigiCollectionDM");


  }
	       
  // Virtual destructor needed.
  DataMixingHcalDigiWorker::~DataMixingHcalDigiWorker() { 
  }  

  void DataMixingHcalDigiWorker::addHcalSignals(const edm::Event &e,const edm::EventSetup& ES) { 
    // Calibration stuff will look like this:                                                 

    // get conditions                                                                         
    edm::ESHandle<HcalDbService> conditions;                                                
    ES.get<HcalDbRecord>().get(conditions);                                         



    // fill in maps of hits

    LogInfo("DataMixingHcalDigiWorker")<<"===============> adding MC signals for "<<e.id();

    // HBHE first

   Handle< HBHEDigiCollection > pHBHEDigis;

   const HBHEDigiCollection*  HBHEDigis = 0;

   if( e.getByLabel( HBHEdigiCollectionSig_, pHBHEDigis) ) {
     HBHEDigis = pHBHEDigis.product(); // get a ptr to the product
     LogDebug("DataMixingHcalDigiWorker") << "total # HBHE digis: " << HBHEDigis->size();
   } 
   //   else { std::cout << "NO HBHE Digis " << HBHEdigiCollectionSig_.label() << std::endl;}
   
 
   if (HBHEDigis)
     {
       // loop over digis, storing them in a map so we can add pileup later
       for(HBHEDigiCollection::const_iterator it  = HBHEDigis->begin();	
	   it != HBHEDigis->end(); ++it) {

         // calibration, for future reference:  (same block for all Hcal types)               

         HcalDetId cell = it->id();                                                         
	 //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);        
         const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);                
	 const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
         HcalCoderDb coder (*channelCoder, *shape);                                         

	 CaloSamples tool;
	 coder.adc2fC((*it),tool);

	 // put sample values back into digi?


	 // RecHit MyHit = reco_.reconstruct(*it,coder,calibrations));                         
         //... can now fish calibrated information from RecHit                                


	 HBHEDigiStorage_.insert(HBHEDigiMap::value_type( ( it->id() ), tool ));
	 
#ifdef DEBUG	 
         LogDebug("DataMixingHcalDigiWorker") << "processed HBHEDigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif

       }
     }

   // HO next

   Handle< HODigiCollection > pHODigis;

   const HODigiCollection*  HODigis = 0;

   if( e.getByLabel( HOdigiCollectionSig_, pHODigis) ){
     HODigis = pHODigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalDigiWorker") << "total # HO digis: " << HODigis->size();
#endif
   } 
   
 
   if (HODigis)
     {
       // loop over digis, storing them in a map so we can add pileup later
       for(HODigiCollection::const_iterator it  = HODigis->begin();	
	   it != HODigis->end(); ++it) {

         // calibration, for future reference:  (same block for all Hcal types)                                
         HcalDetId cell = it->id();
         //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
         const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	 const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
         HcalCoderDb coder (*channelCoder, *shape);

         CaloSamples tool;
         coder.adc2fC((*it),tool);

	 HODigiStorage_.insert(HODigiMap::value_type( ( it->id() ), tool ));
	 
#ifdef DEBUG	 
         LogDebug("DataMixingHcalDigiWorker") << "processed HODigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif

       }
     }

   // HF next

   Handle< HFDigiCollection > pHFDigis;

   const HFDigiCollection*  HFDigis = 0;

   if( e.getByLabel( HFdigiCollectionSig_, pHFDigis) ) {
     HFDigis = pHFDigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalDigiWorker") << "total # HF digis: " << HFDigis->size();
#endif
   } 
   
 
   if (HFDigis)
     {
       // loop over digis, storing them in a map so we can add pileup later
       for(HFDigiCollection::const_iterator it  = HFDigis->begin();	
	   it != HFDigis->end(); ++it) {

         // calibration, for future reference:  (same block for all Hcal types)                                
         HcalDetId cell = it->id();
         //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
         const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	 const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
         HcalCoderDb coder (*channelCoder, *shape);

         CaloSamples tool;
         coder.adc2fC((*it),tool);

	 HFDigiStorage_.insert(HFDigiMap::value_type( ( it->id() ), tool ));
	 
#ifdef DEBUG	 
         LogDebug("DataMixingHcalDigiWorker") << "processed HFDigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif

       }
     }

   // ZDC next

   if(DoZDC_){

     Handle< ZDCDigiCollection > pZDCDigis;

     const ZDCDigiCollection*  ZDCDigis = 0;

     if( e.getByLabel( ZDCdigiCollectionSig_, pZDCDigis) ) {
       ZDCDigis = pZDCDigis.product(); // get a ptr to the product
#ifdef DEBUG
       LogDebug("DataMixingHcalDigiWorker") << "total # ZDC digis: " << ZDCDigis->size();
#endif
     } 
   
 
     if (ZDCDigis)
       {
	 // loop over digis, storing them in a map so we can add pileup later
	 for(ZDCDigiCollection::const_iterator it  = ZDCDigis->begin();	
	     it != ZDCDigis->end(); ++it) {

	   // calibration, for future reference:  (same block for all Hcal types)                                
	   HcalDetId cell = it->id();
	   //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
	   const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	   const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
	   HcalCoderDb coder (*channelCoder, *shape);

	   CaloSamples tool;
	   coder.adc2fC((*it),tool);

	   ZDCDigiStorage_.insert(ZDCDigiMap::value_type( ( it->id() ), tool ));
	 
#ifdef DEBUG	 
	   LogDebug("DataMixingHcalDigiWorker") << "processed ZDCDigi with rawId: "
						<< it->id() << "\n"
						<< " digi energy: " << it->energy();
#endif

	 }
       }
   }
    
  } // end of addHCalSignals

  void DataMixingHcalDigiWorker::addHcalPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,const edm::EventSetup& ES) {
  
    LogDebug("DataMixingHcalDigiWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    // get conditions                                                                                                             
    edm::ESHandle<HcalDbService> conditions;
    ES.get<HcalDbRecord>().get(conditions);

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // HBHE first

    boost::shared_ptr<Wrapper<HBHEDigiCollection>  const> HBHEDigisPTR = 
          getProductByTag<HBHEDigiCollection>(*ep, HBHEPileInputTag_ );
 
    if(HBHEDigisPTR ) {

     const HBHEDigiCollection*  HBHEDigis = const_cast< HBHEDigiCollection * >(HBHEDigisPTR->product());

     LogInfo("DataMixingHcalDigiWorker") << "total # HBHE digis: " << HBHEDigis->size();

       // loop over digis, adding these to the existing maps
       for(HBHEDigiCollection::const_iterator it  = HBHEDigis->begin();
	   it != HBHEDigis->end(); ++it) {

         // calibration, for future reference:  (same block for all Hcal types)                                
         HcalDetId cell = it->id();
         //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
         const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	 const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
         HcalCoderDb coder (*channelCoder, *shape);

         CaloSamples tool;
         coder.adc2fC((*it),tool);

	 HBHEDigiStorage_.insert(HBHEDigiMap::value_type( (it->id()), tool ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalDigiWorker") << "processed HBHEDigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
    }
    // HO Next

    boost::shared_ptr<Wrapper<HODigiCollection>  const> HODigisPTR = 
          getProductByTag<HODigiCollection>(*ep, HOPileInputTag_ );
 
    if(HODigisPTR ) {

     const HODigiCollection*  HODigis = const_cast< HODigiCollection * >(HODigisPTR->product());

     LogDebug("DataMixingHcalDigiWorker") << "total # HO digis: " << HODigis->size();

       // loop over digis, adding these to the existing maps
       for(HODigiCollection::const_iterator it  = HODigis->begin();
	   it != HODigis->end(); ++it) {

         // calibration, for future reference:  (same block for all Hcal types)                                
         HcalDetId cell = it->id();
         //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
         const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	 const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
         HcalCoderDb coder (*channelCoder, *shape);

         CaloSamples tool;
         coder.adc2fC((*it),tool);

	 HODigiStorage_.insert(HODigiMap::value_type( (it->id()), tool ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalDigiWorker") << "processed HODigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
    }


    // HF Next

    boost::shared_ptr<Wrapper<HFDigiCollection>  const> HFDigisPTR = 
          getProductByTag<HFDigiCollection>(*ep, HFPileInputTag_ );
 
    if(HFDigisPTR ) {

     const HFDigiCollection*  HFDigis = const_cast< HFDigiCollection * >(HFDigisPTR->product());

     LogDebug("DataMixingHcalDigiWorker") << "total # HF digis: " << HFDigis->size();

       // loop over digis, adding these to the existing maps
       for(HFDigiCollection::const_iterator it  = HFDigis->begin();
	   it != HFDigis->end(); ++it) {

         // calibration, for future reference:  (same block for all Hcal types)                                
         HcalDetId cell = it->id();
         //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
         const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	 const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
         HcalCoderDb coder (*channelCoder, *shape);

         CaloSamples tool;
         coder.adc2fC((*it),tool);

	 HFDigiStorage_.insert(HFDigiMap::value_type( (it->id()), tool ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalDigiWorker") << "processed HFDigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
    }


    // ZDC Next

    if(DoZDC_) {


      boost::shared_ptr<Wrapper<ZDCDigiCollection>  const> ZDCDigisPTR = 
	getProductByTag<ZDCDigiCollection>(*ep, ZDCPileInputTag_ );
 
      if(ZDCDigisPTR ) {

	const ZDCDigiCollection*  ZDCDigis = const_cast< ZDCDigiCollection * >(ZDCDigisPTR->product());

	LogDebug("DataMixingHcalDigiWorker") << "total # ZDC digis: " << ZDCDigis->size();

	// loop over digis, adding these to the existing maps
	for(ZDCDigiCollection::const_iterator it  = ZDCDigis->begin();
	    it != ZDCDigis->end(); ++it) {

	  // calibration, for future reference:  (same block for all Hcal types)                                
	  HcalDetId cell = it->id();
	  //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
	  HcalCoderDb coder (*channelCoder, *shape);

	  CaloSamples tool;
	  coder.adc2fC((*it),tool);

	  ZDCDigiStorage_.insert(ZDCDigiMap::value_type( (it->id()), tool ));
	 
#ifdef DEBUG	 
	  LogDebug("DataMixingHcalDigiWorker") << "processed ZDCDigi with rawId: "
					       << it->id() << "\n"
					       << " digi energy: " << it->energy();
#endif
	}
      }
    }


  }
 
  void DataMixingHcalDigiWorker::putHcal(edm::Event &e,const edm::EventSetup& ES) {

    // collection of digis to put in the event
    std::auto_ptr< HBHEDigiCollection > HBHEdigis( new HBHEDigiCollection );
    std::auto_ptr< HODigiCollection > HOdigis( new HODigiCollection );
    std::auto_ptr< HFDigiCollection > HFdigis( new HFDigiCollection );
    std::auto_ptr< ZDCDigiCollection > ZDCdigis( new ZDCDigiCollection );

    // get conditions                                                                                                             
    edm::ESHandle<HcalDbService> conditions;
    ES.get<HcalDbRecord>().get(conditions);

    // loop over the maps we have, re-making individual hits or digis if necessary.
    DetId formerID = 0;
    DetId currentID;

    CaloSamples HB_old;

    double fC_new;
    double fC_old;
    double fC_sum;

    // HB first...

    HBHEDigiMap::const_iterator iHBchk;

    for(HBHEDigiMap::const_iterator iHB  = HBHEDigiStorage_.begin();
	iHB != HBHEDigiStorage_.end(); ++iHB) {

      currentID = iHB->first; 

      if (currentID == formerID) { // we have to add these digis together

        //loop over digi samples in each CaloSample                                                  

        unsigned int sizenew = (iHB->second).size();
        unsigned int sizeold = HB_old.size();

	bool usenew = false;

	if(sizenew > sizeold) usenew = true;

	unsigned int max_samp = std::max(sizenew, sizeold);

	CaloSamples HB_bigger(currentID,max_samp);	

	// HB_old.setSize(max_samp);  --> can't do this...

        // samples from different events can be of different lengths - sum all                      
	// that overlap.                

        for(unsigned int isamp = 0; isamp<max_samp; isamp++) {
          if(isamp < sizenew) {
            fC_new = (iHB->second)[isamp]; // should return nominal_fC();
          }
          else { fC_new = 0;}

          if(isamp < sizeold) {
	    fC_old = HB_old[isamp];
          }
          else { fC_old = 0;}

          // add values   
          fC_sum = fC_new + fC_old;
	  
	  //uint fCS = int(fC_sum);
	  //const HcalQIESample fC(fCS); 
	  //HB_old.setSample(isamp, fC);

	  if(usenew) {HB_bigger[isamp] = fC_sum; }
	  else { HB_old[isamp] = fC_sum; }  // overwrite old sample, adding new info     

        }
	if(usenew) HB_old = HB_bigger; // save new, larger sized sample in "old" slot

      }
      else {
	if(formerID>0) {
	  // make new digi
	  HBHEdigis->push_back(HBHEDataFrame(formerID));	  

	  // set up information to convert back

	  HcalDetId cell = HB_old.id();
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
	  HcalCoderDb coder (*channelCoder, *shape);

	  unsigned int sizeold = HB_old.size();
	  for(unsigned int isamp = 0; isamp<sizeold; isamp++) {
	    coder.fC2adc(HB_old,(HBHEdigis->back()), 0 );   // as per simulation, capid=0???
	  }
	}
	//save pointers for next iteration                                                                 
	formerID = currentID;
	HB_old = iHB->second;
	//OldUpAdd = HB_old.id(); 
      }

      iHBchk = iHB;
      if((++iHBchk) == HBHEDigiStorage_.end()) {  //make sure not to lose the last one                         

	// make new digi                                                                                     
	HBHEdigis->push_back(HBHEDataFrame(currentID));

	// set up information to convert back                                                                

	HcalDetId cell = (iHB->second).id();
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
	HcalCoderDb coder (*channelCoder, *shape);

        unsigned int sizenew = (iHB->second).size();
	for(unsigned int isamp = 0; isamp<sizenew; isamp++) {
	  coder.fC2adc(HB_old,(HBHEdigis->back()), 0 );  // as per simulation, capid=0???
  	}
      }
    }


    // HO next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    CaloSamples HO_old;

    HODigiMap::const_iterator iHOchk;

    for(HODigiMap::const_iterator iHO  = HODigiStorage_.begin();
	iHO != HODigiStorage_.end(); ++iHO) {

      currentID = iHO->first; 

      if (currentID == formerID) { // we have to add these digis together

        //loop over digi samples in each CaloSample                                                           
        unsigned int sizenew = (iHO->second).size();
        unsigned int sizeold = HO_old.size();

        unsigned int max_samp = std::max(sizenew, sizeold);

        CaloSamples HO_bigger(currentID,max_samp);

        bool usenew = false;

        if(sizenew > sizeold) usenew = true;

        // samples from different events can be of different lengths - sum all                               
        // that overlap.                                                                                     

        for(unsigned int isamp = 0; isamp<max_samp; isamp++) {
          if(isamp < sizenew) {
            fC_new = (iHO->second)[isamp];
          }
          else { fC_new = 0;}

          if(isamp < sizeold) {
	    fC_old = HO_old[isamp];
          }
          else { fC_old = 0;}

          // add values                                                                                      
          fC_sum = fC_new + fC_old;

	  if(usenew) {HO_bigger[isamp] = fC_sum; }
	  else { HO_old[isamp] = fC_sum; }  // overwrite old sample, adding new info     

        }
	if(usenew) HO_old = HO_bigger; // save new, larger sized sample in "old" slot
      
      }
      else {
	if(formerID>0) {
	  // make new digi
	  HOdigis->push_back(HODataFrame(formerID));	  

	  // set up information to convert back

	  HcalDetId cell = HO_old.id();
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
	  HcalCoderDb coder (*channelCoder, *shape);

	  unsigned int sizeold = HO_old.size();
	  for(unsigned int isamp = 0; isamp<sizeold; isamp++) {
	    coder.fC2adc(HO_old,(HOdigis->back()), 0 );   // as per simulation, capid=0???
	  }
	}
	//save pointers for next iteration                                                                 
	formerID = currentID;
	HO_old = iHO->second;
      }

      iHOchk = iHO;
      if((++iHOchk) == HODigiStorage_.end()) {  //make sure not to lose the last one                         
	  // make new digi
	  HOdigis->push_back(HODataFrame(currentID));	  

	  // set up information to convert back

	  HcalDetId cell = (iHO->second).id();
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
	  HcalCoderDb coder (*channelCoder, *shape);

	  unsigned int sizeold = (iHO->second).size();
	  for(unsigned int isamp = 0; isamp<sizeold; isamp++) {
	    coder.fC2adc(HO_old,(HOdigis->back()), 0 );   // as per simulation, capid=0???
	  }

      }
    }

    // HF next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    CaloSamples HF_old;

    HFDigiMap::const_iterator iHFchk;

    for(HFDigiMap::const_iterator iHF  = HFDigiStorage_.begin();
	iHF != HFDigiStorage_.end(); ++iHF) {

      currentID = iHF->first; 

      if (currentID == formerID) { // we have to add these digis together

        //loop over digi samples in each CaloSample                                                           
        unsigned int sizenew = (iHF->second).size();
        unsigned int sizeold = HF_old.size();

        unsigned int max_samp = std::max(sizenew, sizeold);

        CaloSamples HF_bigger(currentID,max_samp);

        bool usenew = false;

        if(sizenew > sizeold) usenew = true;

        // samples from different events can be of different lengths - sum all                               
        // that overlap.                                                                                     

        for(unsigned int isamp = 0; isamp<max_samp; isamp++) {
          if(isamp < sizenew) {
            fC_new = (iHF->second)[isamp];
          }
          else { fC_new = 0;}

          if(isamp < sizeold) {
	    fC_old = HF_old[isamp];
          }
          else { fC_old = 0;}

          // add values                                                                                      
          fC_sum = fC_new + fC_old;

	  if(usenew) {HF_bigger[isamp] = fC_sum; }
	  else { HF_old[isamp] = fC_sum; }  // overwrite old sample, adding new info     

        }
	if(usenew) HF_old = HF_bigger; // save new, larger sized sample in "old" slot
      
      }
      else {
	if(formerID>0) {
	  // make new digi
	  HFdigis->push_back(HFDataFrame(formerID));	  

	  // set up information to convert back

	  HcalDetId cell = HF_old.id();
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
	  HcalCoderDb coder (*channelCoder, *shape);

	  unsigned int sizeold = HF_old.size();
	  for(unsigned int isamp = 0; isamp<sizeold; isamp++) {
	    coder.fC2adc(HF_old,(HFdigis->back()), 0 );   // as per simulation, capid=0???
	  }
	}
	//save pointers for next iteration                                                                 
	formerID = currentID;
	HF_old = iHF->second;
      }

      iHFchk = iHF;
      if((++iHFchk) == HFDigiStorage_.end()) {  //make sure not to lose the last one                         
	  // make new digi
	  HFdigis->push_back(HFDataFrame(currentID));	  

	  // set up information to convert back

	  HcalDetId cell = (iHF->second).id();
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
	  HcalCoderDb coder (*channelCoder, *shape);

	  unsigned int sizeold = (iHF->second).size();
	  for(unsigned int isamp = 0; isamp<sizeold; isamp++) {
	    coder.fC2adc(HF_old,(HFdigis->back()), 0 );   // as per simulation, capid=0???
	  }

      }
    }


    // ZDC next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    CaloSamples ZDC_old;

    ZDCDigiMap::const_iterator iZDCchk;

    for(ZDCDigiMap::const_iterator iZDC  = ZDCDigiStorage_.begin();
	iZDC != ZDCDigiStorage_.end(); ++iZDC) {

      currentID = iZDC->first; 

      if (currentID == formerID) { // we have to add these digis together

        //loop over digi samples in each CaloSample                                                           
        unsigned int sizenew = (iZDC->second).size();
        unsigned int sizeold = ZDC_old.size();

        unsigned int max_samp = std::max(sizenew, sizeold);

        CaloSamples ZDC_bigger(currentID,max_samp);

        bool usenew = false;

        if(sizenew > sizeold) usenew = true;

        // samples from different events can be of different lengths - sum all                               
        // that overlap.                                                                                     

        for(unsigned int isamp = 0; isamp<max_samp; isamp++) {
          if(isamp < sizenew) {
            fC_new = (iZDC->second)[isamp];
          }
          else { fC_new = 0;}

          if(isamp < sizeold) {
	    fC_old = ZDC_old[isamp];
          }
          else { fC_old = 0;}

          // add values                                                                                      
          fC_sum = fC_new + fC_old;

	  if(usenew) {ZDC_bigger[isamp] = fC_sum; }
	  else { ZDC_old[isamp] = fC_sum; }  // overwrite old sample, adding new info     

        }
	if(usenew) ZDC_old = ZDC_bigger; // save new, larger sized sample in "old" slot
      
      }
      else {
	if(formerID>0) {
	  // make new digi
	  ZDCdigis->push_back(ZDCDataFrame(formerID));	  

	  // set up information to convert back

	  HcalDetId cell = ZDC_old.id();
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
	  HcalCoderDb coder (*channelCoder, *shape);

	  unsigned int sizeold = ZDC_old.size();
	  for(unsigned int isamp = 0; isamp<sizeold; isamp++) {
	    coder.fC2adc(ZDC_old,(ZDCdigis->back()), 0 );   // as per simulation, capid=0???
	  }
	}
	//save pointers for next iteration                                                                 
	formerID = currentID;
	ZDC_old = iZDC->second;
      }

      iZDCchk = iZDC;
      if((++iZDCchk) == ZDCDigiStorage_.end()) {  //make sure not to lose the last one                         
	  // make new digi
	  ZDCdigis->push_back(ZDCDataFrame(currentID));	  

	  // set up information to convert back

	  HcalDetId cell = (iZDC->second).id();
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
	  HcalCoderDb coder (*channelCoder, *shape);

	  unsigned int sizeold = (iZDC->second).size();
	  for(unsigned int isamp = 0; isamp<sizeold; isamp++) {
	    coder.fC2adc(ZDC_old,(ZDCdigis->back()), 0 );   // as per simulation, capid=0???
	  }

      }
    }


  
   //done merging

    // put the collection of recunstructed hits in the event   
    LogInfo("DataMixingHcalDigiWorker") << "total # HBHE Merged digis: " << HBHEdigis->size() ;
    LogInfo("DataMixingHcalDigiWorker") << "total # HO Merged digis: " << HOdigis->size() ;
    LogInfo("DataMixingHcalDigiWorker") << "total # HF Merged digis: " << HFdigis->size() ;
    LogInfo("DataMixingHcalDigiWorker") << "total # ZDC Merged digis: " << ZDCdigis->size() ;

    e.put( HBHEdigis, HBHEDigiCollectionDM_ );
    e.put( HOdigis, HODigiCollectionDM_ );
    e.put( HFdigis, HFDigiCollectionDM_ );
    e.put( ZDCdigis, ZDCDigiCollectionDM_ );

    // clear local storage after this event
    HBHEDigiStorage_.clear();
    HODigiStorage_.clear();
    HFDigiStorage_.clear();
    ZDCDigiStorage_.clear();

  }

} //edm
