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
#include "DataMixingHcalDigiWorker.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingHcalDigiWorker::DataMixingHcalDigiWorker() {sel_=0;} 

  // Constructor 
  DataMixingHcalDigiWorker::DataMixingHcalDigiWorker(const edm::ParameterSet& ps) : 
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

    HBHEdigiCollection_  = ps.getParameter<edm::InputTag>("HBHEProducer");
    HOdigiCollection_    = ps.getParameter<edm::InputTag>("HOProducer");
    HFdigiCollection_    = ps.getParameter<edm::InputTag>("HFProducer");
    ZDCdigiCollection_   = ps.getParameter<edm::InputTag>("ZDCdigiCollection");

    HBHEDigiCollectionDM_ = ps.getParameter<std::string>("HBHEDigiCollectionDM");
    HODigiCollectionDM_   = ps.getParameter<std::string>("HODigiCollectionDM");
    HFDigiCollectionDM_   = ps.getParameter<std::string>("HFDigiCollectionDM");
    ZDCDigiCollectionDM_  = ps.getParameter<std::string>("ZDCDigiCollectionDM");


  }
	       
  // Virtual destructor needed.
  DataMixingHcalDigiWorker::~DataMixingHcalDigiWorker() { 
    delete sel_;
    sel_=0;
  }  

  void DataMixingHcalDigiWorker::addHcalSignals(const edm::Event &e,const edm::EventSetup& ES) { 
    // Calibration stuff will look like this:                                                 

    // get conditions                                                                         
    edm::ESHandle<HcalDbService> conditions;                                                
    ES.get<HcalDbRecord>().get(conditions);                                         

    const HcalQIEShape* shape = conditions->getHcalShape (); // this one is generic         


    // fill in maps of hits

    LogInfo("DataMixingHcalDigiWorker")<<"===============> adding MC signals for "<<e.id();

    // HBHE first

   Handle< HBHEDigiCollection > pHBHEDigis;

   const HBHEDigiCollection*  HBHEDigis = 0;

   if( e.getByLabel( HBHEdigiCollection_.label(), pHBHEDigis) ) {
     HBHEDigis = pHBHEDigis.product(); // get a ptr to the product
     LogDebug("DataMixingHcalDigiWorker") << "total # HBHE digis: " << HBHEDigis->size();
   } 
   
 
   if (HBHEDigis)
     {
       // loop over digis, storing them in a map so we can add pileup later
       for(HBHEDigiCollection::const_iterator it  = HBHEDigis->begin();	
	   it != HBHEDigis->end(); ++it) {

         // calibration, for future reference:  (same block for all Hcal types)               

         HcalDetId cell = it->id();                                                         
	 //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);        
         const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);                
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

   if( e.getByLabel( HOdigiCollection_.label(), pHODigis) ){
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

   if( e.getByLabel( HFdigiCollection_.label(), pHFDigis) ) {
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

   Handle< ZDCDigiCollection > pZDCDigis;

   const ZDCDigiCollection*  ZDCDigis = 0;

   if( e.getByLabel( ZDCdigiCollection_.label(), pZDCDigis) ) {
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
    
  } // end of addEMSignals

  void DataMixingHcalDigiWorker::addHcalPileups(const int bcr, Event *e, unsigned int eventNr,const edm::EventSetup& ES) {
  
    LogDebug("DataMixingHcalDigiWorker") <<"\n===============> adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    // get conditions                                                                                                             
    edm::ESHandle<HcalDbService> conditions;
    ES.get<HcalDbRecord>().get(conditions);

    const HcalQIEShape* shape = conditions->getHcalShape (); // this one is generic         

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // HBHE first

   Handle< HBHEDigiCollection > pHBHEDigis;
   const HBHEDigiCollection*  HBHEDigis = 0;

   if( e->getByLabel( HBHEdigiCollection_.label(), pHBHEDigis) ) {
     HBHEDigis = pHBHEDigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalDigiWorker") << "total # HEHB digis: " << HBHEDigis->size();
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

	 HBHEDigiStorage_.insert(HBHEDigiMap::value_type( (it->id()), tool ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalDigiWorker") << "processed HBHEDigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
     }
    // HO Next

   Handle< HODigiCollection > pHODigis;
   const HODigiCollection*  HODigis = 0;

   if( e->getByLabel( HOdigiCollection_.label(), pHODigis) ) {
     HODigis = pHODigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalDigiWorker") << "total # HO digis: " << HODigis->size();
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

	 HODigiStorage_.insert(HODigiMap::value_type( (it->id()), tool ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalDigiWorker") << "processed HODigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
     }

    // HF Next

   Handle< HFDigiCollection > pHFDigis;
   const HFDigiCollection*  HFDigis = 0;

   if( e->getByLabel( HFdigiCollection_.label(), pHFDigis) ) {
     HFDigis = pHFDigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalDigiWorker") << "total # HF digis: " << HFDigis->size();
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

	 HFDigiStorage_.insert(HFDigiMap::value_type( (it->id()), tool ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalDigiWorker") << "processed HFDigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif
       }
     }

    // ZDC Next

   Handle< ZDCDigiCollection > pZDCDigis;
   const ZDCDigiCollection*  ZDCDigis = 0;

   if( e->getByLabel( ZDCdigiCollection_.label(), pZDCDigis) ) {
     ZDCDigis = pZDCDigis.product(); // get a ptr to the product
#ifdef DEBUG
     LogDebug("DataMixingHcalDigiWorker") << "total # ZDC digis: " << ZDCDigis->size();
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

	 ZDCDigiStorage_.insert(ZDCDigiMap::value_type( (it->id()), tool ));
	 
#ifdef DEBUG	 
	 LogDebug("DataMixingHcalDigiWorker") << "processed ZDCDigi with rawId: "
				      << it->id() << "\n"
				      << " digi energy: " << it->energy();
#endif
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

    // un-calibrate
    const HcalQIEShape* shape = conditions->getHcalShape (); // this one is generic                            

    // loop over the maps we have, re-making individual hits or digis if necessary.
    DetId formerID = 0;
    DetId currentID;

    HBHEDataFrame HB_old;

    double fC_new;
    double fC_old;
    double fC_sum;
    uint16_t data;
    uint16_t OldUpAdd;


    // HB first...

    HBHEDigiMap::const_iterator iHBchk;

    for(HBHEDigiMap::const_iterator iHB  = HBHEDigiStorage_.begin();
	iHB != HBHEDigiStorage_.end(); ++iHB) {

      currentID = iHB->first; 

      if (currentID == formerID) { // we have to add these digis together

        //loop over digi samples in each CaloSample                                                           


        uint sizenew = (iHB->second).size();
        uint sizeold = HB_old.size();

        uint max_samp = max(sizenew, sizeold);

	HB_old.setSize(max_samp);

        // samples from different events can be of different lengths - sum all                               
        // that overlap.                                                                                     

        for(uint isamp = 0; isamp<max_samp; isamp++) {
          if(isamp < sizenew) {
            fC_new = (iHB->second)[isamp].nominal_fC();
          }
          else { fC_new = 0;}

          if(isamp < sizeold) {
	    fC_old = HB_old[isamp].nominal_fC();
          }
          else { fC_old = 0;}

          // add values                                                                                      
          fC_sum = fC_new + fC_old;
	  
          HB_old[isamp] = fC_sum;  // overwrite old sample, adding new info     
        }

      }
      else {
	if(formerID>0) {
	  // make new digi
	  HBHEdigis->push_back(HBHEDataFrame(formerID));	  

	  // set up information to convert back

	  HcalDetId cell = HB_old.id();
	  const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	  HcalCoderDb coder (*channelCoder, *shape);

	  uint sizeold = HB_old.size();
	  for(uint isamp = 0; isamp<sizeold; isamp++) {
	    coder.fC2adc(HB_old,(HBHEdigis->back()),HB_old[isamp].capid());
	  }
	}
	//save pointers for next iteration                                                                 
	formerID = currentID;
	HB_old = iHB->second;
	OldUpAdd = HB_old.id(); 
      }

      iHBchk = iHB;
      if((++iHBchk) == HBHEDigiStorage_.end()) {  //make sure not to lose the last one                         

	// make new digi                                                                                     
	HBHEdigis->push_back(HBHEDataFrame(currentID));

	// set up information to convert back                                                                

	HcalDetId cell = (iHB->second).id();
	const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
	HcalCoderDb coder (*channelCoder, *shape);

        uint sizenew = (iHB->second).size();
	for(uint isamp = 0; isamp<sizenew; isamp++) {
	  coder.fC2adc(HB_old,(HBHEdigis->back()),(iHB->second)[isamp].data().capid());
	}
      }
    }


    // HO next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    HOCaloSample HO_old;

    HODigiMap::const_iterator iHOchk;

    for(HODigiMap::const_iterator iHO  = HODigiStorage_.begin();
	iHO != HODigiStorage_.end(); ++iHO) {

      currentID = iHO->first; 

      if (currentID == formerID) { // we have to add these digis together

        //loop over digi samples in each CaloSample                                                           
        uint sizenew = (iHO->second).size();
        uint sizeold = HO_old.size();

        uint max_samp = max(sizenew, sizeold);

        // samples from different events can be of different lengths - sum all                               
        // that overlap.                                                                                     

        for(uint isamp = 0; isamp<max_samp; isamp++) {
          if(isamp < sizenew) {
            fC_new = (iHO->second)[isamp].adc();
          }
          else { fC_new = 0;}

          if(isamp < sizeold) {
	    fC_old = HO_old[isamp].adc();
          }
          else { fC_old = 0;}

          // add values                                                                                      
          fC_sum = fC_new + fC_old;
	  fC_sum = min(fC_sum,127); //first 7 bits of (uint)                                             
          // replace right bits of sample with new ADC

          fC_sum = min(fC_sum,127); //first 7 bits of Sample                                             
          data = OldUpAdd + fC_sum;  //add new data to old address
          HO_old.setSample(isamp,data);  // overwrite old sample, adding new info                            
        }

      }
      else {
	if(formerID>0) {
	  HOdigis->push_back(HO_old);
	}
	//save pointers for next iteration                                                                 
	formerID = currentID;
	HO_old = iHO->second;
	OldUpAdd = HO_old.id(); 
      }

      iHOchk = iHO;
      if((++iHOchk) == HODigiStorage_.end()) {  //make sure not to lose the last one                         
	HOdigis->push_back((iHO->second));
      }
    }

    // HF next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    HFCaloSample HF_old;

    HFDigiMap::const_iterator iHFchk;

    for(HFDigiMap::const_iterator iHF  = HFDigiStorage_.begin();
	iHF != HFDigiStorage_.end(); ++iHF) {

      currentID = iHF->first; 

      if (currentID == formerID) { // we have to add these digis together

        //loop over digi samples in each CaloSample                                                           
        uint sizenew = (iHF->second).size();
        uint sizeold = HF_old.size();

        uint max_samp = max(sizenew, sizeold);

        // samples from different events can be of different lengths - sum all                               
        // that overlap.                                                                                     

        for(uint isamp = 0; isamp<max_samp; isamp++) {
          if(isamp < sizenew) {
            fC_new = (iHF->second)[isamp].adc();
          }
          else { fC_new = 0;}

          if(isamp < sizeold) {
	    fC_old = HF_old[isamp].adc();
          }
          else { fC_old = 0;}

          // add values                                                                                      
          fC_sum = fC_new + fC_old;
	  fC_sum = min(fC_sum,127); //first 7 bits of (uint)                                             
          // replace right bits of sample with new ADC

          fC_sum = min(fC_sum,127); //first 7 bits of Sample                                             
          data = OldUpAdd + fC_sum;  //add new data to old address
          HF_old.setSample(isamp,data);  // overwrite old sample, adding new info                            
        }

      }
      else {
	if(formerID>0) {
	  HFdigis->push_back(HF_old);
	}
	//save pointers for next iteration                                                                 
	formerID = currentID;
	HF_old = iHF->second;
	OldUpAdd = HF_old.id(); // mask off adc information
      }

      iHFchk = iHF;
      if((++iHFchk) == HFDigiStorage_.end()) {  //make sure not to lose the last one                         
	HFdigis->push_back((iHF->second));
      }
    }

    // ZDC next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    ZDCCaloSample ZDC_old;

    ZDCDigiMap::const_iterator iZDCchk;

    for(ZDCDigiMap::const_iterator iZDC  = ZDCDigiStorage_.begin();
	iZDC != ZDCDigiStorage_.end(); ++iZDC) {

      currentID = iZDC->first; 

      if (currentID == formerID) { // we have to add these digis together

        //loop over digi samples in each CaloSample                                                           
        uint sizenew = (iZDC->second).size();
        uint sizeold = ZDC_old.size();

        uint max_samp = max(sizenew, sizeold);

        // samples from different events can be of different lengths - sum all                               
        // that overlap.                                                                                     

        for(uint isamp = 0; isamp<max_samp; isamp++) {
          if(isamp < sizenew) {
            fC_new = (iZDC->second)[isamp].adc();
          }
          else { fC_new = 0;}

          if(isamp < sizeold) {
	    fC_old = ZDC_old[isamp].adc();
          }
          else { fC_old = 0;}

          // add values                                                                                      
          fC_sum = fC_new + fC_old;
	  fC_sum = min(fC_sum,127); //first 7 bits of (uint)                                             
          // replace right bits of sample with new ADC

          fC_sum = min(fC_sum,127); //first 7 bits of Sample                                             
          data = OldUpAdd + fC_sum;  //add new data to old address
          ZDC_old.setSample(isamp,data);  // overwrite old sample, adding new info                            
        }

      }
      else {
	if(formerID>0) {
	  ZDCdigis->push_back(ZDC_old);
	}
	//save pointers for next iteration                                                                 
	formerID = currentID;
	ZDC_old = iZDC->second;
	OldUpAdd = ZDC_old.id(); // mask off adc information
      }

      iZDCchk = iZDC;
      if((++iZDCchk) == ZDCDigiStorage_.end()) {  //make sure not to lose the last one                         
	ZDCdigis->push_back((iZDC->second));
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
