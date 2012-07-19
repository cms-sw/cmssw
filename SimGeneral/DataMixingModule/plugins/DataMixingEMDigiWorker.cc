// File: DataMixingEMDigiWorker.cc
// Description:  see DataMixingEMDigiWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include <cmath>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"

//
//
#include "DataMixingEMDigiWorker.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingEMDigiWorker::DataMixingEMDigiWorker() { }

  // Constructor 
  DataMixingEMDigiWorker::DataMixingEMDigiWorker(const edm::ParameterSet& ps) : 
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce, retrieve

    EBProducerSig_ = ps.getParameter<edm::InputTag>("EBdigiProducerSig");
    EEProducerSig_ = ps.getParameter<edm::InputTag>("EEdigiProducerSig");
    ESProducerSig_ = ps.getParameter<edm::InputTag>("ESdigiProducerSig");

    EBPileInputTag_ = ps.getParameter<edm::InputTag>("EBPileInputTag");
    EEPileInputTag_ = ps.getParameter<edm::InputTag>("EEPileInputTag");
    ESPileInputTag_ = ps.getParameter<edm::InputTag>("ESPileInputTag");

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
  }  

  void DataMixingEMDigiWorker::addEMSignals(const edm::Event &e,const edm::EventSetup& ES) { 
    // fill in maps of hits

    LogInfo("DataMixingEMDigiWorker")<<"===============> adding MC signals for "<<e.id();

    // EB first

   Handle< EBDigiCollection > pEBDigis;

   const EBDigiCollection*  EBDigis = 0;

   if(e.getByLabel(EBProducerSig_, pEBDigis) ){
     EBDigis = pEBDigis.product(); // get a ptr to the product
     LogDebug("DataMixingEMDigiWorker") << "total # EB digis: " << EBDigis->size();
   }
   else { std::cout << "NO EBDigis! " << EBProducerSig_.label() << " " << EBdigiCollectionSig_.label() << std::endl;}
 
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

   
   if(e.getByLabel(EEProducerSig_, pEEDigis) ){
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

   
   if(e.getByLabel( ESProducerSig_, pESDigis) ){
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

  void DataMixingEMDigiWorker::addEMPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr, const edm::EventSetup& ES) {
  
    LogInfo("DataMixingEMDigiWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // EB first

    boost::shared_ptr<Wrapper<EBDigiCollection>  const> EBDigisPTR = 
          getProductByTag<EBDigiCollection>(*ep, EBPileInputTag_ );
 
   if(EBDigisPTR ) {

     const EBDigiCollection*  EBDigis = const_cast< EBDigiCollection * >(EBDigisPTR->product());

     LogDebug("DataMixingEMDigiWorker") << "total # EB digis: " << EBDigis->size();

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

    boost::shared_ptr<Wrapper<EEDigiCollection>  const> EEDigisPTR =
          getProductByTag<EEDigiCollection>(*ep, EEPileInputTag_ );

    if(EEDigisPTR ) {

     const EEDigiCollection*  EEDigis = const_cast< EEDigiCollection * >(EEDigisPTR->product()); 

     LogDebug("DataMixingEMDigiWorker") << "total # EE digis: " << EEDigis->size();

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

    boost::shared_ptr<Wrapper<ESDigiCollection>  const> ESDigisPTR =
      getProductByTag<ESDigiCollection>(*ep, ESPileInputTag_ );

    if(ESDigisPTR ) {

      const ESDigiCollection*  ESDigis = const_cast< ESDigiCollection * >(ESDigisPTR->product());

      LogDebug("DataMixingEMDigiWorker") << "total # ES digis: " << ESDigis->size();

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
	/*	
	cout<< " Adding signals " << EBDetId(currentID).ieta() << " " 
	                          << EBDetId(currentID).iphi() << std::endl;

	cout << 1 << " " ; 
	for (int i=0; i<10;++i)  std::cout << EB_old[i].adc()<< "["<<EB_old[i].gainId()<< "] " ; std::cout << std::endl;
 
	cout << 2 << " " ; 
	for (int i=0; i<10;++i)  std::cout << (iEB->second)[i].adc()<< "["<<(iEB->second)[i].gainId()<< "] " ; std::cout << std::endl;
	*/
	//loop over digi samples in each DataFrame
	unsigned int sizenew = (iEB->second).size();
	unsigned int sizeold = EB_old.size();

	unsigned int max_samp = std::max(sizenew, sizeold);

	// samples from different events can be of different lengths - sum all
	// that overlap.
	// check to see if gains match - if not, scale smaller cell down.

	int sw_gain_consensus=0;


	for(unsigned int isamp = 0; isamp<max_samp; isamp++) {
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

	  const std::vector<float> pedeStals = GetPedestals(ES,currentID);
	  const std::vector<float> gainRatios = GetGainRatios(ES,currentID);

	  if(adc_new>0 && adc_old>0) {
	    if(gain_old == gain_new) {  // we're happy - easy case
	      gain_consensus = gain_old;
	    }
	    else {  // lower gain sample has more energy
	     	      	      

	      if(gain_old < gain_new) { // old has higher gain than new, scale to lower gain
		
	
		float ratio = gainRatios[gain_new-1]/gainRatios[gain_old-1];
		adc_old = (int) round ((adc_old - pedeStals[gain_old-1]) / ratio + pedeStals[gain_new-1] );  
		gain_consensus = gain_new;
	      }
	      else { // scale to old (lower) gain
		float ratio = gainRatios[gain_old-1]/gainRatios[gain_new-1];
		adc_new = (int) round ( (adc_new - pedeStals[gain_new-1]) / ratio+ pedeStals[gain_old-1] );
		gain_consensus = gain_old;
	      } 
	    }
	  }

	 
	  // add values, but don't count pedestals twice
	  adc_sum = adc_new + adc_old - (int) round (pedeStals[gain_consensus-1]);


	  // if we are now saturating that gain, switch to the next
	  if (adc_sum> 4096) {
	    if (gain_consensus<3){

	      double ratio = gainRatios[gain_consensus]/gainRatios[gain_consensus-1];
	      adc_sum = (int) round ((adc_sum - pedeStals[gain_consensus-1])/ ratio + pedeStals[gain_consensus]  )  ;
	      sw_gain_consensus=++gain_consensus;	      
	    }
	    else adc_sum = 4096;
		
	  } 

	  // furthermore, make sure we don't decrease our gain once we've switched up
	  // in case go back 
	  if (gain_consensus<sw_gain_consensus){

	    double ratio = gainRatios[sw_gain_consensus-1]/gainRatios[gain_consensus-1];
 	    adc_sum = (int) round((adc_sum - pedeStals[gain_consensus-1] )/ratio + pedeStals[sw_gain_consensus-1]);
	    gain_consensus = sw_gain_consensus;
	  }

	  EcalMGPASample sample(adc_sum, gain_consensus);
	  EB_old.setSample(isamp,sample);  // overwrite old sample, adding new info
	} // for sample


      } // if current = former
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
	unsigned int sizenew = (iEE->second).size();
	unsigned int sizeold = EE_old.size();

	unsigned int max_samp = std::max(sizenew, sizeold);

	// samples from different events can be of different lengths - sum all
	// that overlap.
	// check to see if gains match - if not, scale smaller cell down.

	for(unsigned int isamp = 0; isamp<max_samp; isamp++) {
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

	  const std::vector<float> pedeStals = GetPedestals(ES,currentID);
	  const std::vector<float> gainRatios = GetGainRatios(ES,currentID);

	  if(adc_new>0 && adc_old>0) {
	    if(gain_old == gain_new) {  // we're happy - easy case
	      gain_consensus = gain_old;
	    }
	    else {  // lower gain sample has more energy

	      if(gain_old < gain_new) { // old has higher gain than new, scale to lower gain
		
		
		float ratio = gainRatios[gain_new-1]/gainRatios[gain_old-1];
		adc_old = (int) round ((adc_old - pedeStals[gain_old-1]) / ratio + pedeStals[gain_new-1] );  
		gain_consensus = gain_new;
	      }
	      else { // scale to old (lower) gain
		float ratio = gainRatios[gain_old-1]/gainRatios[gain_new-1];
		adc_new = (int) round ( (adc_new - pedeStals[gain_new-1]) / ratio+ pedeStals[gain_old-1] );
		gain_consensus = gain_old;
	      } 
	    }
	    
          }
	     

	  // add values
	  adc_sum = adc_new + adc_old;
	  
	  // if the sum saturates this gain, switch
	  if (adc_sum> 4096) {
	    if (gain_consensus<3){
	      
	      double ratio = gainRatios[gain_consensus]/gainRatios[gain_consensus-1];
	      adc_sum = (int) round ((adc_sum - pedeStals[gain_consensus-1])/ ratio + pedeStals[gain_consensus]  )  ;
	      ++gain_consensus;
	    }
	    else adc_sum = 4096;
	    
	  } 
	  
	  EcalMGPASample sample(adc_sum, gain_consensus);
	  EE_old.setSample(isamp,sample);
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
	unsigned int sizenew = (iES->second).size();
	unsigned int sizeold = ES_old.size();
	uint16_t rawdat = 0;
	unsigned int max_samp = std::max(sizenew, sizeold);

	// samples from different events can be of different lengths - sum all
	// that overlap.
	// check to see if gains match - if not, scale smaller cell down.

	for(unsigned int isamp = 0; isamp<max_samp; isamp++) {
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
	  adc_sum = std::min(adc_sum,4095); //first 12 bits of (uint)
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
  const std::vector<float>  DataMixingEMDigiWorker::GetPedestals (const edm::EventSetup& ES, const DetId& detid) {
    
    std::vector<float> pedeStals(3);

    // get pedestals
    edm::ESHandle<EcalPedestals> pedHandle;
    ES.get<EcalPedestalsRcd>().get( pedHandle );
    
    
    const EcalPedestalsMap & pedMap = pedHandle.product()->getMap(); // map of pedestals
    EcalPedestalsMapIterator pedIter; // pedestal iterator
    EcalPedestals::Item aped; // pedestal object for a single xtal
    
    
    pedIter = pedMap.find(detid);
    if( pedIter != pedMap.end() ) {
      aped = (*pedIter);
      pedeStals[0] = aped.mean_x12;
      pedeStals[1] = aped.mean_x6;
      pedeStals[2] = aped.mean_x1;
    } else {
      edm::LogError("DataMixingMissingInput") << "Cannot find pedestals";  
      pedeStals[0] = 0;
      pedeStals[1] = 0;
      pedeStals[2] = 0;
    }
    
    
    return pedeStals;
  }

  const std::vector<float>  DataMixingEMDigiWorker::GetGainRatios(const edm::EventSetup& ES, const DetId& detid) {

    std::vector<float> gainRatios(3);
    // get gain ratios  
    edm::ESHandle<EcalGainRatios> grHandle;
    ES.get<EcalGainRatiosRcd>().get(grHandle);
    EcalMGPAGainRatio theRatio= (*grHandle)[detid];
    
    
    gainRatios[0] = 1.;
    gainRatios[1] = theRatio.gain12Over6();
    gainRatios[2] = theRatio.gain6Over1()  * theRatio.gain12Over6();

    return gainRatios;
  }


} //edm
