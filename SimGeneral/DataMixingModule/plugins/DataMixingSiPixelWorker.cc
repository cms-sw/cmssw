// File: DataMixingSiPixelWorker.cc
// Description:  see DataMixingSiPixelWorker.h
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
#include "DataMixingSiPixelWorker.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingSiPixelWorker::DataMixingSiPixelWorker() { } 

  // Constructor 
  DataMixingSiPixelWorker::DataMixingSiPixelWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC) : 
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    pixeldigi_collectionSig_   = ps.getParameter<edm::InputTag>("pixeldigiCollectionSig");
    pixeldigi_collectionPile_   = ps.getParameter<edm::InputTag>("pixeldigiCollectionPile");
    PixelDigiCollectionDM_  = ps.getParameter<std::string>("PixelDigiCollectionDM");

    PixelDigiToken_ = iC.consumes<edm::DetSetVector<PixelDigi> >(pixeldigi_collectionSig_);
    PixelDigiPToken_ = iC.consumes<edm::DetSetVector<PixelDigi> >(pixeldigi_collectionPile_);


    // clear local storage for this event                                                                     
    SiHitStorage_.clear();

  }
	       

  // Virtual destructor needed.
  DataMixingSiPixelWorker::~DataMixingSiPixelWorker() { 
  }  



  void DataMixingSiPixelWorker::addSiPixelSignals(const edm::Event &e) { 
    // fill in maps of hits

    LogDebug("DataMixingSiPixelWorker")<<"===============> adding MC signals for "<<e.id();

    Handle< edm::DetSetVector<PixelDigi> >  input;

    if( e.getByToken(PixelDigiToken_,input) ) {

      //loop on all detsets (detectorIDs) inside the input collection
      edm::DetSetVector<PixelDigi>::const_iterator DSViter=input->begin();
      for (; DSViter!=input->end();DSViter++){

#ifdef DEBUG
	LogDebug("DataMixingSiPixelWorker")  << "Processing DetID " << DSViter->id;
#endif

	uint32_t detID = DSViter->id;
	edm::DetSet<PixelDigi>::const_iterator begin =(DSViter->data).begin();
	edm::DetSet<PixelDigi>::const_iterator end   =(DSViter->data).end();
	edm::DetSet<PixelDigi>::const_iterator icopy;
  
	OneDetectorMap LocalMap;

	for (icopy=begin; icopy!=end; icopy++) {
	  LocalMap.insert(OneDetectorMap::value_type( (icopy->channel()), *icopy ));
	}

	SiHitStorage_.insert( SiGlobalIndex::value_type( detID, LocalMap ) );
      }
 
    }    
  } // end of addSiPixelSignals



  void DataMixingSiPixelWorker::addSiPixelPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,
                                                  ModuleCallingContext const* mcc) {
  
    LogDebug("DataMixingSiPixelWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    boost::shared_ptr<Wrapper<edm::DetSetVector<PixelDigi> >  const> inputPTR =
      getProductByTag<edm::DetSetVector<PixelDigi> >(*ep, pixeldigi_collectionPile_, mcc);

    if(inputPTR ) {

      const edm::DetSetVector<PixelDigi>  *input = const_cast< edm::DetSetVector<PixelDigi> * >(inputPTR->product());



      //   Handle< edm::DetSetVector<PixelDigi> >  input;

      //   if( e->getByLabel(pixeldigi_collectionPile_,input) ) {

      //loop on all detsets (detectorIDs) inside the input collection
      edm::DetSetVector<PixelDigi>::const_iterator DSViter=input->begin();
      for (; DSViter!=input->end();DSViter++){

#ifdef DEBUG
	LogDebug("DataMixingSiPixelWorker")  << "Pileups: Processing DetID " << DSViter->id;
#endif

	uint32_t detID = DSViter->id;
	edm::DetSet<PixelDigi>::const_iterator begin =(DSViter->data).begin();
	edm::DetSet<PixelDigi>::const_iterator end   =(DSViter->data).end();
	edm::DetSet<PixelDigi>::const_iterator icopy;

	// find correct local map (or new one) for this detector ID

	SiGlobalIndex::const_iterator itest;

	itest = SiHitStorage_.find(detID);

	if(itest!=SiHitStorage_.end()) {  // this detID already has hits, add to existing map

	  OneDetectorMap LocalMap = itest->second;

	  // fill in local map with extra channels
	  for (icopy=begin; icopy!=end; icopy++) {
	    LocalMap.insert(OneDetectorMap::value_type( (icopy->channel()), *icopy ));
	  }

	  SiHitStorage_[detID]=LocalMap;
	  
	}
	else{ // fill local storage with this information, put in global collection

	  OneDetectorMap LocalMap;

	  for (icopy=begin; icopy!=end; icopy++) {
	    LocalMap.insert(OneDetectorMap::value_type( (icopy->channel()), *icopy ));
	  }

	  SiHitStorage_.insert( SiGlobalIndex::value_type( detID, LocalMap ) );
	}

      }
    }
  }


 
  void DataMixingSiPixelWorker::putSiPixel(edm::Event &e) {

    // collection of Digis to put in the event

    std::vector< edm::DetSet<PixelDigi> > vPixelDigi;

    // loop through our collection of detectors, merging hits and putting new ones in the output

    // big loop over Detector IDs:

    for(SiGlobalIndex::const_iterator IDet = SiHitStorage_.begin();
	IDet != SiHitStorage_.end(); IDet++) {

      edm::DetSet<PixelDigi> SPD(IDet->first); // Make empty collection with this detector ID
	
      OneDetectorMap LocalMap = IDet->second;

      //counter variables
      int formerPixel = -1;
      int currentPixel;
      int ADCSum = 0;


      OneDetectorMap::const_iterator iLocalchk;

      for(OneDetectorMap::const_iterator iLocal  = LocalMap.begin();
	  iLocal != LocalMap.end(); ++iLocal) {

	currentPixel = iLocal->first; 

	if (currentPixel == formerPixel) { // we have to add these digis together
	  ADCSum+=(iLocal->second).adc();
	}
	else{
	  if(formerPixel!=-1){             // ADC info stolen from SiStrips...
	    if (ADCSum > 511) ADCSum = 255;
	    else if (ADCSum > 253 && ADCSum < 512) ADCSum = 254;
	    PixelDigi aHit(formerPixel, ADCSum);
	    SPD.push_back( aHit );	  
	  }
	  // save pointers for next iteration
	  formerPixel = currentPixel;
	  ADCSum = (iLocal->second).adc();
	}

	iLocalchk = iLocal;
	if((++iLocalchk) == LocalMap.end()) {  //make sure not to lose the last one
	  if (ADCSum > 511) ADCSum = 255;
	  else if (ADCSum > 253 && ADCSum < 512) ADCSum = 254;
	  SPD.push_back( PixelDigi(formerPixel, ADCSum) );	  
	} 

      }// end of loop over one detector

      // stick this into the global vector of detector info
      vPixelDigi.push_back(SPD);

    } // end of big loop over all detector IDs

    // put the collection of digis in the event   
    LogInfo("DataMixingSiPixelWorker") << "total # Merged Pixels: " << vPixelDigi.size() ;

    // make new digi collection
    
    std::auto_ptr< edm::DetSetVector<PixelDigi> > MyPixelDigis(new edm::DetSetVector<PixelDigi>(vPixelDigi) );

    // put collection

    e.put( MyPixelDigis, PixelDigiCollectionDM_ );

    // clear local storage for this event
    SiHitStorage_.clear();
  }

} //edm
