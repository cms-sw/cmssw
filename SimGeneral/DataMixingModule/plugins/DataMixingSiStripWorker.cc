// File: DataMixingSiStripWorker.cc
// Description:  see DataMixingSiStripWorker.h
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
#include "DataMixingSiStripWorker.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingSiStripWorker::DataMixingSiStripWorker() { }

  // Constructor 
  DataMixingSiStripWorker::DataMixingSiStripWorker(const edm::ParameterSet& ps) : 
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

    Sistripdigi_collection_   = ps.getParameter<edm::InputTag>("Sistripdigi_collection");
    SiStripDigiCollectionDM_  = ps.getParameter<std::string>("SiStripDigiCollectionDM");

  }
	       

  // Virtual destructor needed.
  DataMixingSiStripWorker::~DataMixingSiStripWorker() { 
    delete sel_;

  }  



  void DataMixingSiStripWorker::addSiStripSignals(const edm::Event &e) { 
    // fill in maps of hits

    LogDebug("DataMixingSiStripWorker")<<"===============> adding MC signals for "<<e.id();

    Handle< edm::DetSetVector<SiStripDigi> >  input;

    e.getByLabel(Sistripdigi_collection_,input);

    //loop on all detsets (detectorIDs) inside the input collection
    edm::DetSetVector<SiStripDigi>::const_iterator DSViter=input->begin();
    for (; DSViter!=input->end();DSViter++){

#ifdef DEBUG
      LogDebug("DataMixingSiStripWorker")  << "Processing DetID " << DSViter->id;
#endif

      uint32_t detID = DSViter->id;
      edm::DetSet<SiStripDigi>::const_iterator begin =(DSViter->data).begin();
      edm::DetSet<SiStripDigi>::const_iterator end   =(DSViter->data).end();
      edm::DetSet<SiStripDigi>::const_iterator icopy;
  
      OneDetectorMap LocalMap;

      for (icopy=begin; icopy!=end; icopy++) {
	LocalMap.insert(OneDetectorMap::value_type( (icopy->strip()), *icopy ));
      }

      SiHitStorage_.insert( SiGlobalIndex::value_type( detID, LocalMap ) );
    }
 
    
  } // end of addSiStripSignals



  void DataMixingSiStripWorker::addSiStripPileups(const int bcr, Event *e, unsigned int eventNr) {
  
    LogDebug("DataMixingSiStripWorker") <<"\n===============> adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    Handle< edm::DetSetVector<SiStripDigi> >  input;

    e->getByLabel(Sistripdigi_collection_,input);

    //loop on all detsets (detectorIDs) inside the input collection
    edm::DetSetVector<SiStripDigi>::const_iterator DSViter=input->begin();
    for (; DSViter!=input->end();DSViter++){

#ifdef DEBUG
      LogDebug("DataMixingSiStripWorker")  << "Pileups: Processing DetID " << DSViter->id;
#endif

      uint32_t detID = DSViter->id;
      edm::DetSet<SiStripDigi>::const_iterator begin =(DSViter->data).begin();
      edm::DetSet<SiStripDigi>::const_iterator end   =(DSViter->data).end();
      edm::DetSet<SiStripDigi>::const_iterator icopy;

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
	  LocalMap.insert(OneDetectorMap::value_type( (icopy->strip()), *icopy ));
	}

	SiHitStorage_.insert( SiGlobalIndex::value_type( detID, LocalMap ) );
      }

    }

  }


 
  void DataMixingSiStripWorker::putSiStrip(edm::Event &e) {

    // collection of Digis to put in the event

    std::vector< edm::DetSet<SiStripDigi> > vSiStripDigi;

    // loop through our collection of detectors, merging hits and putting new ones in the output

    // big loop over Detector IDs:

    for(SiGlobalIndex::const_iterator IDet = SiHitStorage_.begin();
	IDet != SiHitStorage_.end(); IDet++) {

      edm::DetSet<SiStripDigi> SSD(IDet->first); // Make empty collection with this detector ID
	
      OneDetectorMap LocalMap = IDet->second;

      //counter variables
      int formerStrip = 0;
      int currentStrip;
      int ADCSum = 0;
      SiStripDigi OldHit;
      int nmatch=0;

      OneDetectorMap::const_iterator iLocalchk;

      for(OneDetectorMap::const_iterator iLocal  = LocalMap.begin();
	  iLocal != LocalMap.end(); ++iLocal) {

	currentStrip = iLocal->first; 

	if (currentStrip == formerStrip) { // we have to add these digis together
	  nmatch++;                  // use this to avoid using the "count" function
	  ADCSum+=(iLocal->second).adc();          // on every element...

	  iLocalchk = iLocal;
	  if((iLocalchk++) == LocalMap.end()) {  //make sure not to lose the last one
	    SiStripDigi aHit(formerStrip, ADCSum);
	    SSD.push_back( aHit );	  
	    // reset adc sum, nmatch
	    ADCSum = 0 ;
	    nmatch=0;	  
	  }
	}
	else {
	  if(nmatch>0) {
	    SiStripDigi aHit(formerStrip, ADCSum);
	    SSD.push_back( aHit );	  
	    // reset adc sum, nmatch
	    ADCSum = 0 ;
	    nmatch=0;	  
	  }
	  else {
	    SSD.push_back( OldHit );
	  }
	
	  iLocalchk = iLocal;
	  if((iLocalchk++) == LocalMap.end()) {  //make sure not to lose the last one
	    SSD.push_back( iLocal->second );
	  }

	  // save pointers for next iteration
	  OldHit = iLocal->second;
	  formerStrip = currentStrip;
	  ADCSum = (iLocal->second).adc();
	}
      }  // end of loop over one detector

      // stick this into the global vector of detector info
      vSiStripDigi.push_back(SSD);

    } // end of big loop over all detector IDs

    // put the collection of digis in the event   
    LogInfo("DataMixingSiStripWorker") << "total # Merged strips: " << vSiStripDigi.size() ;

    // make new digi collection
    
    std::auto_ptr< edm::DetSetVector<SiStripDigi> > MySiStripDigis(new edm::DetSetVector<SiStripDigi>(vSiStripDigi) );

    // put collection

    e.put( MySiStripDigis, SiStripDigiCollectionDM_ );

    // clear local storage for this event
    SiHitStorage_.clear();
  }

} //edm
