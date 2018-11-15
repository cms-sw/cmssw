// File: DataMixingSiStripWorker.cc
// Description:  see DataMixingSiStripWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include <memory>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
//
//
#include "DataMixingSiStripWorker.h"

using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingSiStripWorker::DataMixingSiStripWorker() { }

  // Constructor 
  DataMixingSiStripWorker::DataMixingSiStripWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC) : 
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    SistripLabelSig_   = ps.getParameter<edm::InputTag>("SistripLabelSig");
    SiStripPileInputTag_ = ps.getParameter<edm::InputTag>("SiStripPileInputTag");

    SiStripDigiCollectionDM_  = ps.getParameter<std::string>("SiStripDigiCollectionDM");

    SiStripDigiToken_ = iC.consumes<edm::DetSetVector<SiStripDigi> >(SistripLabelSig_);
    SiStripDigiPToken_ = iC.consumes<edm::DetSetVector<SiStripDigi> >(SiStripPileInputTag_);



    // clear local storage for this event                                                                     
    SiHitStorage_.clear();

  }
	       

  // Virtual destructor needed.
  DataMixingSiStripWorker::~DataMixingSiStripWorker() { 
  }  



  void DataMixingSiStripWorker::addSiStripSignals(const edm::Event &e) { 
    // fill in maps of hits

    Handle< edm::DetSetVector<SiStripDigi> >  input;

    if( e.getByToken(SiStripDigiToken_,input) ) {
      OneDetectorMap LocalMap;

      //loop on all detsets (detectorIDs) inside the input collection
      edm::DetSetVector<SiStripDigi>::const_iterator DSViter=input->begin();
      for (; DSViter!=input->end();DSViter++){

#ifdef DEBUG
	LogDebug("DataMixingSiStripWorker")  << "Processing DetID " << DSViter->id;
#endif

	LocalMap.clear();
	LocalMap.reserve((DSViter->data).size());
	LocalMap.insert(LocalMap.end(),(DSViter->data).begin(),(DSViter->data).end());	
	
	SiHitStorage_.insert( SiGlobalIndex::value_type( DSViter->id, LocalMap ) );
      }
 
    }
  } // end of addSiStripSignals



  void DataMixingSiStripWorker::addSiStripPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,
                                                  ModuleCallingContext const* mcc) {
    LogDebug("DataMixingSiStripWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    std::shared_ptr<Wrapper<edm::DetSetVector<SiStripDigi> >  const> inputPTR =
      getProductByTag<edm::DetSetVector<SiStripDigi> >(*ep, SiStripPileInputTag_, mcc);

    if(inputPTR ) {

      const edm::DetSetVector<SiStripDigi>  *input = const_cast< edm::DetSetVector<SiStripDigi> * >(inputPTR->product());

      // Handle< edm::DetSetVector<SiStripDigi> >  input;

      // if( e->getByLabel(Sistripdigi_collectionPile_.label(),SistripLabelPile_.label(),input) ) {

      OneDetectorMap LocalMap;

      //loop on all detsets (detectorIDs) inside the input collection
      edm::DetSetVector<SiStripDigi>::const_iterator DSViter=input->begin();
      for (; DSViter!=input->end();DSViter++){

#ifdef DEBUG
	LogDebug("DataMixingSiStripWorker")  << "Pileups: Processing DetID " << DSViter->id;
#endif

	// find correct local map (or new one) for this detector ID

	SiGlobalIndex::const_iterator itest;

	itest = SiHitStorage_.find(DSViter->id);

	if(itest!=SiHitStorage_.end()) {  // this detID already has hits, add to existing map

	  LocalMap = itest->second;

	  // fill in local map with extra channels
	  LocalMap.insert(LocalMap.end(),(DSViter->data).begin(),(DSViter->data).end());
	  std::stable_sort(LocalMap.begin(),LocalMap.end(),DataMixingSiStripWorker::StrictWeakOrdering());
	  SiHitStorage_[DSViter->id]=LocalMap;
	  
	}
	else{ // fill local storage with this information, put in global collection

	  LocalMap.clear();
	  LocalMap.reserve((DSViter->data).size());
	  LocalMap.insert(LocalMap.end(),(DSViter->data).begin(),(DSViter->data).end());

	  SiHitStorage_.insert( SiGlobalIndex::value_type( DSViter->id, LocalMap ) );
	}
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
      int formerStrip = -1;
      int currentStrip;
      int ADCSum = 0;

      OneDetectorMap::const_iterator iLocalchk;
      OneDetectorMap::const_iterator iLocal  = LocalMap.begin();
      for(;iLocal != LocalMap.end(); ++iLocal) {

	currentStrip = iLocal->strip(); 

	if (currentStrip == formerStrip) { // we have to add these digis together
	  ADCSum+=iLocal->adc();          // on every element...
	}
	else{
	  if(formerStrip!=-1){
	    if (ADCSum > 511) ADCSum = 255;
	    else if (ADCSum > 253 && ADCSum < 512) ADCSum = 254;
	    SiStripDigi aHit(formerStrip, ADCSum);
	    SSD.push_back( aHit );	  
	  }
	  // save pointers for next iteration
	  formerStrip = currentStrip;
	  ADCSum = iLocal->adc();
	}

	iLocalchk = iLocal;
	if((++iLocalchk) == LocalMap.end()) {  //make sure not to lose the last one
	  if (ADCSum > 511) ADCSum = 255;
	  else if (ADCSum > 253 && ADCSum < 512) ADCSum = 254;
	  SSD.push_back( SiStripDigi(formerStrip, ADCSum) );	  
	} // end of loop over one detector
	
      }
      // stick this into the global vector of detector info
      vSiStripDigi.push_back(SSD);

    } // end of big loop over all detector IDs

    // put the collection of digis in the event   
    LogInfo("DataMixingSiStripWorker") << "total # Merged strips: " << vSiStripDigi.size() ;

    // make new digi collection
    
    std::unique_ptr< edm::DetSetVector<SiStripDigi> > MySiStripDigis(new edm::DetSetVector<SiStripDigi>(vSiStripDigi) );

    // put collection

    e.put(std::move(MySiStripDigis), SiStripDigiCollectionDM_ );

    // clear local storage for this event
    SiHitStorage_.clear();
  }

} //edm
