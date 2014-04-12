// File: DataMixingSiStripRawWorker.cc
// Description:  see DataMixingSiStripRawWorker.h
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
#include "DataMixingSiStripRawWorker.h"

using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingSiStripRawWorker::DataMixingSiStripRawWorker() { }

  // Constructor 
  DataMixingSiStripRawWorker::DataMixingSiStripRawWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC) : 
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    Sistripdigi_collectionSig_   = ps.getParameter<edm::InputTag>("SistripdigiCollectionSig");
    SistripLabelSig_   = ps.getParameter<edm::InputTag>("SistripLabelSig");

    SiStripPileInputTag_ = ps.getParameter<edm::InputTag>("SiStripPileInputTag");
    SiStripRawInputTag_ = ps.getParameter<edm::InputTag>("SiStripRawInputTag");

    SiStripDigiCollectionDM_  = ps.getParameter<std::string>("SiStripDigiCollectionDM");

    SiStripRawDigiSource_ = ps.getParameter<std::string>("SiStripRawDigiSource");

    // clear local storage for this event                                                                     
    SiHitStorage_.clear();
    
    edm::InputTag tag = edm::InputTag(Sistripdigi_collectionSig_.label(),SistripLabelSig_.label());

    SiStripInputTok_ = iC.consumes< edm::DetSetVector<SiStripDigi> >(tag);
    SiStripRawInputTok_ = iC.consumes< edm::DetSetVector<SiStripRawDigi> >(SiStripRawInputTag_);

  }
	       

  // Virtual destructor needed.
  DataMixingSiStripRawWorker::~DataMixingSiStripRawWorker() { 
  }  



  void DataMixingSiStripRawWorker::addSiStripSignals(const edm::Event &e) { 

    
    edm::Handle< edm::DetSetVector<SiStripDigi> >      hSSD;
    edm::Handle< edm::DetSetVector<SiStripRawDigi> >   hSSRD;
    
    if (SiStripRawDigiSource_=="SIGNAL") {
      e.getByToken(SiStripRawInputTok_,hSSRD);
      rawdigicollection_ = hSSRD.product();
    } else if (SiStripRawDigiSource_=="PILEUP") {
      e.getByToken(SiStripInputTok_,hSSD);
      digicollection_ =  hSSD.product();
    } else {
      std::cout << "you shouldn't be here" << std::endl;
    }
    

  } // end of addSiStripSignals



  void DataMixingSiStripRawWorker::addSiStripPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,
                                                     ModuleCallingContext const* mcc) {
    
    LogDebug("DataMixingSiStripRawWorker") << "\n===============> adding pileups from event  "
					   << ep->id() << " for bunchcrossing " << bcr;

    boost::shared_ptr<Wrapper<edm::DetSetVector<SiStripDigi> > const>    pSSD;
    boost::shared_ptr<Wrapper<edm::DetSetVector<SiStripRawDigi> > const> pSSRD;
    
    if (SiStripRawDigiSource_=="SIGNAL") {
      pSSD = getProductByTag<edm::DetSetVector<SiStripDigi> >(*ep, SiStripPileInputTag_, mcc);
      digicollection_ = const_cast< edm::DetSetVector<SiStripDigi> * >(pSSD->product());
    } else if (SiStripRawDigiSource_=="PILEUP") {
      pSSRD = getProductByTag<edm::DetSetVector<SiStripRawDigi> >(*ep, SiStripRawInputTag_, mcc);
      rawdigicollection_ = const_cast< edm::DetSetVector<SiStripRawDigi> * >(pSSRD->product());
    } else {
      std::cout << "you shouldn't be here" << std::endl;
    }

  } // end of addSiStripPileups

 
  void DataMixingSiStripRawWorker::putSiStrip(edm::Event &e) {


    //------------------
    //  (1) Fill a map from the Digi collection
    //

    // fill in maps of SiStripDigis
    OneDetectorMap LocalMap;
    
    //loop on all detsets (detectorIDs) inside the input collection
    edm::DetSetVector<SiStripDigi>::const_iterator DSViter=digicollection_->begin();
    for (; DSViter!=digicollection_->end();DSViter++){
      
#ifdef DEBUG
      LogDebug("DataMixingSiStripRawWorker")  << "Processing DetID " << DSViter->id;
#endif
      
      LocalMap.clear();
      LocalMap.reserve((DSViter->data).size());
      LocalMap.insert(LocalMap.end(),(DSViter->data).begin(),(DSViter->data).end());	
      
      SiHitStorage_.insert( SiGlobalIndex::value_type( DSViter->id, LocalMap ) );
    }


    //------------------
    //  (2) Loop over the input RawDigi collection and add the Digis from the map
    //

    // collection of RawDigis to put back in the event
    std::vector< edm::DetSet<SiStripRawDigi> > vSiStripRawDigi;

    //loop on all detsets (detectorIDs) inside the SiStripRawDigis collection
    edm::DetSetVector<SiStripRawDigi>::const_iterator rawDSViter=rawdigicollection_->begin();
    for (; rawDSViter!=rawdigicollection_->end();rawDSViter++){

      // Make empty collection with this detID
      edm::DetSet<SiStripRawDigi> SSRD(rawDSViter->id); 

      // find local map (if it exists) for this detector ID
      SiGlobalIndex::const_iterator itest;
      itest = SiHitStorage_.find(rawDSViter->id);

      // if detID already has digis in existing map, add them to rawdigis
      if(itest!=SiHitStorage_.end()) {  

#ifdef DEBUG
	LogDebug("DataMixingSiStripRawWorker")  << "Pileups: Processing DetID " << rawDSViter->id;
#endif

	// get the map from storage
	LocalMap = itest->second;
	OneDetectorMap::const_iterator iLocal  = LocalMap.begin();

	// loop on all strips in rawdigi detset
	int currentstrip=0;
	edm::DetSet<SiStripRawDigi>::const_iterator iRawDigi = rawDSViter->begin();
	while( iRawDigi != rawDSViter->end() ) {

	  int ADCSum = iRawDigi->adc();

	  // if current strip exists in map, add ADC values
	  if(iLocal->strip() == currentstrip) {
	    ADCSum += iLocal->adc();
	    iLocal++;
	  }

	  // put ADC sum in DetSet and go to next strip
	  SSRD.push_back( SiStripRawDigi(ADCSum) );
	  iRawDigi++;
	  currentstrip++;

	}

	// copy combined digi+rawdigi into rawdigi DetSetVector
	vSiStripRawDigi.push_back(SSRD);

      // otherwise, just copy the rawdigis from the background event to the output
      } else {
	vSiStripRawDigi.push_back(*rawDSViter);
      }

    }


    //------------------
    //  (3) Put the new RawDigi collection back into the event
    //

    // make new raw digi collection
    std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > MySiStripRawDigis(new edm::DetSetVector<SiStripRawDigi>(vSiStripRawDigi) );

    // put collection
    e.put( MySiStripRawDigis, SiStripDigiCollectionDM_ );

    // clear local storage for this event
    SiHitStorage_.clear();
  }

}
