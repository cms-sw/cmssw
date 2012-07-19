// File: DataMixingMuonWorker.cc
// Description:  see DataMixingMuonWorker.h
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
#include "DataMixingMuonWorker.h"


using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingMuonWorker::DataMixingMuonWorker() { } 

  // Constructor 
  DataMixingMuonWorker::DataMixingMuonWorker(const edm::ParameterSet& ps) : 
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // Declare the products to produce

    DTDigiTagSig_           = ps.getParameter<edm::InputTag>("DTDigiTagSig");
    RPCDigiTagSig_          = ps.getParameter<edm::InputTag>("RPCDigiTagSig");

    CSCstripdigi_collectionSig_   = ps.getParameter<edm::InputTag>("CSCstripdigiCollectionSig");
    CSCwiredigi_collectionSig_    = ps.getParameter<edm::InputTag>("CSCwiredigiCollectionSig");
    CSCCompdigi_collectionSig_    = ps.getParameter<edm::InputTag>("CSCCompdigiCollectionSig");

    DTPileInputTag_       = ps.getParameter<edm::InputTag>("DTPileInputTag");
    RPCPileInputTag_      = ps.getParameter<edm::InputTag>("RPCPileInputTag");
    CSCWirePileInputTag_  = ps.getParameter<edm::InputTag>("CSCWirePileInputTag");
    CSCStripPileInputTag_ = ps.getParameter<edm::InputTag>("CSCStripPileInputTag");
    CSCCompPileInputTag_  = ps.getParameter<edm::InputTag>("CSCCompPileInputTag");

    // outputs:

    DTDigiCollectionDM_  = ps.getParameter<std::string>("DTDigiCollectionDM");
    RPCDigiCollectionDM_ = ps.getParameter<std::string>("RPCDigiCollectionDM");
    CSCStripDigiCollectionDM_ = ps.getParameter<std::string>("CSCStripDigiCollectionDM");
    CSCWireDigiCollectionDM_  = ps.getParameter<std::string>("CSCWireDigiCollectionDM");
    CSCComparatorDigiCollectionDM_  = ps.getParameter<std::string>("CSCComparatorDigiCollectionDM");


  }
	       

  // Virtual destructor needed.
  DataMixingMuonWorker::~DataMixingMuonWorker() { 
  }  

  void DataMixingMuonWorker::addMuonSignals(const edm::Event &e) { 
    // fill in maps of hits

    LogDebug("DataMixingMuonWorker")<<"===============> adding MC signals for "<<e.id();

    // DT
    // 

    OurDTDigis_ = new DTDigiCollection();
    Handle<DTDigiCollection> pDTdigis; 

    // Get the digis from the event
    if( e.getByLabel(DTDigiTagSig_, pDTdigis) ) {

    //    LogInfo("DataMixingMuonWorker") << "total # DT Digis: " << DTdigis->size();

    // Loop over digis, copying them to our own local storage
      const DTDigiCollection* DTdigis = pDTdigis.product();
      DTDigiCollection::DigiRangeIterator DLayerIt;
      for (DLayerIt = DTdigis->begin(); DLayerIt != DTdigis->end(); ++DLayerIt) {
	// The layerId
	const DTLayerId& layerId = (*DLayerIt).first;

	// Get the iterators over the digis associated with this LayerId
	const DTDigiCollection::Range& range = (*DLayerIt).second;

	OurDTDigis_->put(range, layerId);
      }
    }
    // RPC
    // 

    OurRPCDigis_ = new RPCDigiCollection();

    // Get the digis from the event
    Handle<RPCDigiCollection> pRPCdigis; 

    if( e.getByLabel(RPCDigiTagSig_, pRPCdigis) ) {

    // Loop over digis, copying them to our own local storage

      const RPCDigiCollection* RPCdigis = pRPCdigis.product();
      RPCDigiCollection::DigiRangeIterator RLayerIt;
      for (RLayerIt = RPCdigis->begin(); RLayerIt != RPCdigis->end(); ++RLayerIt) {
	// The layerId
	const RPCDetId& layerId = (*RLayerIt).first;

	// Get the iterators over the digis associated with this LayerId
	const RPCDigiCollection::Range& range = (*RLayerIt).second;

	OurRPCDigis_->put(range, layerId);
      
      }
    }
    // CSCStrip
    // 

    OurCSCStripDigis_ = new CSCStripDigiCollection();

    // Get the digis from the event
    Handle<CSCStripDigiCollection> pCSCStripdigis; 

    if( e.getByLabel(CSCstripdigi_collectionSig_, pCSCStripdigis) ) {

    //if(pCSCStripdigis.isValid() ) { std::cout << "Signal: have CSCStripDigis" << std::endl;}
    //else { std::cout << "Signal: NO CSCStripDigis" << std::endl;}


    // Loop over digis, copying them to our own local storage

      const CSCStripDigiCollection* CSCStripdigis = pCSCStripdigis.product();
      CSCStripDigiCollection::DigiRangeIterator CSLayerIt;
      for (CSLayerIt = CSCStripdigis->begin(); CSLayerIt != CSCStripdigis->end(); ++CSLayerIt) {
	// The layerId
	const CSCDetId& layerId = (*CSLayerIt).first;

	// Get the iterators over the digis associated with this LayerId
	const CSCStripDigiCollection::Range& range = (*CSLayerIt).second;

	OurCSCStripDigis_->put(range, layerId);
      }
    }
    // CSCWire
    // 

    OurCSCWireDigis_ = new CSCWireDigiCollection();

    // Get the digis from the event
    Handle<CSCWireDigiCollection> pCSCWiredigis; 

    if( e.getByLabel(CSCwiredigi_collectionSig_, pCSCWiredigis) ) {
   

    //if(pCSCWiredigis.isValid() ) { std::cout << "Signal: have CSCWireDigis" << std::endl;}
    //else { std::cout << "Signal: NO CSCWireDigis" << std::endl;}
    
    // Loop over digis, copying them to our own local storage

      const CSCWireDigiCollection* CSCWiredigis = pCSCWiredigis.product();
      CSCWireDigiCollection::DigiRangeIterator CWLayerIt;
      for (CWLayerIt = CSCWiredigis->begin(); CWLayerIt != CSCWiredigis->end(); ++CWLayerIt) {
	// The layerId
	const CSCDetId& layerId = (*CWLayerIt).first;

	// Get the iterators over the digis associated with this LayerId
	const CSCWireDigiCollection::Range& range = (*CWLayerIt).second;

	OurCSCWireDigis_->put(range, layerId);
      
      }
    }

    // CSCComparators
    // 

    OurCSCComparatorDigis_ = new CSCComparatorDigiCollection();

    // Get the digis from the event
    Handle<CSCComparatorDigiCollection> pCSCComparatordigis; 

    //std::cout << "CSCComp label: " << CSCDigiTagSig_.label() << " " << CSCCompdigi_collectionSig_.label() << std::endl;

    if( e.getByLabel(CSCCompdigi_collectionSig_, pCSCComparatordigis) ) {
   

      //if(pCSCComparatordigis.isValid() ) { std::cout << "Signal: have CSCComparatorDigis" << std::endl;}
      //else { std::cout << "Signal: NO CSCComparatorDigis" << std::endl;}
    
    // Loop over digis, copying them to our own local storage

      const CSCComparatorDigiCollection* CSCComparatordigis = pCSCComparatordigis.product();
      CSCComparatorDigiCollection::DigiRangeIterator CWLayerIt;
      for (CWLayerIt = CSCComparatordigis->begin(); CWLayerIt != CSCComparatordigis->end(); ++CWLayerIt) {
	// The layerId
	const CSCDetId& layerId = (*CWLayerIt).first;

	// Get the iterators over the digis associated with this LayerId
	const CSCComparatorDigiCollection::Range& range = (*CWLayerIt).second;

	OurCSCComparatorDigis_->put(range, layerId);
      
      }
    }

    
  } // end of addMuonSignals

  void DataMixingMuonWorker::addMuonPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr) {
  
    LogDebug("DataMixingMuonWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // DT
    // 
    // Get the digis from the event

   boost::shared_ptr<Wrapper<DTDigiCollection>  const> DTDigisPTR = 
          getProductByTag<DTDigiCollection>(*ep, DTPileInputTag_ );
 
   if(DTDigisPTR ) {

     const DTDigiCollection*  DTDigis = const_cast< DTDigiCollection * >(DTDigisPTR->product());

     DTDigiCollection::DigiRangeIterator DTLayerIt;
     for (DTLayerIt = DTDigis->begin(); DTLayerIt != DTDigis->end(); ++DTLayerIt) {
	// The layerId
	const DTLayerId& layerId = (*DTLayerIt).first;

	// Get the iterators over the Digis associated with this LayerId
	const DTDigiCollection::Range& range = (*DTLayerIt).second;

	OurDTDigis_->put(range, layerId);
      
      }
    }
    // RPC
    // 

    // Get the digis from the event


   boost::shared_ptr<Wrapper<RPCDigiCollection>  const> RPCDigisPTR = 
          getProductByTag<RPCDigiCollection>(*ep, RPCPileInputTag_ );
 
   if(RPCDigisPTR ) {

     const RPCDigiCollection*  RPCDigis = const_cast< RPCDigiCollection * >(RPCDigisPTR->product());

     RPCDigiCollection::DigiRangeIterator RPCLayerIt;
     for (RPCLayerIt = RPCDigis->begin(); RPCLayerIt != RPCDigis->end(); ++RPCLayerIt) {
	// The layerId
	const RPCDetId& layerId = (*RPCLayerIt).first;

	// Get the iterators over the digis associated with this LayerId
	const RPCDigiCollection::Range& range = (*RPCLayerIt).second;

	OurRPCDigis_->put(range, layerId);
      
      }
    }

    // CSCStrip
    // 

    // Get the digis from the event

   boost::shared_ptr<Wrapper<CSCStripDigiCollection>  const> CSCStripDigisPTR = 
          getProductByTag<CSCStripDigiCollection>(*ep, CSCStripPileInputTag_ );
 
   if(CSCStripDigisPTR ) {

     const CSCStripDigiCollection*  CSCStripDigis = const_cast< CSCStripDigiCollection * >(CSCStripDigisPTR->product());

     CSCStripDigiCollection::DigiRangeIterator CSCStripLayerIt;
     for (CSCStripLayerIt = CSCStripDigis->begin(); CSCStripLayerIt != CSCStripDigis->end(); ++CSCStripLayerIt) {
	// The layerId
	const CSCDetId& layerId = (*CSCStripLayerIt).first;

	// Get the iterators over the digis associated with this LayerId
	const CSCStripDigiCollection::Range& range = (*CSCStripLayerIt).second;

	OurCSCStripDigis_->put(range, layerId);
      
      }
    }

    // CSCWire
    // 

    // Get the digis from the event

   boost::shared_ptr<Wrapper<CSCWireDigiCollection>  const> CSCWireDigisPTR = 
          getProductByTag<CSCWireDigiCollection>(*ep, CSCWirePileInputTag_ );
 
   if(CSCWireDigisPTR ) {

     const CSCWireDigiCollection*  CSCWireDigis = const_cast< CSCWireDigiCollection * >(CSCWireDigisPTR->product());

     CSCWireDigiCollection::DigiRangeIterator CSCWireLayerIt;
     for (CSCWireLayerIt = CSCWireDigis->begin(); CSCWireLayerIt != CSCWireDigis->end(); ++CSCWireLayerIt) {
	// The layerId
	const CSCDetId& layerId = (*CSCWireLayerIt).first;

	// Get the iterators over the digis associated with this LayerId
	const CSCWireDigiCollection::Range& range = (*CSCWireLayerIt).second;

	OurCSCWireDigis_->put(range, layerId);
      
      }
    }

   // CSCComparators
   //

   // Get the digis from the event

   boost::shared_ptr<Wrapper<CSCComparatorDigiCollection>  const> CSCComparatorDigisPTR =
     getProductByTag<CSCComparatorDigiCollection>(*ep, CSCCompPileInputTag_ );

   if(CSCComparatorDigisPTR ) {

     const CSCComparatorDigiCollection*  CSCComparatorDigis = const_cast< CSCComparatorDigiCollection * >(CSCComparatorDigisPTR->product());

     CSCComparatorDigiCollection::DigiRangeIterator CSCComparatorLayerIt;
     for (CSCComparatorLayerIt = CSCComparatorDigis->begin(); CSCComparatorLayerIt != CSCComparatorDigis->end(); ++CSCComparatorLayerIt) {
       // The layerId
       const CSCDetId& layerId = (*CSCComparatorLayerIt).first;

       // Get the iterators over the digis associated with this LayerId
       const CSCComparatorDigiCollection::Range& range = (*CSCComparatorLayerIt).second;

       OurCSCComparatorDigis_->put(range, layerId);

     }
   }


  }
 
  void DataMixingMuonWorker::putMuon(edm::Event &e) {

    // collections of digis to put in the event
    std::auto_ptr< DTDigiCollection > DTDigiMerge( new DTDigiCollection );
    std::auto_ptr< RPCDigiCollection > RPCDigiMerge( new RPCDigiCollection );
    std::auto_ptr< CSCStripDigiCollection > CSCStripDigiMerge( new CSCStripDigiCollection );
    std::auto_ptr< CSCWireDigiCollection > CSCWireDigiMerge( new CSCWireDigiCollection );
    std::auto_ptr< CSCComparatorDigiCollection > CSCComparatorDigiMerge( new CSCComparatorDigiCollection );

    // Loop over DT digis, copying them from our own local storage

    DTDigiCollection::DigiRangeIterator DLayerIt;
    for (DLayerIt = OurDTDigis_->begin(); DLayerIt != OurDTDigis_->end(); ++DLayerIt) {
      // The layerId
      const DTLayerId& layerId = (*DLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const DTDigiCollection::Range& range = (*DLayerIt).second;

      DTDigiMerge->put(range, layerId);
      
    }

    // Loop over RPC digis, copying them from our own local storage

    RPCDigiCollection::DigiRangeIterator RLayerIt;
    for (RLayerIt = OurRPCDigis_->begin(); RLayerIt != OurRPCDigis_->end(); ++RLayerIt) {
      // The layerId
      const RPCDetId& layerId = (*RLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const RPCDigiCollection::Range& range = (*RLayerIt).second;

      RPCDigiMerge->put(range, layerId);
      
    }
    // Loop over CSCStrip digis, copying them from our own local storage

    CSCStripDigiCollection::DigiRangeIterator CSLayerIt;
    for (CSLayerIt = OurCSCStripDigis_->begin(); CSLayerIt != OurCSCStripDigis_->end(); ++CSLayerIt) {
      // The layerId
      const CSCDetId& layerId = (*CSLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const CSCStripDigiCollection::Range& range = (*CSLayerIt).second;

      CSCStripDigiMerge->put(range, layerId);
      
    }
    // Loop over CSCStrip digis, copying them from our own local storage

    CSCWireDigiCollection::DigiRangeIterator CWLayerIt;
    for (CWLayerIt = OurCSCWireDigis_->begin(); CWLayerIt != OurCSCWireDigis_->end(); ++CWLayerIt) {
      // The layerId
      const CSCDetId& layerId = (*CWLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const CSCWireDigiCollection::Range& range = (*CWLayerIt).second;

      CSCWireDigiMerge->put(range, layerId);
      
    }

    // Loop over CSCComparator digis, copying them from our own local storage

    CSCComparatorDigiCollection::DigiRangeIterator CCLayerIt;
    for (CCLayerIt = OurCSCComparatorDigis_->begin(); CCLayerIt != OurCSCComparatorDigis_->end(); ++CCLayerIt) {
      // The layerId
      const CSCDetId& layerId = (*CCLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const CSCComparatorDigiCollection::Range& range = (*CCLayerIt).second;

      CSCComparatorDigiMerge->put(range, layerId);
      
    }


    // put the collection of recunstructed hits in the event   
    //    LogDebug("DataMixingMuonWorker") << "total # DT Merged Digis: " << DTDigiMerge->size() ;
    //    LogDebug("DataMixingMuonWorker") << "total # RPC Merged Digis: " << RPCDigiMerge->size() ;
    //    LogDebug("DataMixingMuonWorker") << "total # CSCStrip Merged Digis: " << CSCStripDigiMerge->size() ;
    //    LogDebug("DataMixingMuonWorker") << "total # CSCWire Merged Digis: " << CSCWireDigiMerge->size() ;

    e.put( DTDigiMerge );
    e.put( RPCDigiMerge );
    e.put( CSCStripDigiMerge, CSCStripDigiCollectionDM_ );
    e.put( CSCWireDigiMerge, CSCWireDigiCollectionDM_ );
    e.put( CSCComparatorDigiMerge, CSCComparatorDigiCollectionDM_ );

    // clear local storage for this event
    delete OurDTDigis_;
    delete OurRPCDigis_;
    delete OurCSCStripDigis_;
    delete OurCSCWireDigis_;
    delete OurCSCComparatorDigis_;

  }

} //edm
