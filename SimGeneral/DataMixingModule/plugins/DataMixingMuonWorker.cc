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

    // create input selector
    if (label_.size()>0){
      sel_=new Selector( ModuleLabelSelector(label_));
    }
    else {
      sel_=new Selector( MatchAllSelector());
    }

    // declare the products to produce

    DTdigi_collection_   = ps.getParameter<edm::InputTag>("DTdigi_collection");
    RPCdigi_collection_  = ps.getParameter<edm::InputTag>("RPCdigi_collection");
    CSCstripdigi_collection_   = ps.getParameter<edm::InputTag>("CSCstripdigi_collection");
    CSCwiredigi_collection_    = ps.getParameter<edm::InputTag>("CSCwiredigi_collection");


    DTDigiCollectionDM_  = ps.getParameter<std::string>("DTDigiCollectionDM");
    RPCDigiCollectionDM_ = ps.getParameter<std::string>("RPCDigiCollectionDM");
    CSCStripDigiCollectionDM_ = ps.getParameter<std::string>("CSCStripDigiCollectionDM");
    CSCWireDigiCollectionDM_  = ps.getParameter<std::string>("CSCWireDigiCollectionDM");


  }
	       

  // Virtual destructor needed.
  DataMixingMuonWorker::~DataMixingMuonWorker() { 
    delete sel_;

  }  

  void DataMixingMuonWorker::addMuonSignals(const edm::Event &e) { 
    // fill in maps of hits

    LogDebug("DataMixingMuonWorker")<<"===============> adding MC signals for "<<e.id();

    // DT
    // 

    OurDTDigis_ = new DTDigiCollection();

    // Get the digis from the event
    Handle<DTDigiCollection> DTdigis; 
    e.getByLabel(DTdigi_collection_, DTdigis);

    // Loop over digis, copying them to our own local storage

    DTDigiCollection::DigiRangeIterator DLayerIt;
    for (DLayerIt = DTdigis->begin(); DLayerIt != DTdigis->end(); ++DLayerIt) {
      // The layerId
      const DTLayerId& layerId = (*DLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const DTDigiCollection::Range& range = (*DLayerIt).second;

      OurDTDigis_->put(range, layerId);
      
    }
    // RPC
    // 

    OurRPCDigis_ = new RPCDigiCollection();

    // Get the digis from the event
    Handle<RPCDigiCollection> RPCdigis; 
    e.getByLabel(RPCdigi_collection_, RPCdigis);

    // Loop over digis, copying them to our own local storage

    RPCDigiCollection::DigiRangeIterator RLayerIt;
    for (RLayerIt = RPCdigis->begin(); RLayerIt != RPCdigis->end(); ++RLayerIt) {
      // The layerId
      const RPCDetId& layerId = (*RLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const RPCDigiCollection::Range& range = (*RLayerIt).second;

      OurRPCDigis_->put(range, layerId);
      
    }
    // CSCStrip
    // 

    OurCSCStripDigis_ = new CSCStripDigiCollection();

    // Get the digis from the event
    Handle<CSCStripDigiCollection> CSCStripdigis; 
    e.getByLabel(CSCstripdigi_collection_, CSCStripdigis);

    // Loop over digis, copying them to our own local storage

    CSCStripDigiCollection::DigiRangeIterator CSLayerIt;
    for (CSLayerIt = CSCStripdigis->begin(); CSLayerIt != CSCStripdigis->end(); ++CSLayerIt) {
      // The layerId
      const CSCDetId& layerId = (*CSLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const CSCStripDigiCollection::Range& range = (*CSLayerIt).second;

      OurCSCStripDigis_->put(range, layerId);
      
    }
    // CSCWire
    // 

    OurCSCWireDigis_ = new CSCWireDigiCollection();

    // Get the digis from the event
    Handle<CSCWireDigiCollection> CSCWiredigis; 
    e.getByLabel(CSCwiredigi_collection_, CSCWiredigis);

    // Loop over digis, copying them to our own local storage

    CSCWireDigiCollection::DigiRangeIterator CWLayerIt;
    for (CWLayerIt = CSCWiredigis->begin(); CWLayerIt != CSCWiredigis->end(); ++CWLayerIt) {
      // The layerId
      const CSCDetId& layerId = (*CWLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const CSCWireDigiCollection::Range& range = (*CWLayerIt).second;

      OurCSCWireDigis_->put(range, layerId);
      
    }

    
  } // end of addMuonSignals

  void DataMixingMuonWorker::addMuonPileups(const int bcr, Event *e, unsigned int eventNr) {
  
    LogDebug("DataMixingMuonWorker") <<"\n===============> adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // DT
    // 

    // Get the digis from the event
    Handle<DTDigiCollection> DTdigis; 
    e->getByLabel(DTdigi_collection_, DTdigis);

    // Loop over digis, copying them to our own local storage

    DTDigiCollection::DigiRangeIterator DLayerIt;
    for (DLayerIt = DTdigis->begin(); DLayerIt != DTdigis->end(); ++DLayerIt) {
      // The layerId
      const DTLayerId& layerId = (*DLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const DTDigiCollection::Range& range = (*DLayerIt).second;

      OurDTDigis_->put(range, layerId);
      
    }
    // RPC
    // 

    // Get the digis from the event
    Handle<RPCDigiCollection> RPCdigis; 
    e->getByLabel(RPCdigi_collection_, RPCdigis);

    // Loop over digis, copying them to our own local storage

    RPCDigiCollection::DigiRangeIterator RLayerIt;
    for (RLayerIt = RPCdigis->begin(); RLayerIt != RPCdigis->end(); ++RLayerIt) {
      // The layerId
      const RPCDetId& layerId = (*RLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const RPCDigiCollection::Range& range = (*RLayerIt).second;

      OurRPCDigis_->put(range, layerId);
      
    }
    // CSCStrip
    // 

    // Get the digis from the event
    Handle<CSCStripDigiCollection> CSCStripdigis; 
    e->getByLabel(CSCstripdigi_collection_, CSCStripdigis);

    // Loop over digis, copying them to our own local storage

    CSCStripDigiCollection::DigiRangeIterator CSLayerIt;
    for (CSLayerIt = CSCStripdigis->begin(); CSLayerIt != CSCStripdigis->end(); ++CSLayerIt) {
      // The layerId
      const CSCDetId& layerId = (*CSLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const CSCStripDigiCollection::Range& range = (*CSLayerIt).second;

      OurCSCStripDigis_->put(range, layerId);
      
    }
    // CSCWire
    // 

    // Get the digis from the event
    Handle<CSCWireDigiCollection> CSCWiredigis; 
    e->getByLabel(CSCwiredigi_collection_, CSCWiredigis);

    // Loop over digis, copying them to our own local storage

    CSCWireDigiCollection::DigiRangeIterator CWLayerIt;
    for (CWLayerIt = CSCWiredigis->begin(); CWLayerIt != CSCWiredigis->end(); ++CWLayerIt) {
      // The layerId
      const CSCDetId& layerId = (*CWLayerIt).first;

      // Get the iterators over the digis associated with this LayerId
      const CSCWireDigiCollection::Range& range = (*CWLayerIt).second;

      OurCSCWireDigis_->put(range, layerId);
      
    }

  }
 
  void DataMixingMuonWorker::putMuon(edm::Event &e) {

    // collections of digis to put in the event
    std::auto_ptr< DTDigiCollection > DTDigiMerge( new DTDigiCollection );
    std::auto_ptr< RPCDigiCollection > RPCDigiMerge( new RPCDigiCollection );
    std::auto_ptr< CSCStripDigiCollection > CSCStripDigiMerge( new CSCStripDigiCollection );
    std::auto_ptr< CSCWireDigiCollection > CSCWireDigiMerge( new CSCWireDigiCollection );

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


    // put the collection of recunstructed hits in the event   
    //    LogInfo("DataMixingMuonWorker") << "total # DT Merged Digis: " << DTDigiMerge->size() ;
    //LogInfo("DataMixingMuonWorker") << "total # RPC Merged Digis: " << RPCDigiMerge->size() ;
    //LogInfo("DataMixingMuonWorker") << "total # CSCStrip Merged Digis: " << CSCStripDigiMerge->size() ;
    //LogInfo("DataMixingMuonWorker") << "total # CSCWire Merged Digis: " << CSCWireDigiMerge->size() ;

    e.put( DTDigiMerge, DTDigiCollectionDM_ );
    e.put( RPCDigiMerge, RPCDigiCollectionDM_ );
    e.put( CSCStripDigiMerge, CSCStripDigiCollectionDM_ );
    e.put( CSCWireDigiMerge, CSCWireDigiCollectionDM_ );

    // clear local storage for this event
    delete OurDTDigis_;
    delete OurRPCDigis_;
    delete OurCSCStripDigis_;
    delete OurCSCWireDigis_;


  }

} //edm
