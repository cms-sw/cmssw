// File: DataMixingModule.cc
// Description:  see DataMixingModule.h
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
#include "DataMixingModule.h"


using namespace std;

namespace edm
{

  // Constructor 
  DataMixingModule::DataMixingModule(const edm::ParameterSet& ps) : BMixingModule(ps),
							    label_(ps.getParameter<std::string>("Label"))

  {                                                         // what's "label_"?

    // get the subdetector names
    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // create input selector
    if (label_.size()>0){
      sel_=new Selector( ModuleLabelSelector(label_));
    }
    else {
      sel_=new Selector( MatchAllSelector());
    }

    // For now, list all of them here.  Later, make this selectable with input parameters
    // 
    // declare the products to produce

    // Start with EM

    EBrechitCollection_ = ps.getParameter<edm::InputTag>("EBrechitCollection");
    EErechitCollection_ = ps.getParameter<edm::InputTag>("EErechitCollection");
    EBRecHitCollectionDM_        = ps.getParameter<std::string>("EBRecHitCollectionDM");
    EERecHitCollectionDM_        = ps.getParameter<std::string>("EERecHitCollectionDM");
    //   nMaxPrintout_            = ps.getUntrackedParameter<int>("nMaxPrintout",10);

    produces< EBRecHitCollection >(EBRecHitCollectionDM_);
    produces< EERecHitCollection >(EERecHitCollectionDM_);

    EMWorker_ = DataMixingEMWorker(ps);

    // Hcal next

    HBHErechitCollection_  = ps.getParameter<edm::InputTag>("HBHErechitCollection");
    HOrechitCollection_    = ps.getParameter<edm::InputTag>("HOrechitCollection");
    HFrechitCollection_    = ps.getParameter<edm::InputTag>("HFrechitCollection");
    ZDCrechitCollection_   = ps.getParameter<edm::InputTag>("ZDCrechitCollection");

    HBHERecHitCollectionDM_ = ps.getParameter<std::string>("HBHERecHitCollectionDM");
    HORecHitCollectionDM_   = ps.getParameter<std::string>("HORecHitCollectionDM");
    HFRecHitCollectionDM_   = ps.getParameter<std::string>("HFRecHitCollectionDM");
    ZDCRecHitCollectionDM_  = ps.getParameter<std::string>("ZDCRecHitCollectionDM");


    produces< HBHERecHitCollection >(HBHERecHitCollectionDM_);
    produces< HORecHitCollection >(HORecHitCollectionDM_);
    produces< HFRecHitCollection >(HFRecHitCollectionDM_);
    produces< ZDCRecHitCollection >(ZDCRecHitCollectionDM_);

    HcalWorker_ = DataMixingHcalWorker(ps);

    // Muons

    DTdigi_collection_   = ps.getParameter<edm::InputTag>("DTdigi_collection");
    RPCdigi_collection_  = ps.getParameter<edm::InputTag>("RPCdigi_collection");
    CSCstripdigi_collection_   = ps.getParameter<edm::InputTag>("CSCstripdigi_collection");
    CSCwiredigi_collection_    = ps.getParameter<edm::InputTag>("CSCwiredigi_collection");

    DTDigiCollectionDM_  = ps.getParameter<std::string>("DTDigiCollectionDM");
    RPCDigiCollectionDM_ = ps.getParameter<std::string>("RPCDigiCollectionDM");
    CSCStripDigiCollectionDM_ = ps.getParameter<std::string>("CSCStripDigiCollectionDM");
    CSCWireDigiCollectionDM_  = ps.getParameter<std::string>("CSCWireDigiCollectionDM");


    produces< DTDigiCollection >(DTDigiCollectionDM_);
    produces< RPCDigiCollection >(RPCDigiCollectionDM_);
    produces< CSCStripDigiCollection >(CSCStripDigiCollectionDM_);
    produces< CSCWireDigiCollection >(CSCWireDigiCollectionDM_);

    MuonWorker_ = DataMixingMuonWorker(ps);

    // Si-Strips

    Sistripdigi_collection_   = ps.getParameter<edm::InputTag>("Sistripdigi_collection");

    SiStripDigiCollectionDM_  = ps.getParameter<std::string>("SiStripDigiCollectionDM");

    produces< edm::DetSetVector<SiStripDigi> > (SiStripDigiCollectionDM_);
    
    SiStripWorker_ = DataMixingSiStripWorker(ps);

    // Pixels

    pixeldigi_collection_   = ps.getParameter<edm::InputTag>("pixeldigi_collection");

    PixelDigiCollectionDM_  = ps.getParameter<std::string>("PixelDigiCollectionDM");

    produces< edm::DetSetVector<PixelDigi> > (PixelDigiCollectionDM_);

    SiPixelWorker_ = DataMixingSiPixelWorker(ps);


  }

  void DataMixingModule::getSubdetectorNames() {
    // get subdetector names
    edm::Service<edm::ConstProductRegistry> reg;
    // Loop over provenance of products in registry.
    for (edm::ProductRegistry::ProductList::const_iterator it = reg->productList().begin(); it != reg->productList().end(); ++it) {

      //  **** Check this out.... ****

      // See FWCore/Framework/interface/BranchDescription.h
      // BranchDescription contains all the information for the product.

      // This section not very backwards-compatible in terms of digi-merging.  Need to be able to specify here which data format
      // to look at...

      edm::BranchDescription desc = it->second;
      if (!desc.friendlyClassName_.compare(0,9,"EBRecHitC")) {
	Subdetectors_.push_back(desc.productInstanceName_);
	LogInfo("DataMixingModule") <<"Adding container "<<desc.productInstanceName_ <<" for pileup treatment";
      }
      else if (!desc.friendlyClassName_.compare(0,9,"EERecHitC")) {
	//      else if (!desc.friendlyClassName_.compare(0,9,"EErechitC") && desc.productInstanceName_.compare(0,11,"TrackerHits")) {
	Subdetectors_.push_back(desc.productInstanceName_);
        LogInfo("DataMixingModule") <<"Adding container "<<desc.productInstanceName_ <<" for pileup treatment";
      }
      else if (!desc.friendlyClassName_.compare(0,9,"HBRecHitC")) {
	Subdetectors_.push_back(desc.productInstanceName_);
	LogInfo("DataMixingModule") <<"Adding container "<<desc.productInstanceName_ <<" for pileup treatment";
      }
      else if (!desc.friendlyClassName_.compare(0,9,"HERecHitC")) {
	Subdetectors_.push_back(desc.productInstanceName_);
	LogInfo("DataMixingModule") <<"Adding container "<<desc.productInstanceName_ <<" for pileup treatment";
      }
	// and so on with other detector types...
    }
  }       
	       

  void DataMixingModule::beginJob(edm::EventSetup const&iSetup) {
  }

  void DataMixingModule::createnewEDProduct() {
  }
 

  // Virtual destructor needed.
  DataMixingModule::~DataMixingModule() { 
    delete sel_;
  }  

  

  void DataMixingModule::addSignals(const edm::Event &e) { 
    // fill in maps of hits

    LogDebug("DataMixingModule")<<"===============> adding MC signals for "<<e.id();

    // Ecal
    EMWorker_.addEMSignals(e);

    // Hcal
    HcalWorker_.addHcalSignals(e);
    
    // Muon
    MuonWorker_.addMuonSignals(e);

    // SiStrips
    SiStripWorker_.addSiStripSignals(e);

    // SiPixels
    SiPixelWorker_.addSiPixelSignals(e);
    
  } // end of addSignals

  

  void DataMixingModule::addPileups(const int bcr, Event *e, unsigned int eventNr) {
  
    LogDebug("DataMixingModule") <<"\n===============> adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // Ecal
    EMWorker_.addEMPileups(bcr, e, eventNr);

    // Hcal
    HcalWorker_.addHcalPileups(bcr, e, eventNr);

    // Muon
    MuonWorker_.addMuonPileups(bcr, e, eventNr);

    // SiStrips
    SiStripWorker_.addSiStripPileups(bcr, e, eventNr);

    // SiPixels
    SiPixelWorker_.addSiPixelPileups(bcr, e, eventNr);
 
  }
 


  void DataMixingModule::put(edm::Event &e) {

    // individual workers...

    // Ecal
    EMWorker_.putEM(e);

    // Hcal
    HcalWorker_.putHcal(e);

    // Muon
    MuonWorker_.putMuon(e);

    // SiStrips
    SiStripWorker_.putSiStrip(e);

    // SiPixels
    SiPixelWorker_.putSiPixel(e);

  }

} //edm
