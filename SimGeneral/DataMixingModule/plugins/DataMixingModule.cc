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
    ESrechitCollection_ = ps.getParameter<edm::InputTag>("ESrechitCollection");
    EBRecHitCollectionDM_        = ps.getParameter<std::string>("EBRecHitCollectionDM");
    EERecHitCollectionDM_        = ps.getParameter<std::string>("EERecHitCollectionDM");
    ESRecHitCollectionDM_        = ps.getParameter<std::string>("ESRecHitCollectionDM");
    //   nMaxPrintout_            = ps.getUntrackedParameter<int>("nMaxPrintout",10);

    produces< EBRecHitCollection >(EBRecHitCollectionDM_);
    produces< EERecHitCollection >(EERecHitCollectionDM_);
    produces< ESRecHitCollection >(ESRecHitCollectionDM_);

    EMWorker_ = new DataMixingEMWorker(ps);

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

    HcalWorker_ = new DataMixingHcalWorker(ps);

    // Muons

    DTdigi_collection_   = ps.getParameter<edm::InputTag>("DTdigiCollection");
    RPCdigi_collection_  = ps.getParameter<edm::InputTag>("RPCdigiCollection");
    CSCstripdigi_collection_   = ps.getParameter<edm::InputTag>("CSCstripdigiCollection");
    CSCwiredigi_collection_    = ps.getParameter<edm::InputTag>("CSCwiredigiCollection");
    DTDigiCollectionDM_  = ps.getParameter<std::string>("DTDigiCollectionDM");
    RPCDigiCollectionDM_ = ps.getParameter<std::string>("RPCDigiCollectionDM");
    CSCStripDigiCollectionDM_ = ps.getParameter<std::string>("CSCStripDigiCollectionDM");
    CSCWireDigiCollectionDM_  = ps.getParameter<std::string>("CSCWireDigiCollectionDM");

    produces< DTDigiCollection >(DTDigiCollectionDM_);
    produces< RPCDigiCollection >(RPCDigiCollectionDM_);
    produces< CSCStripDigiCollection >(CSCStripDigiCollectionDM_);
    produces< CSCWireDigiCollection >(CSCWireDigiCollectionDM_);

    MuonWorker_ = new DataMixingMuonWorker(ps);

    // Si-Strips

    Sistripdigi_collection_   = ps.getParameter<edm::InputTag>("SistripdigiCollection");

    SiStripDigiCollectionDM_  = ps.getParameter<std::string>("SiStripDigiCollectionDM");

    produces< edm::DetSetVector<SiStripDigi> > (SiStripDigiCollectionDM_);
    
    SiStripWorker_ = new DataMixingSiStripWorker(ps);

    // Pixels

    pixeldigi_collection_   = ps.getParameter<edm::InputTag>("pixeldigiCollection");

    PixelDigiCollectionDM_  = ps.getParameter<std::string>("PixelDigiCollectionDM");

    produces< edm::DetSetVector<PixelDigi> > (PixelDigiCollectionDM_);

    SiPixelWorker_ = new DataMixingSiPixelWorker(ps);

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
    delete EMWorker_;
    delete HcalWorker_;
    delete MuonWorker_;
    delete SiStripWorker_;
    delete SiPixelWorker_;
  }  

  

  void DataMixingModule::addSignals(const edm::Event &e) { 
    // fill in maps of hits

    LogDebug("DataMixingModule")<<"===============> adding MC signals for "<<e.id();

    // Ecal
    EMWorker_->addEMSignals(e);

    // Hcal
    HcalWorker_->addHcalSignals(e);
    
    // Muon
    MuonWorker_->addMuonSignals(e);

    // SiStrips
    SiStripWorker_->addSiStripSignals(e);

    // SiPixels
    SiPixelWorker_->addSiPixelSignals(e);
    
  } // end of addSignals

  

  void DataMixingModule::addPileups(const int bcr, Event *e, unsigned int eventNr, unsigned int worker) {  


    LogDebug("DataMixingModule") <<"\n===============> adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // Ecal
    EMWorker_->addEMPileups(bcr, e, eventNr);

    // Hcal
    HcalWorker_->addHcalPileups(bcr, e, eventNr);

    // Muon
    MuonWorker_->addMuonPileups(bcr, e, eventNr);

    // SiStrips
    SiStripWorker_->addSiStripPileups(bcr, e, eventNr);

    // SiPixels
    SiPixelWorker_->addSiPixelPileups(bcr, e, eventNr);

  }
 


  void DataMixingModule::doPileUp(edm::Event &e)
  {// 

    for (int bunchCrossing=minBunch_;bunchCrossing<=maxBunch_;++bunchCrossing) {
      setBcrOffset();
      for (unsigned int isource=0;isource<maxNbSources_;++isource) {
	setSourceOffset(isource);
	if (doit_[isource]) {
	  merge(bunchCrossing, (pileup_[isource])[bunchCrossing-minBunch_],1);
	}
      }
    }
  }



  void DataMixingModule::put(edm::Event &e) {

    // individual workers...

    // Ecal
    EMWorker_->putEM(e);

    // Hcal
    HcalWorker_->putHcal(e);

    // Muon
    MuonWorker_->putMuon(e);

    // SiStrips
    SiStripWorker_->putSiStrip(e);

    // SiPixels
    SiPixelWorker_->putSiPixel(e);

  }

  void DataMixingModule::setBcrOffset() {
  }

  void DataMixingModule::setSourceOffset(const unsigned int is) {
  }

} //edm
