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

    MergeEMDigis_ = (ps.getParameter<std::string>("EcalMergeType")=="Digis");

    if(MergeEMDigis_) {
      EBdigiCollectionSig_ = ps.getParameter<edm::InputTag>("EBdigiCollectionSig");
      EEdigiCollectionSig_ = ps.getParameter<edm::InputTag>("EEdigiCollectionSig");
      ESdigiCollectionSig_ = ps.getParameter<edm::InputTag>("ESdigiCollectionSig");
      EBdigiCollectionPile_ = ps.getParameter<edm::InputTag>("EBdigiCollectionPile");
      EEdigiCollectionPile_ = ps.getParameter<edm::InputTag>("EEdigiCollectionPile");
      ESdigiCollectionPile_ = ps.getParameter<edm::InputTag>("ESdigiCollectionPile");
      EBDigiCollectionDM_        = ps.getParameter<std::string>("EBDigiCollectionDM");
      EEDigiCollectionDM_        = ps.getParameter<std::string>("EEDigiCollectionDM");
      ESDigiCollectionDM_        = ps.getParameter<std::string>("ESDigiCollectionDM");
      //   nMaxPrintout_            = ps.getUntrackedParameter<int>("nMaxPrintout",10);

      produces< EBDigiCollection >(EBDigiCollectionDM_);
      produces< EEDigiCollection >(EEDigiCollectionDM_);
      produces< ESDigiCollection >(ESDigiCollectionDM_);

      EMDigiWorker_ = new DataMixingEMDigiWorker(ps);
    }
    else { // merge RecHits
      EBrechitCollectionSig_ = ps.getParameter<edm::InputTag>("EBrechitCollectionSig");
      EErechitCollectionSig_ = ps.getParameter<edm::InputTag>("EErechitCollectionSig");
      ESrechitCollectionSig_ = ps.getParameter<edm::InputTag>("ESrechitCollectionSig");
      EBrechitCollectionPile_ = ps.getParameter<edm::InputTag>("EBrechitCollectionPile");
      EErechitCollectionPile_ = ps.getParameter<edm::InputTag>("EErechitCollectionPile");
      ESrechitCollectionPile_ = ps.getParameter<edm::InputTag>("ESrechitCollectionPile");
      EBRecHitCollectionDM_        = ps.getParameter<std::string>("EBRecHitCollectionDM");
      EERecHitCollectionDM_        = ps.getParameter<std::string>("EERecHitCollectionDM");
      ESRecHitCollectionDM_        = ps.getParameter<std::string>("ESRecHitCollectionDM");
      //   nMaxPrintout_            = ps.getUntrackedParameter<int>("nMaxPrintout",10);

      produces< EBRecHitCollection >(EBRecHitCollectionDM_);
      produces< EERecHitCollection >(EERecHitCollectionDM_);
      produces< ESRecHitCollection >(ESRecHitCollectionDM_);

      EMWorker_ = new DataMixingEMWorker(ps);
    }
    // Hcal next

    MergeHcalDigis_ = (ps.getParameter<std::string>("HcalMergeType")=="Digis");

    if(MergeHcalDigis_){
      HBHEdigiCollectionSig_  = ps.getParameter<edm::InputTag>("HBHEdigiCollectionSig");
      HOdigiCollectionSig_    = ps.getParameter<edm::InputTag>("HOdigiCollectionSig");
      HFdigiCollectionSig_    = ps.getParameter<edm::InputTag>("HFdigiCollectionSig");
      ZDCdigiCollectionSig_   = ps.getParameter<edm::InputTag>("ZDCdigiCollectionSig");
      HBHEdigiCollectionPile_  = ps.getParameter<edm::InputTag>("HBHEdigiCollectionPile");
      HOdigiCollectionPile_    = ps.getParameter<edm::InputTag>("HOdigiCollectionPile");
      HFdigiCollectionPile_    = ps.getParameter<edm::InputTag>("HFdigiCollectionPile");
      ZDCdigiCollectionPile_   = ps.getParameter<edm::InputTag>("ZDCdigiCollectionPile");

      HBHEDigiCollectionDM_ = ps.getParameter<std::string>("HBHEDigiCollectionDM");
      HODigiCollectionDM_   = ps.getParameter<std::string>("HODigiCollectionDM");
      HFDigiCollectionDM_   = ps.getParameter<std::string>("HFDigiCollectionDM");
      ZDCDigiCollectionDM_  = ps.getParameter<std::string>("ZDCDigiCollectionDM");

      produces< HBHEDigiCollection >(HBHEDigiCollectionDM_);
      produces< HODigiCollection >(HODigiCollectionDM_);
      produces< HFDigiCollection >(HFDigiCollectionDM_);
      produces< ZDCDigiCollection >(ZDCDigiCollectionDM_);

      HcalDigiWorker_ = new DataMixingHcalDigiWorker(ps);

    }
    else{

      HBHErechitCollectionSig_  = ps.getParameter<edm::InputTag>("HBHErechitCollectionSig");
      HOrechitCollectionSig_    = ps.getParameter<edm::InputTag>("HOrechitCollectionSig");
      HFrechitCollectionSig_    = ps.getParameter<edm::InputTag>("HFrechitCollectionSig");
      ZDCrechitCollectionSig_   = ps.getParameter<edm::InputTag>("ZDCrechitCollectionSig");
      HBHErechitCollectionPile_  = ps.getParameter<edm::InputTag>("HBHErechitCollectionPile");
      HOrechitCollectionPile_    = ps.getParameter<edm::InputTag>("HOrechitCollectionPile");
      HFrechitCollectionPile_    = ps.getParameter<edm::InputTag>("HFrechitCollectionPile");
      ZDCrechitCollectionPile_   = ps.getParameter<edm::InputTag>("ZDCrechitCollectionPile");

      HBHERecHitCollectionDM_ = ps.getParameter<std::string>("HBHERecHitCollectionDM");
      HORecHitCollectionDM_   = ps.getParameter<std::string>("HORecHitCollectionDM");
      HFRecHitCollectionDM_   = ps.getParameter<std::string>("HFRecHitCollectionDM");
      ZDCRecHitCollectionDM_  = ps.getParameter<std::string>("ZDCRecHitCollectionDM");

      produces< HBHERecHitCollection >(HBHERecHitCollectionDM_);
      produces< HORecHitCollection >(HORecHitCollectionDM_);
      produces< HFRecHitCollection >(HFRecHitCollectionDM_);
      produces< ZDCRecHitCollection >(ZDCRecHitCollectionDM_);

      HcalWorker_ = new DataMixingHcalWorker(ps);
    }

    // Muons

    DTdigi_collectionSig_   = ps.getParameter<edm::InputTag>("DTdigiCollectionSig");
    RPCdigi_collectionSig_  = ps.getParameter<edm::InputTag>("RPCdigiCollectionSig");
    CSCstripdigi_collectionSig_   = ps.getParameter<edm::InputTag>("CSCstripdigiCollectionSig");
    CSCwiredigi_collectionSig_    = ps.getParameter<edm::InputTag>("CSCwiredigiCollectionSig");
    DTdigi_collectionPile_   = ps.getParameter<edm::InputTag>("DTdigiCollectionPile");
    RPCdigi_collectionPile_  = ps.getParameter<edm::InputTag>("RPCdigiCollectionPile");
    CSCstripdigi_collectionPile_   = ps.getParameter<edm::InputTag>("CSCstripdigiCollectionPile");
    CSCwiredigi_collectionPile_    = ps.getParameter<edm::InputTag>("CSCwiredigiCollectionPile");

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

    Sistripdigi_collectionSig_   = ps.getParameter<edm::InputTag>("SistripdigiCollectionSig");
    Sistripdigi_collectionPile_   = ps.getParameter<edm::InputTag>("SistripdigiCollectionPile");

    SiStripDigiCollectionDM_  = ps.getParameter<std::string>("SiStripDigiCollectionDM");


    produces< edm::DetSetVector<SiStripDigi> > (SiStripDigiCollectionDM_);
    
    SiStripWorker_ = new DataMixingSiStripWorker(ps);

    // Pixels

    pixeldigi_collectionSig_   = ps.getParameter<edm::InputTag>("pixeldigiCollectionSig");
    pixeldigi_collectionPile_   = ps.getParameter<edm::InputTag>("pixeldigiCollectionPile");

    PixelDigiCollectionDM_  = ps.getParameter<std::string>("PixelDigiCollectionDM");

    produces< edm::DetSetVector<PixelDigi> > (PixelDigiCollectionDM_);

    SiPixelWorker_ = new DataMixingSiPixelWorker(ps);

  }

  void DataMixingModule::getSubdetectorNames() {
    // get subdetector names
    // edm::Service<edm::ConstProductRegistry> reg;
    // Loop over provenance of products in registry.
    //for (edm::ProductRegistry::ProductList::const_iterator it = reg->productList().begin(); it != reg->productList().end(); ++it) {

      //  **** Check this out.... ****

      // See FWCore/Framework/interface/BranchDescription.h
      // BranchDescription contains all the information for the product.

      // This section not very backwards-compatible in terms of digi-merging.  Need to be able to specify here which data format
      // to look at...

      //      edm::BranchDescription desc = it->second;
      //if (!desc.friendlyClassName_.compare(0,9,"EBRecHitC")) {
      //	Subdetectors_.push_back(desc.productInstanceName_);
      //LogInfo("DataMixingModule") <<"Adding container "<<desc.productInstanceName_ <<" for pileup treatment";
      //}
      //else if (!desc.friendlyClassName_.compare(0,9,"EERecHitC")) {
	//      else if (!desc.friendlyClassName_.compare(0,9,"EErechitC") && desc.productInstanceName_.compare(0,11,"TrackerHits")) {
      //	Subdetectors_.push_back(desc.productInstanceName_);
      //LogInfo("DataMixingModule") <<"Adding container "<<desc.productInstanceName_ <<" for pileup treatment";
      //}
      //else if (!desc.friendlyClassName_.compare(0,9,"HBRecHitC")) {
      //	Subdetectors_.push_back(desc.productInstanceName_);
      //LogInfo("DataMixingModule") <<"Adding container "<<desc.productInstanceName_ <<" for pileup treatment";
      //}
      //else if (!desc.friendlyClassName_.compare(0,9,"HERecHitC")) {
      //	Subdetectors_.push_back(desc.productInstanceName_);
      //LogInfo("DataMixingModule") <<"Adding container "<<desc.productInstanceName_ <<" for pileup treatment";
      // }
	// and so on with other detector types...
    // }
  }       
	       

  void DataMixingModule::beginJob(edm::EventSetup const&iSetup) {
  }

  void DataMixingModule::createnewEDProduct() {
  }
 

  // Virtual destructor needed.
  DataMixingModule::~DataMixingModule() { 
    delete sel_;
    if(MergeEMDigis_){ delete EMDigiWorker_;}
    else {delete EMWorker_;}
    if(MergeHcalDigis_) { delete HcalDigiWorker_;}
    else {delete HcalWorker_;}
    delete MuonWorker_;
    delete SiStripWorker_;
    delete SiPixelWorker_;
  }  



  void DataMixingModule::addSignals(const edm::Event &e, const edm::EventSetup& ES) { 
    // fill in maps of hits

    LogDebug("DataMixingModule")<<"===============> adding MC signals for "<<e.id();

    // Ecal
    if(MergeEMDigis_) { EMDigiWorker_->addEMSignals(e, ES); }
    else{ EMWorker_->addEMSignals(e);}

    // Hcal
    if(MergeHcalDigis_) { HcalDigiWorker_->addHcalSignals(e, ES);}
    else {HcalWorker_->addHcalSignals(e);}
    
    // Muon
    MuonWorker_->addMuonSignals(e);

    // SiStrips
    SiStripWorker_->addSiStripSignals(e);

    // SiPixels
    SiPixelWorker_->addSiPixelSignals(e);
    
  } // end of addSignals

  


  void DataMixingModule::addPileups(const int bcr, Event *e, unsigned int eventNr, unsigned int worker, const edm::EventSetup& ES) {  


    LogDebug("DataMixingModule") <<"\n===============> adding pileups from event  "<<e->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // Ecal
    if(MergeEMDigis_) {    EMDigiWorker_->addEMPileups(bcr, e, eventNr, ES);}
    else {EMWorker_->addEMPileups(bcr, e, eventNr); }

    // Hcal
    if(MergeHcalDigis_) {    HcalDigiWorker_->addHcalPileups(bcr, e, eventNr, ES);}
    else {HcalWorker_->addHcalPileups(bcr, e, eventNr);}

    // Muon
    MuonWorker_->addMuonPileups(bcr, e, eventNr);

    // SiStrips
    SiStripWorker_->addSiStripPileups(bcr, e, eventNr);

    // SiPixels
    SiPixelWorker_->addSiPixelPileups(bcr, e, eventNr);

  }


  void DataMixingModule::doPileUp(edm::Event &e, const edm::EventSetup& ES)
  {//                                                                                       
                                       
    for (int bunchCrossing=minBunch_;bunchCrossing<=maxBunch_;++bunchCrossing) {
      setBcrOffset();
      for (unsigned int isource=0;isource<maxNbSources_;++isource) {
        setSourceOffset(isource);
        if (doit_[isource]) {
          merge(bunchCrossing, (pileup_[isource])[bunchCrossing-minBunch_],1, ES);
        }
      }
    }
  }


  void DataMixingModule::put(edm::Event &e,const edm::EventSetup& ES) {

    // individual workers...

    // Ecal
    if(MergeEMDigis_) {EMDigiWorker_->putEM(e,ES);}
    else {EMWorker_->putEM(e);}

    // Hcal
    if(MergeHcalDigis_) {HcalDigiWorker_->putHcal(e,ES);}
    else {HcalWorker_->putHcal(e);}

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
