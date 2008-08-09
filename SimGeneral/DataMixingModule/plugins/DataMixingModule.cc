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
      EBdigiCollection_ = ps.getParameter<edm::InputTag>("EBdigiCollection");
      EEdigiCollection_ = ps.getParameter<edm::InputTag>("EEdigiCollection");
      ESdigiCollection_ = ps.getParameter<edm::InputTag>("ESdigiCollection");
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
    }
    // Hcal next

    MergeHcalDigis_ = (ps.getParameter<std::string>("HcalMergeType")=="Digis");

    if(MergeHcalDigis_){
      HBHEdigiCollection_  = ps.getParameter<edm::InputTag>("HBHEdigiCollection");
      HOdigiCollection_    = ps.getParameter<edm::InputTag>("HOdigiCollection");
      HFdigiCollection_    = ps.getParameter<edm::InputTag>("HFdigiCollection");
      ZDCdigiCollection_   = ps.getParameter<edm::InputTag>("ZDCdigiCollection");

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
    }

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

    void DataMixingModule::produce(edm::Event& e, const edm::EventSetup& ES) { 

    // Create EDProduct
    createnewEDProduct();

    // Add signals 
    addSignals(e, ES);

    // Read the PileUp 
    //    std::vector<EventPrincipalVector> pileup[maxNbSources_];
    //    bool doit[maxNbSources_];
    for (unsigned int is=0;is< maxNbSources_;++is) {
      doit_[is]=false;
      pileup_[is].clear();
    }

    if ( input_)  {  
      if (playback_) {
	getEventStartInfo(e,0);
	input_->readPileUp(pileup_[0],eventIDs_, fileSeqNrs_, nrEvents_);
      } else {
	input_->readPileUp(pileup_[0],eventIDs_, fileSeqNrs_, nrEvents_); 
        setEventStartInfo(0);
      }
      if (input_->doPileup()) {  
	LogDebug("DataMixingModule") <<"\n\n==============================>Adding pileup to signal event "<<e.id(); 
	doit_[0]=true;
      } 
    }
    if (cosmics_) {
      if (playback_) {
	getEventStartInfo(e,1);
	cosmics_->readPileUp(pileup_[1],eventIDs_, fileSeqNrs_, nrEvents_); 
      } else {
	cosmics_->readPileUp(pileup_[1],eventIDs_, fileSeqNrs_, nrEvents_); 
	setEventStartInfo(1);
      }
      if (cosmics_->doPileup()) {  
	LogDebug("DataMixingModule") <<"\n\n==============================>Adding cosmics to signal event "<<e.id(); 
	doit_[1]=true;
      } 
    }

    if (beamHalo_p_) {
      if (playback_) {
	getEventStartInfo(e,2);
	beamHalo_p_->readPileUp(pileup_[2],eventIDs_, fileSeqNrs_, nrEvents_);
      } else {
	beamHalo_p_->readPileUp(pileup_[2],eventIDs_, fileSeqNrs_, nrEvents_);
	setEventStartInfo(2);
      }
      if (beamHalo_p_->doPileup()) {  
	LogDebug("DataMixingModule") <<"\n\n==============================>Adding beam halo+ to signal event "<<e.id();
	doit_[2]=true;
      } 
    }

    if (beamHalo_m_) {
      if (playback_) {
	getEventStartInfo(e,3);
	beamHalo_m_->readPileUp(pileup_[3],eventIDs_, fileSeqNrs_, nrEvents_);
      } else {
	beamHalo_m_->readPileUp(pileup_[3],eventIDs_, fileSeqNrs_, nrEvents_);
	setEventStartInfo(3);
      }
      if (beamHalo_m_->doPileup()) {  
	LogDebug("DataMixingModule") <<"\n\n==============================>Adding beam halo- to signal event "<<e.id();
	doit_[3]=true;
      }
    }

    if (fwdDet_) {
      if (playback_) {
	getEventStartInfo(e,4);
	fwdDet_->readPileUp(pileup_[4],eventIDs_, fileSeqNrs_, nrEvents_);
      } else {
	fwdDet_->readPileUp(pileup_[4],eventIDs_, fileSeqNrs_, nrEvents_);
	setEventStartInfo(4);
      }

      if (fwdDet_->doPileup()) {  
	LogDebug("DataMixingModule") <<"\n\n==============================>Adding fwd detector source  to signal event "<<e.id();
	doit_[4]=true;
      }  
    }

    doPileUp(e, ES);

    // Put output into event (here only playback info)
    put(e);
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

  void DataMixingModule::merge(const int bcr, const EventPrincipalVector& vec, unsigned int worker, const edm::EventSetup& ES) {
    //
    // main loop: loop over events and merge 
    //

    // should use eventId_ here from BMixing...

    eventId_=0;
    LogDebug("MixingModule") <<"For bunchcrossing "<<bcr<<", "<<vec.size()<< " events will be merged";
    vertexoffset=0;
    for (EventPrincipalVector::const_iterator it = vec.begin(); it != vec.end();
	 ++it) {
      Event e(**it, md_);
      LogDebug("MixingModule") <<" merging Event:  id " << e.id();
      addPileups(bcr, &e, ++eventId_ ,worker, ES);
    }// end main loop
  }

  void DataMixingModule::merge(const int bcr, const EventPrincipalVector& vec, unsigned int worker) {
    //
    // main loop: loop over events and merge 
    //
    eventId_=0;
    LogDebug("MixingModule") <<"For bunchcrossing "<<bcr<<", "<<vec.size()<< " events will be merged";
    vertexoffset=0;
    for (EventPrincipalVector::const_iterator it = vec.begin(); it != vec.end();
	 ++it) {
      Event e(**it, md_);
      LogDebug("MixingModule") <<" merging Event:  id " << e.id();
      addPileups(bcr, &e, ++eventId_ ,worker);
    }// end main loop
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
