// File: DataMixingModule.cc
// Description:  see DataMixingModule.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include <iostream>
#include <boost/bind.hpp>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
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
  {  
                                                       // what's "label_"?

    // get the subdetector names
    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // For now, list all of them here.  Later, make this selectable with input parameters
    // 

    // Check to see if we are working in Full or Fast Simulation

    DoFastSim_ = (ps.getParameter<std::string>("IsThisFastSim")).compare("YES") == 0;
    MergeEMDigis_ = (ps.getParameter<std::string>("EcalMergeType")).compare("Digis") == 0;
    MergeHcalDigis_ = (ps.getParameter<std::string>("HcalMergeType")).compare("Digis") == 0;
    if(MergeHcalDigis_) MergeHcalDigisProd_ = (ps.getParameter<std::string>("HcalDigiMerge")=="FullProd");

    addMCDigiNoise_ = false;

    addMCDigiNoise_ = ps.getUntrackedParameter<bool>("addMCDigiNoise");  // for Sim on Sim mixing

    

    // Put Fast Sim Sequences here for Simplification: Fewer options!

    if(DoFastSim_) {

    // declare the products to produce

      //Ecal:

      EBRecHitCollectionDM_        = ps.getParameter<std::string>("EBRecHitCollectionDM");
      EERecHitCollectionDM_        = ps.getParameter<std::string>("EERecHitCollectionDM");
      ESRecHitCollectionDM_        = ps.getParameter<std::string>("ESRecHitCollectionDM");

      produces< EBRecHitCollection >(EBRecHitCollectionDM_);
      produces< EERecHitCollection >(EERecHitCollectionDM_);
      produces< ESRecHitCollection >(ESRecHitCollectionDM_);

      EMWorker_ = new DataMixingEMWorker(ps, consumesCollector() );

      //Hcal:

      HBHERecHitCollectionDM_ = ps.getParameter<std::string>("HBHERecHitCollectionDM");
      HORecHitCollectionDM_   = ps.getParameter<std::string>("HORecHitCollectionDM");
      HFRecHitCollectionDM_   = ps.getParameter<std::string>("HFRecHitCollectionDM");
      ZDCRecHitCollectionDM_  = ps.getParameter<std::string>("ZDCRecHitCollectionDM");

      produces< HBHERecHitCollection >(HBHERecHitCollectionDM_);
      produces< HORecHitCollection >(HORecHitCollectionDM_);
      produces< HFRecHitCollection >(HFRecHitCollectionDM_);
      produces< ZDCRecHitCollection >(ZDCRecHitCollectionDM_);

      HcalWorker_ = new DataMixingHcalWorker(ps, consumesCollector());

      //Muons:

      DTDigiCollectionDM_  = ps.getParameter<std::string>("DTDigiCollectionDM");
      RPCDigiCollectionDM_ = ps.getParameter<std::string>("RPCDigiCollectionDM");
      CSCStripDigiCollectionDM_ = ps.getParameter<std::string>("CSCStripDigiCollectionDM");
      CSCWireDigiCollectionDM_  = ps.getParameter<std::string>("CSCWireDigiCollectionDM");
      CSCComparatorDigiCollectionDM_  = ps.getParameter<std::string>("CSCComparatorDigiCollectionDM");

      produces< DTDigiCollection >();
      produces< RPCDigiCollection >();
      produces< CSCStripDigiCollection >(CSCStripDigiCollectionDM_);
      produces< CSCWireDigiCollection >(CSCWireDigiCollectionDM_);
      produces< CSCComparatorDigiCollection >(CSCComparatorDigiCollectionDM_);

      MuonWorker_ = new DataMixingMuonWorker(ps, consumesCollector());

      //Tracks:

      GeneralTrackCollectionDM_  = ps.getParameter<std::string>("GeneralTrackDigiCollectionDM");
      produces< reco::TrackCollection >(GeneralTrackCollectionDM_);
      GeneralTrackWorker_ = new DataMixingGeneralTrackWorker(ps, consumesCollector());

    }
    else{  // Full Simulation options

      //cout<<"FastSim False!!!"<<endl;

    // declare the products to produce
    // Start with EM
    if(MergeEMDigis_) {

      // cout<<"EM Digis TRUE!!!"<<endl;

      EBDigiCollectionDM_        = ps.getParameter<std::string>("EBDigiCollectionDM");
      EEDigiCollectionDM_        = ps.getParameter<std::string>("EEDigiCollectionDM");
      ESDigiCollectionDM_        = ps.getParameter<std::string>("ESDigiCollectionDM");
      //   nMaxPrintout_            = ps.getUntrackedParameter<int>("nMaxPrintout",10);

      produces< EBDigiCollection >(EBDigiCollectionDM_);
      produces< EEDigiCollection >(EEDigiCollectionDM_);
      produces< ESDigiCollection >(ESDigiCollectionDM_);

      EMDigiWorker_ = new DataMixingEMDigiWorker(ps, consumesCollector());
    }
    else { // merge RecHits 
      EBRecHitCollectionDM_        = ps.getParameter<std::string>("EBRecHitCollectionDM");
      EERecHitCollectionDM_        = ps.getParameter<std::string>("EERecHitCollectionDM");
      ESRecHitCollectionDM_        = ps.getParameter<std::string>("ESRecHitCollectionDM");
      //   nMaxPrintout_            = ps.getUntrackedParameter<int>("nMaxPrintout",10);

      produces< EBRecHitCollection >(EBRecHitCollectionDM_);
      produces< EERecHitCollection >(EERecHitCollectionDM_);
      produces< ESRecHitCollection >(ESRecHitCollectionDM_);

      EMWorker_ = new DataMixingEMWorker(ps, consumesCollector());
    }
    // Hcal next

    if(MergeHcalDigis_){
      //       cout<<"Hcal Digis TRUE!!!"<<endl;

      HBHEDigiCollectionDM_ = ps.getParameter<std::string>("HBHEDigiCollectionDM");
      HODigiCollectionDM_   = ps.getParameter<std::string>("HODigiCollectionDM");
      HFDigiCollectionDM_   = ps.getParameter<std::string>("HFDigiCollectionDM");
      ZDCDigiCollectionDM_  = ps.getParameter<std::string>("ZDCDigiCollectionDM");

      produces< HBHEDigiCollection >();
      produces< HODigiCollection >();
      produces< HFDigiCollection >();
      produces< ZDCDigiCollection >();

      produces<HBHEUpgradeDigiCollection>("HBHEUpgradeDigiCollection");
      produces<HFUpgradeDigiCollection>("HFUpgradeDigiCollection");


      if(MergeHcalDigisProd_) {
	//        edm::ConsumesCollector iC(consumesCollector());
	HcalDigiWorkerProd_ = new DataMixingHcalDigiWorkerProd(ps, consumesCollector());
      }
      else {HcalDigiWorker_ = new DataMixingHcalDigiWorker(ps, consumesCollector());
      }


    }
    else{
      HBHERecHitCollectionDM_ = ps.getParameter<std::string>("HBHERecHitCollectionDM");
      HORecHitCollectionDM_   = ps.getParameter<std::string>("HORecHitCollectionDM");
      HFRecHitCollectionDM_   = ps.getParameter<std::string>("HFRecHitCollectionDM");
      ZDCRecHitCollectionDM_  = ps.getParameter<std::string>("ZDCRecHitCollectionDM");

      produces< HBHERecHitCollection >(HBHERecHitCollectionDM_);
      produces< HORecHitCollection >(HORecHitCollectionDM_);
      produces< HFRecHitCollection >(HFRecHitCollectionDM_);
      produces< ZDCRecHitCollection >(ZDCRecHitCollectionDM_);

      HcalWorker_ = new DataMixingHcalWorker(ps, consumesCollector());
    }

    // Muons

    DTDigiCollectionDM_  = ps.getParameter<std::string>("DTDigiCollectionDM");
    RPCDigiCollectionDM_ = ps.getParameter<std::string>("RPCDigiCollectionDM");
    CSCStripDigiCollectionDM_ = ps.getParameter<std::string>("CSCStripDigiCollectionDM");
    CSCWireDigiCollectionDM_  = ps.getParameter<std::string>("CSCWireDigiCollectionDM");
    CSCComparatorDigiCollectionDM_  = ps.getParameter<std::string>("CSCComparatorDigiCollectionDM");


    produces< DTDigiCollection >();
    produces< RPCDigiCollection >();
    produces< CSCStripDigiCollection >(CSCStripDigiCollectionDM_);
    produces< CSCWireDigiCollection >(CSCWireDigiCollectionDM_);
    produces< CSCComparatorDigiCollection >(CSCComparatorDigiCollectionDM_);

    MuonWorker_ = new DataMixingMuonWorker(ps, consumesCollector());

    // Si-Strips

    useSiStripRawDigi_ = ps.exists("SiStripRawDigiSource")?
      ps.getParameter<std::string>("SiStripRawDigiSource")=="PILEUP" ||
      ps.getParameter<std::string>("SiStripRawDigiSource")=="SIGNAL" : false;

    SiStripDigiCollectionDM_  = ps.getParameter<std::string>("SiStripDigiCollectionDM");

    if(useSiStripRawDigi_) {

      produces< edm::DetSetVector<SiStripRawDigi> > (SiStripDigiCollectionDM_);
      SiStripRawWorker_ = new DataMixingSiStripRawWorker(ps, consumesCollector());

    } else {

      produces< edm::DetSetVector<SiStripDigi> > (SiStripDigiCollectionDM_);
      SiStripWorker_ = new DataMixingSiStripWorker(ps, consumesCollector());

      if( addMCDigiNoise_ ) {
	SiStripMCDigiWorker_ = new DataMixingSiStripMCDigiWorker(ps, consumesCollector());
      }
      else {
	SiStripWorker_ = new DataMixingSiStripWorker(ps, consumesCollector());
      }
    }

    // Pixels

    PixelDigiCollectionDM_  = ps.getParameter<std::string>("PixelDigiCollectionDM");

    produces< edm::DetSetVector<PixelDigi> > (PixelDigiCollectionDM_);

    SiPixelWorker_ = new DataMixingSiPixelWorker(ps, consumesCollector());

    }

    // Pileup Information: if doing pre-mixing, we have to save the pileup information from the Secondary stream

    MergePileup_ = ps.getParameter<bool>("MergePileupInfo");

    if(MergePileup_) {
      produces< std::vector<PileupSummaryInfo> >();
      produces<CrossingFramePlaybackInfoExtended>();

      PUWorker_ = new DataMixingPileupCopy(ps, consumesCollector());
    }

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
	       

  void DataMixingModule::initializeEvent(const edm::Event &e, const edm::EventSetup& ES) { 
    if( addMCDigiNoise_ ) {
      SiStripMCDigiWorker_->initializeEvent( e, ES );
    }
  }

  // Virtual destructor needed.
  DataMixingModule::~DataMixingModule() { 
    if(MergeEMDigis_){ delete EMDigiWorker_;}
    else {delete EMWorker_;}
    if(MergeHcalDigis_) { 
      if(MergeHcalDigisProd_) { delete HcalDigiWorkerProd_;}
      else { delete HcalDigiWorker_; }}
    else {delete HcalWorker_;}
    if(MuonWorker_) delete MuonWorker_;
    if(DoFastSim_){
      delete GeneralTrackWorker_;
    }else{
      if(useSiStripRawDigi_)
	delete SiStripRawWorker_;
      else
	if(addMCDigiNoise_ ) delete SiStripMCDigiWorker_;
	else delete SiStripWorker_;
      delete SiPixelWorker_;
    }
    if(MergePileup_) { delete PUWorker_;}
  }

  void DataMixingModule::addSignals(const edm::Event &e, const edm::EventSetup& ES) { 
    // fill in maps of hits

    LogDebug("DataMixingModule")<<"===============> adding MC signals for "<<e.id();

    // Ecal
    if(MergeEMDigis_) { EMDigiWorker_->addEMSignals(e, ES); }
    else{ EMWorker_->addEMSignals(e);}

    // Hcal
    if(MergeHcalDigis_) { 
      if(MergeHcalDigisProd_){
	HcalDigiWorkerProd_->addHcalSignals(e, ES);
      }
      else{
	HcalDigiWorker_->addHcalSignals(e, ES);
      }
    }
    else {HcalWorker_->addHcalSignals(e);}
    
    // Muon
    MuonWorker_->addMuonSignals(e);

    if(DoFastSim_){
       GeneralTrackWorker_->addGeneralTrackSignals(e);
    }else{
    // SiStrips
    if(useSiStripRawDigi_) SiStripRawWorker_->addSiStripSignals(e);
    else if(addMCDigiNoise_ ) SiStripMCDigiWorker_->addSiStripSignals(e);
    else SiStripWorker_->addSiStripSignals(e);

    // SiPixels
    SiPixelWorker_->addSiPixelSignals(e);
    }    
    AddedPileup_ = false;

  } // end of addSignals

  


  void DataMixingModule::pileWorker(const EventPrincipal &ep, int bcr, int eventNr, const edm::EventSetup& ES, edm::ModuleCallingContext const* mcc) {  

    InternalContext internalContext(ep.id(), mcc);
    ParentContext parentContext(&internalContext);
    ModuleCallingContext moduleCallingContext(&moduleDescription());
    ModuleContextSentry moduleContextSentry(&moduleCallingContext, parentContext);

    LogDebug("DataMixingModule") <<"\n===============> adding pileups from event  "<<ep.id()<<" for bunchcrossing "<<bcr;

    // Note:  setupPileUpEvent may modify the run and lumi numbers of the EventPrincipal to match that of the primary event.
    setupPileUpEvent(ES);

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // Ecal
    if(MergeEMDigis_) {    EMDigiWorker_->addEMPileups(bcr, &ep, eventNr, ES, &moduleCallingContext);}
    else {EMWorker_->addEMPileups(bcr, &ep, eventNr, &moduleCallingContext); }

    // Hcal
    if(MergeHcalDigis_) {    
      if(MergeHcalDigisProd_) {    
	HcalDigiWorkerProd_->addHcalPileups(bcr, &ep, eventNr, ES, &moduleCallingContext);
      }
      else{
	HcalDigiWorker_->addHcalPileups(bcr, &ep, eventNr, ES, &moduleCallingContext);}
    }
    else {HcalWorker_->addHcalPileups(bcr, &ep, eventNr, &moduleCallingContext);}

    // Muon
    MuonWorker_->addMuonPileups(bcr, &ep, eventNr, &moduleCallingContext);

    if(DoFastSim_){
      GeneralTrackWorker_->addGeneralTrackPileups(bcr, &ep, eventNr, &moduleCallingContext);
    }else{
      
      // SiStrips
      if(useSiStripRawDigi_) SiStripRawWorker_->addSiStripPileups(bcr, &ep, eventNr, &moduleCallingContext);
      else if(addMCDigiNoise_ ) SiStripMCDigiWorker_->addSiStripPileups(bcr, &ep, eventNr, &moduleCallingContext);
      else SiStripWorker_->addSiStripPileups(bcr, &ep, eventNr, &moduleCallingContext);
      
      // SiPixels
      SiPixelWorker_->addSiPixelPileups(bcr, &ep, eventNr, &moduleCallingContext);
    }

    // check and see if we need to copy the pileup information from 
    // secondary stream to the output stream  
    // We only have the pileup event here, so pick the first time and store the info

    if(MergePileup_ && !AddedPileup_){
      
      PUWorker_->addPileupInfo(&ep, eventNr, &moduleCallingContext);

      AddedPileup_ = true;
    }

  }


  
  void DataMixingModule::doPileUp(edm::Event &e, const edm::EventSetup& ES)
  {
    std::vector<edm::EventID> recordEventID;
    std::vector<int> PileupList;
    PileupList.clear();
    TrueNumInteractions_.clear();

    ModuleCallingContext const* mcc = e.moduleCallingContext();

    for (int bunchCrossing=minBunch_;bunchCrossing<=maxBunch_;++bunchCrossing) {
      for (unsigned int isource=0;isource<maxNbSources_;++isource) {
        boost::shared_ptr<PileUp> source = inputSources_[isource];
        if (not source or not source->doPileUp()) 
          continue;

	if (isource==0) 
          source->CalculatePileup(minBunch_, maxBunch_, PileupList, TrueNumInteractions_, e.streamID());

	int NumPU_Events = 0;
	if (isource ==0) { 
          NumPU_Events = PileupList[bunchCrossing - minBunch_];
        } else {
          // non-minbias pileup only gets one event for now. Fix later if desired.
          NumPU_Events = 1;
        }  

        source->readPileUp(
                e.id(),
                recordEventID,
                boost::bind(&DataMixingModule::pileWorker, boost::ref(*this),
                            _1, bunchCrossing, _2, boost::cref(ES), mcc),
		NumPU_Events,
                e.streamID()
                );
      }
    }

  }


  void DataMixingModule::put(edm::Event &e,const edm::EventSetup& ES) {

    // individual workers...

    // Ecal
    if(MergeEMDigis_) {EMDigiWorker_->putEM(e,ES);}
    else {EMWorker_->putEM(e);}

    // Hcal
    if(MergeHcalDigis_) {
      if(MergeHcalDigisProd_) {
	HcalDigiWorkerProd_->putHcal(e,ES);
      }
      else{
	HcalDigiWorker_->putHcal(e,ES);
      }
    }
    else {HcalWorker_->putHcal(e);}

    // Muon
    MuonWorker_->putMuon(e);

    if(DoFastSim_){
       GeneralTrackWorker_->putGeneralTrack(e);
    }else{
       // SiStrips
      if(useSiStripRawDigi_) SiStripRawWorker_->putSiStrip(e);
      else if(addMCDigiNoise_ ) SiStripMCDigiWorker_->putSiStrip(e, ES);
      else SiStripWorker_->putSiStrip(e);
       
       // SiPixels
       SiPixelWorker_->putSiPixel(e);
    }

    if(MergePileup_) { PUWorker_->putPileupInfo(e);}


  }


} //edm
