// File: DataMixingModule.cc
// Description:  see DataMixingModule.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include <iostream>
#include <memory>
#include <functional>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
//
//
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoNew.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataMixingModule.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

using namespace std;

namespace edm
{

  // Constructor 
  DataMixingModule::DataMixingModule(const edm::ParameterSet& ps, MixingCache::Config const* globalConf) :
    BMixingModule(ps, globalConf),
    EBPileInputTag_(ps.getParameter<edm::InputTag>("EBPileInputTag")),
    EEPileInputTag_(ps.getParameter<edm::InputTag>("EEPileInputTag")),
    ESPileInputTag_(ps.getParameter<edm::InputTag>("ESPileInputTag")),
    HBHEPileInputTag_(ps.getParameter<edm::InputTag>("HBHEPileInputTag")),
    HOPileInputTag_(ps.getParameter<edm::InputTag>("HOPileInputTag")),
    HFPileInputTag_(ps.getParameter<edm::InputTag>("HFPileInputTag")),
    ZDCPileInputTag_(ps.getParameter<edm::InputTag>("ZDCPileInputTag")),
    QIE10PileInputTag_(ps.getParameter<edm::InputTag>("QIE10PileInputTag")),
    QIE11PileInputTag_(ps.getParameter<edm::InputTag>("QIE11PileInputTag")),
							    label_(ps.getParameter<std::string>("Label"))
  {  
    // prepare for data access in DataMixingHcalDigiWorkerProd
    tok_hbhe_ = consumes<HBHEDigitizerTraits::DigiCollection>(HBHEPileInputTag_);
    tok_ho_ = consumes<HODigitizerTraits::DigiCollection>(HOPileInputTag_);
    tok_hf_ = consumes<HFDigitizerTraits::DigiCollection>(HFPileInputTag_);
    tok_zdc_ = consumes<ZDCDigitizerTraits::DigiCollection>(ZDCPileInputTag_);
    tok_qie10_ = consumes<HcalQIE10DigitizerTraits::DigiCollection>(QIE10PileInputTag_);
    tok_qie11_ = consumes<HcalQIE11DigitizerTraits::DigiCollection>(QIE11PileInputTag_);

    // get the subdetector names
    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // For now, list all of them here.  Later, make this selectable with input parameters
    // 

    // Check to see if we are working in Full or Fast Simulation

    MergeTrackerDigis_ = (ps.getParameter<std::string>("TrackerMergeType")) == "Digis";
    MergeEMDigis_ = (ps.getParameter<std::string>("EcalMergeType")) == "Digis";
    MergeHcalDigis_ = (ps.getParameter<std::string>("HcalMergeType")) == "Digis";
    if(MergeHcalDigis_) MergeHcalDigisProd_ = (ps.getParameter<std::string>("HcalDigiMerge")=="FullProd");

    // Put Fast Sim Sequences here for Simplification: Fewer options!

    
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
      HBHEDigiCollectionDM_ = ps.getParameter<std::string>("HBHEDigiCollectionDM");
      HODigiCollectionDM_   = ps.getParameter<std::string>("HODigiCollectionDM");
      HFDigiCollectionDM_   = ps.getParameter<std::string>("HFDigiCollectionDM");
      ZDCDigiCollectionDM_  = ps.getParameter<std::string>("ZDCDigiCollectionDM");
      QIE10DigiCollectionDM_  = ps.getParameter<std::string>("QIE10DigiCollectionDM");
      QIE11DigiCollectionDM_  = ps.getParameter<std::string>("QIE11DigiCollectionDM");

      produces< HBHEDigiCollection >();
      produces< HODigiCollection >();
      produces< HFDigiCollection >();
      produces< ZDCDigiCollection >();

      produces<QIE10DigiCollection>("HFQIE10DigiCollection");
      produces<QIE11DigiCollection>("HBHEQIE11DigiCollection");

      if(ps.getParameter<bool>("debugCaloSamples")){
        produces<CaloSamplesCollection>("HcalSamples");
      }
      if(ps.getParameter<bool>("injectTestHits")){
        produces<edm::PCaloHitContainer>("HcalHits");
      }

      if(MergeHcalDigisProd_) HcalDigiWorkerProd_ = new DataMixingHcalDigiWorkerProd(ps, consumesCollector());
      else HcalDigiWorker_ = new DataMixingHcalDigiWorker(ps, consumesCollector());

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
    }

    // Pixels

    PixelDigiCollectionDM_  = ps.getParameter<std::string>("PixelDigiCollectionDM");

    produces< edm::DetSetVector<PixelDigi> > (PixelDigiCollectionDM_);

    SiPixelWorker_ = new DataMixingSiPixelWorker(ps, consumesCollector());


    // Pileup Information: if doing pre-mixing, we have to save the pileup information from the Secondary stream

    MergePileup_ = ps.getParameter<bool>("MergePileupInfo");

    if(MergePileup_) {
      produces< std::vector<PileupSummaryInfo> >();
      produces< int >("bunchSpacing");
      produces<CrossingFramePlaybackInfoNew>();

      std::vector<edm::InputTag> GenPUProtonsInputTags;
      GenPUProtonsInputTags = ps.getParameter<std::vector<edm::InputTag> >("GenPUProtonsInputTags");
      for(std::vector<edm::InputTag>::const_iterator it_InputTag = GenPUProtonsInputTags.begin(); 
                                                     it_InputTag != GenPUProtonsInputTags.end(); ++it_InputTag) 
         produces< std::vector<reco::GenParticle> >( it_InputTag->label() );

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
  }
  

  void DataMixingModule::beginRun(edm::Run const& run, const edm::EventSetup& ES) { 
    BMixingModule::beginRun( run, ES);
  }

  void DataMixingModule::endRun(edm::Run const& run, const edm::EventSetup& ES) { 
    BMixingModule::endRun( run, ES);
  }

  // Virtual destructor needed.
  DataMixingModule::~DataMixingModule() { 
    if(MergeEMDigis_){ 
      delete EMDigiWorker_;
    }    
    else {delete EMWorker_;}
    if(MergeHcalDigis_) { 
      if(MergeHcalDigisProd_) { delete HcalDigiWorkerProd_;}
      else { delete HcalDigiWorker_; }}
    else {delete HcalWorker_;}
    if(MuonWorker_) delete MuonWorker_;
    if(useSiStripRawDigi_)
      delete SiStripRawWorker_;
    else delete SiStripWorker_;
    delete SiPixelWorker_;
    if(MergePileup_) { delete PUWorker_;}
  }

  void DataMixingModule::addSignals(const edm::Event &e, const edm::EventSetup& ES) { 
    // fill in maps of hits

    LogDebug("DataMixingModule")<<"===============> adding MC signals for "<<e.id();

    // Ecal
    if(MergeEMDigis_) { 
      EMDigiWorker_->addEMSignals(e, ES);
    }
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

    // SiStrips
    if(useSiStripRawDigi_) SiStripRawWorker_->addSiStripSignals(e);
    else SiStripWorker_->addSiStripSignals(e);

    // SiPixels
    SiPixelWorker_->addSiPixelSignals(e);
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

    // check and see if we need to copy the pileup information from 
    // secondary stream to the output stream  
    // We only have the pileup event here, so pick the first time and store the info


    if(MergePileup_ && !AddedPileup_){
      
      PUWorker_->addPileupInfo(&ep, eventNr, &moduleCallingContext);

      AddedPileup_ = true;
    }

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    // Ecal
    if(MergeEMDigis_) {  
      EMDigiWorker_->addEMPileups(bcr, &ep, eventNr, ES, &moduleCallingContext);
    }
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

    // SiStrips
    if(useSiStripRawDigi_) SiStripRawWorker_->addSiStripPileups(bcr, &ep, eventNr, &moduleCallingContext);
    else SiStripWorker_->addSiStripPileups(bcr, &ep, eventNr, &moduleCallingContext);

    // SiPixels
    //whoops this should be for the MC worker ????? SiPixelWorker_->setPileupInfo(ps,bunchSpacing);
    SiPixelWorker_->addSiPixelPileups(bcr, &ep, eventNr, &moduleCallingContext);
  }


  
  void DataMixingModule::doPileUp(edm::Event &e, const edm::EventSetup& ES)
  {
    using namespace std::placeholders;

    std::vector<edm::SecondaryEventIDAndFileInfo> recordEventID;
    std::vector<int> PileupList;
    PileupList.clear();
    TrueNumInteractions_.clear();

    ModuleCallingContext const* mcc = e.moduleCallingContext();

    for (int bunchCrossing=minBunch_;bunchCrossing<=maxBunch_;++bunchCrossing) {
      for (unsigned int isource=0;isource<maxNbSources_;++isource) {
        std::shared_ptr<PileUp> source = inputSources_[isource];
        if (!source || !(source->doPileUp(bunchCrossing))) 
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
                std::bind(&DataMixingModule::pileWorker, std::ref(*this),
			  _1, bunchCrossing, _2, std::cref(ES), mcc),
		NumPU_Events,
                e.streamID()
			   );
      }
    }

  }


  void DataMixingModule::put(edm::Event &e,const edm::EventSetup& ES) {

    // individual workers...

    // move pileup first so we have access to the information for the put step

    std::vector<PileupSummaryInfo> ps;
    int bunchSpacing=10000;

    if(MergePileup_) { 
      PUWorker_->getPileupInfo(ps,bunchSpacing);      
      PUWorker_->putPileupInfo(e);
    }

    // Ecal
    if(MergeEMDigis_) {
      EMDigiWorker_->putEM(e,ES);
    }
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

    // SiStrips
    if(useSiStripRawDigi_) SiStripRawWorker_->putSiStrip(e);
    else SiStripWorker_->putSiStrip(e);

    // SiPixels
    SiPixelWorker_->putSiPixel(e);
  }

  void DataMixingModule::beginLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) {
    BMixingModule::beginLuminosityBlock(l1, c);
  }

  void DataMixingModule::endLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) {
    BMixingModule::endLuminosityBlock(l1, c);
  }


} //edm
