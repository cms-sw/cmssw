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
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoNew.h"
#include "DataMixingModule.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"

using namespace std;

namespace edm
{

  // Constructor 
  DataMixingModule::DataMixingModule(const edm::ParameterSet& ps) : BMixingModule(ps),
    EBPileInputTag_(ps.getParameter<edm::InputTag>("EBPileInputTag")),
    EEPileInputTag_(ps.getParameter<edm::InputTag>("EEPileInputTag")),
    ESPileInputTag_(ps.getParameter<edm::InputTag>("ESPileInputTag")),
    HBHEPileInputTag_(ps.getParameter<edm::InputTag>("HBHEPileInputTag")),
    HOPileInputTag_(ps.getParameter<edm::InputTag>("HOPileInputTag")),
    HFPileInputTag_(ps.getParameter<edm::InputTag>("HFPileInputTag")),
    ZDCPileInputTag_(ps.getParameter<edm::InputTag>("ZDCPileInputTag")),
							    label_(ps.getParameter<std::string>("Label"))
  {  

    // prepare for data access in DataMixingEcalDigiWorkerProd
    tok_eb_ = consumes<EBDigitizerTraits::DigiCollection>(EBPileInputTag_);
    tok_ee_ = consumes<EEDigitizerTraits::DigiCollection>(EEPileInputTag_);
    tok_es_ = consumes<ESDigitizerTraits::DigiCollection>(ESPileInputTag_);

    // prepare for data access in DataMixingHcalDigiWorkerProd
    tok_hbhe_ = consumes<HBHEDigitizerTraits::DigiCollection>(HBHEPileInputTag_);
    tok_ho_ = consumes<HODigitizerTraits::DigiCollection>(HOPileInputTag_);
    tok_hf_ = consumes<HFDigitizerTraits::DigiCollection>(HFPileInputTag_);
    tok_zdc_ = consumes<ZDCDigitizerTraits::DigiCollection>(ZDCPileInputTag_);

    // get the subdetector names
    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // For now, list all of them here.  Later, make this selectable with input parameters
    // 

    // Check to see if we are working in Full or Fast Simulation

    MergeTrackerDigis_ = (ps.getParameter<std::string>("TrackerMergeType")).compare("Digis") == 0;
    MergeEMDigis_ = (ps.getParameter<std::string>("EcalMergeType")).compare("Digis") == 0;
    MergeHcalDigis_ = (ps.getParameter<std::string>("HcalMergeType")).compare("Digis") == 0;
    if(MergeHcalDigis_) MergeHcalDigisProd_ = (ps.getParameter<std::string>("HcalDigiMerge")=="FullProd");

    addMCDigiNoise_ = false;

    addMCDigiNoise_ = ps.getUntrackedParameter<bool>("addMCDigiNoise");  // for Sim on Sim mixing

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


      if(addMCDigiNoise_ ) {
	edm::ConsumesCollector iC(consumesCollector());
        EcalDigiWorkerProd_ = new DataMixingEcalDigiWorkerProd(ps, iC);
        EcalDigiWorkerProd_->setEBAccess(tok_eb_);
        EcalDigiWorkerProd_->setEEAccess(tok_ee_);
        EcalDigiWorkerProd_->setESAccess(tok_es_);
      }
      else { EMDigiWorker_ = new DataMixingEMDigiWorker(ps, consumesCollector()); }
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


    if(MergeTrackerDigis_) {

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
      
      if( addMCDigiNoise_ ) {
	SiPixelMCDigiWorker_ = new DataMixingSiPixelMCDigiWorker(ps, consumesCollector());
      }
      else {
	SiPixelWorker_ = new DataMixingSiPixelWorker(ps, consumesCollector());
      }

    } 
    else{
      //Tracks:
      edm::ConsumesCollector iC(consumesCollector());
      GeneralTrackWorker_ = DigiAccumulatorMixModFactory::get()->makeDigiAccumulator(ps.getParameterSet("tracker"), *this, iC).release();
    }

    // Pileup Information: if doing pre-mixing, we have to save the pileup information from the Secondary stream

    MergePileup_ = ps.getParameter<bool>("MergePileupInfo");

    if(MergePileup_) {
      produces< std::vector<PileupSummaryInfo> >();
      produces< int >("bunchSpacing");
      produces<CrossingFramePlaybackInfoNew>();

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
      if(MergeTrackerDigis_){
	SiStripMCDigiWorker_->initializeEvent( e, ES );
	SiPixelMCDigiWorker_->initializeEvent( e, ES );
      }
      else{
	GeneralTrackWorker_->initializeEvent(e,ES);
      }
      EcalDigiWorkerProd_->initializeEvent( e, ES );
    }
    if( addMCDigiNoise_ && MergeHcalDigisProd_) {
      HcalDigiWorkerProd_->initializeEvent( e, ES );
    }
  }
  

  void DataMixingModule::beginRun(edm::Run const& run, const edm::EventSetup& ES) { 
    BMixingModule::beginRun( run, ES);
    if( addMCDigiNoise_ ) {
      EcalDigiWorkerProd_->beginRun( ES );
      HcalDigiWorkerProd_->beginRun( run, ES );
    }
  }

  void DataMixingModule::endRun(edm::Run const& run, const edm::EventSetup& ES) { 
    //if( addMCDigiNoise_ ) {
      // HcalDigiWorkerProd_->endRun( run, ES ); // FIXME not implemented
      // EcalDigiWorkerProd_->endRun( ES );      // FIXME not implemented
    //}
    BMixingModule::endRun( run, ES);
  }

  // Virtual destructor needed.
  DataMixingModule::~DataMixingModule() { 
    if(MergeEMDigis_){ 
      if(addMCDigiNoise_ ) {delete EcalDigiWorkerProd_;}
      else {delete EMDigiWorker_;}
    }    
    else {delete EMWorker_;}
    if(MergeHcalDigis_) { 
      if(MergeHcalDigisProd_) { delete HcalDigiWorkerProd_;}
      else { delete HcalDigiWorker_; }}
    else {delete HcalWorker_;}
    if(MuonWorker_) delete MuonWorker_;
    if(MergeTrackerDigis_){
      if(useSiStripRawDigi_)
	delete SiStripRawWorker_;
      else if(addMCDigiNoise_ ) delete SiStripMCDigiWorker_;
      else delete SiStripWorker_;
      if(addMCDigiNoise_ ) delete SiPixelMCDigiWorker_;
      else delete SiPixelWorker_;
    }
    else{
      delete GeneralTrackWorker_;
    }
    if(MergePileup_) { delete PUWorker_;}
  }

  void DataMixingModule::addSignals(const edm::Event &e, const edm::EventSetup& ES) { 
    // fill in maps of hits

    LogDebug("DataMixingModule")<<"===============> adding MC signals for "<<e.id();

    // Ecal
    if(MergeEMDigis_) { 
      if(addMCDigiNoise_ ){ EcalDigiWorkerProd_->addEcalSignals(e, ES);}
      else {EMDigiWorker_->addEMSignals(e, ES); }
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

    if(MergeTrackerDigis_){
      // SiStrips
      if(useSiStripRawDigi_) SiStripRawWorker_->addSiStripSignals(e);
      else if(addMCDigiNoise_ ) SiStripMCDigiWorker_->addSiStripSignals(e);
      else SiStripWorker_->addSiStripSignals(e);
      
      // SiPixels
      if(addMCDigiNoise_ ) SiPixelMCDigiWorker_->addSiPixelSignals(e);
      else SiPixelWorker_->addSiPixelSignals(e);
    }else{
      //GeneralTrackWorker_->addGeneralTrackSignal(e);
      GeneralTrackWorker_->accumulate(e,ES);
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
      if(addMCDigiNoise_ ) { EcalDigiWorkerProd_->addEcalPileups(bcr, &ep, eventNr, ES, &moduleCallingContext);}
      else { EMDigiWorker_->addEMPileups(bcr, &ep, eventNr, ES, &moduleCallingContext);}
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

    if(MergeTrackerDigis_){      
      // SiStrips
      if(useSiStripRawDigi_) SiStripRawWorker_->addSiStripPileups(bcr, &ep, eventNr, &moduleCallingContext);
      else if(addMCDigiNoise_ ) SiStripMCDigiWorker_->addSiStripPileups(bcr, &ep, eventNr, &moduleCallingContext);
      else SiStripWorker_->addSiStripPileups(bcr, &ep, eventNr, &moduleCallingContext);
      
      // SiPixels
      //whoops this should be for the MC worker ????? SiPixelWorker_->setPileupInfo(ps,bunchSpacing);
      if(addMCDigiNoise_ ) SiPixelMCDigiWorker_->addSiPixelPileups(bcr, &ep, eventNr, &moduleCallingContext);
      else SiPixelWorker_->addSiPixelPileups(bcr, &ep, eventNr, &moduleCallingContext);
    }else{
      PileUpEventPrincipal pep(ep,&moduleCallingContext,bcr);
      GeneralTrackWorker_->accumulate(pep, ES,ep.streamID());
    }
    
    
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

	if(!MergeTrackerDigis_)
	  GeneralTrackWorker_->initializeBunchCrossing(e, ES, bunchCrossing);

        source->readPileUp(
                e.id(),
                recordEventID,
                std::bind(&DataMixingModule::pileWorker, std::ref(*this),
			  _1, bunchCrossing, _2, std::cref(ES), mcc),
		NumPU_Events,
                e.streamID()
			   );

	if(!MergeTrackerDigis_)
	  GeneralTrackWorker_->finalizeBunchCrossing(e, ES, bunchCrossing);
	
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
      if(addMCDigiNoise_ ) {EcalDigiWorkerProd_->putEcal(e,ES);}
      else { EMDigiWorker_->putEM(e,ES);}
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

    if(MergeTrackerDigis_){
       // SiStrips
      if(useSiStripRawDigi_) SiStripRawWorker_->putSiStrip(e);
      else if(addMCDigiNoise_ ) SiStripMCDigiWorker_->putSiStrip(e, ES);
      else SiStripWorker_->putSiStrip(e);
       
       // SiPixels
      if(addMCDigiNoise_ ) SiPixelMCDigiWorker_->putSiPixel(e, ES, ps, bunchSpacing); 
      else SiPixelWorker_->putSiPixel(e);
    }else{
      GeneralTrackWorker_->finalizeEvent(e,ES);
    }


  }

  void DataMixingModule::beginLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) {
    BMixingModule::beginLuminosityBlock(l1, c);
    EcalDigiWorkerProd_->beginLuminosityBlock(l1,c);
  }

  void DataMixingModule::endLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) {
    // EcalDigiWorkerProd_->endLuminosityBlock(l1,c);  // FIXME Not implemented.
    BMixingModule::endLuminosityBlock(l1, c);
  }


} //edm
