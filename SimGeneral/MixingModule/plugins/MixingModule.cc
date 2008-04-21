// File: MixingModule.cc
// Description:  see MixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau
//
//--------------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfo.h"
#include "MixingModule.h"
#include "MixingWorker.h"


using namespace std;

namespace edm
{

  // Constructor 
  MixingModule::MixingModule(const edm::ParameterSet& ps) : BMixingModule(ps),
							    label_(ps.getParameter<std::string>("Label"))

  {
    // get the subdetector names
    this->getSubdetectorNames();

    // create input selector
    if (label_.size()>0){
      sel_=new Selector( ModuleLabelSelector(label_));
    }
    else {
      sel_=new Selector( MatchAllSelector());
    }

    // create worker for selected objects
    workers_.push_back(new MixingWorker<SimTrack>(minBunch_,maxBunch_,bunchSpace_,std::string(""),maxNbSources_,sel_));  
    produces<CrossingFrame<SimTrack> >();

    workers_.push_back(new MixingWorker<SimVertex>(minBunch_,maxBunch_,bunchSpace_,std::string(""),maxNbSources_,sel_)); 
    produces<CrossingFrame<SimVertex> >();

    workers_.push_back(new MixingWorker<edm::HepMCProduct>(minBunch_,maxBunch_,bunchSpace_,std::string(""),maxNbSources_,sel_));  
    produces<CrossingFrame<edm::HepMCProduct> >();

    //FIXME: no need to keep selectors for the moment!
    for (unsigned int ii=0;ii<caloSubdetectors_.size();ii++) {
      caloSelectors_.push_back(new Selector(*sel_ && ProductInstanceNameSelector(caloSubdetectors_[ii])));
      workers_.push_back(new MixingWorker<PCaloHit>(minBunch_,maxBunch_,bunchSpace_,caloSubdetectors_[ii],maxNbSources_,caloSelectors_[ii]));  
      produces<CrossingFrame<PCaloHit> > (caloSubdetectors_[ii]);
    }
    for (unsigned int ii=0;ii<nonTrackerPids_.size();ii++) {
      simSelectors_.push_back(new Selector(*sel_ && ProductInstanceNameSelector(nonTrackerPids_[ii])));
      workers_.push_back(new MixingWorker<PSimHit>(minBunch_,maxBunch_,bunchSpace_,nonTrackerPids_[ii],maxNbSources_,simSelectors_[ii]));  
      produces<CrossingFrame<PSimHit> > (nonTrackerPids_[ii]);
    }
    // we have to treat specially the tracker subdetectors
    // in order to correctly treat the High/low Tof business
    for (unsigned int ii=0;ii<trackerPids_.size();ii++) {

      simSelectors_.push_back(new Selector(*sel_ && ProductInstanceNameSelector(trackerPids_[ii])));
      workers_.push_back(new MixingWorker<PSimHit>(minBunch_,maxBunch_,bunchSpace_,trackerPids_[ii],maxNbSources_,simSelectors_[simSelectors_.size()-1],true)); 
      // here we have to give the opposite selector too (low for high, high for low)
      Selector * simSelectorOpp;//FIXME: memleak
      int slow=(trackerPids_[ii]).find("LowTof");//FIXME: to be done before when creating trackerPids
      int iend=(trackerPids_[ii]).size();
      if (slow>0) {
 	  std::string productInstanceNameOpp=trackerPids_[ii].substr(0,iend-6)+"HighTof";
	  simSelectorOpp=new Selector(*sel_ && ProductInstanceNameSelector(productInstanceNameOpp));
      }else{
 	  std::string productInstanceNameOpp=trackerPids_[ii].substr(0,iend-7)+"LowTof";
	  simSelectorOpp=new Selector(*sel_ && ProductInstanceNameSelector(productInstanceNameOpp));
      }
      workers_[workers_.size()-1]->setOppositeSel(simSelectorOpp);
      workers_[workers_.size()-1]->setCheckTof(ps.getUntrackedParameter<bool>("checktof",true));
    
      produces<CrossingFrame<PSimHit> > (trackerPids_[ii]);
    }

    produces<CrossingFramePlaybackInfo>();

  }

  void MixingModule::getSubdetectorNames() {
    // get subdetector names
    edm::Service<edm::ConstProductRegistry> reg;
    // Loop over provenance of products in registry.
    for (edm::ProductRegistry::ProductList::const_iterator it = reg->productList().begin();
	 it != reg->productList().end(); ++it) {
      // See FWCore/Framework/interface/BranchDescription.h
      // BranchDescription contains all the information for the product.
      edm::BranchDescription desc = it->second;
      if (!desc.friendlyClassName_.compare(0,9,"PCaloHits")) {
	caloSubdetectors_.push_back(desc.productInstanceName_);
	LogInfo("Constructor") <<"Adding calo container "<<desc.productInstanceName_ <<" for mixing";
      }
      else if (!desc.friendlyClassName_.compare(0,8,"PSimHits") && desc.productInstanceName_.compare(0,11,"TrackerHits")) {
	//	simHitSubdetectors_.push_back(desc.productInstanceName_);
	nonTrackerPids_.push_back(desc.productInstanceName_);
        LogInfo("MixingModule") <<"Adding non tracker simhit container "<<desc.productInstanceName_ <<" for mixing";
      }
      else if (!desc.friendlyClassName_.compare(0,8,"PSimHits") && !desc.productInstanceName_.compare(0,11,"TrackerHits")) {
	//	simHitSubdetectors_.push_back(desc.productInstanceName_);
	// here we store the tracker subdetector name  for low and high part
 	  trackerPids_.push_back(desc.productInstanceName_);
// 	int slow=(desc.productInstanceName_).find("LowTof");
// 	int iend=(desc.productInstanceName_).size();
//         if (slow>0) {
//  	  trackerPids_.push_back(desc.productInstanceName_.substr(0,iend-6));
// 	  LogInfo("MixingModule") <<"Adding tracker simhit container "<<desc.productInstanceName_.substr(0,iend-6) <<" for mixing";
//         }
      }
    }
  }
  void MixingModule::beginJob(edm::EventSetup const&iSetup) {
  }

  void MixingModule::createnewEDProduct() {
    //create playback info
    playbackInfo_=new CrossingFramePlaybackInfo(minBunch_,maxBunch_,maxNbSources_); 

    //and CrossingFrames
   for (unsigned int ii=0;ii<workers_.size();ii++) 
        workers_[ii]->createnewEDProduct();
  }
 

  // Virtual destructor needed.
  MixingModule::~MixingModule() { 
    delete sel_;
    for (unsigned int ii=0;ii<workers_.size();ii++) 
        delete workers_[ii];
  }  

  void MixingModule::addSignals(const edm::Event &e) { 
    // fill in signal part of CrossingFrame

    LogDebug("MixingModule")<<"===============> adding signals for "<<e.id();
    for (unsigned int ii=0;ii<workers_.size();ii++) 
        workers_[ii]->addSignals(e);

  }

  void MixingModule::doPileUp(edm::Event &e)
  {//     we first loop over workers
    // in order not to keep all CrossingFrames in memory simultaneously
    //

    for (unsigned int ii=0;ii<workers_.size();ii++) {
      // we have to loop over bunchcrossings first since added objects are all stored in one vector, 
      // ordered by bunchcrossing
      for (int bunchCrossing=minBunch_;bunchCrossing<=maxBunch_;++bunchCrossing) {
	workers_[ii]->setBcrOffset();
	for (unsigned int isource=0;isource<maxNbSources_;++isource) {
	  workers_[ii]->setSourceOffset(isource);
	  if (doit_[isource])   {
	    merge(bunchCrossing, (pileup_[isource])[bunchCrossing-minBunch_],ii);
	  }	
	}
      }
      workers_[ii]->put(e);
    }
  }

  void MixingModule::addPileups(const int bcr, Event *e, unsigned int eventNr,unsigned int worker) {    // fill in pileup part of CrossingFrame

  
    LogDebug("MixingModule") <<"\n===============> adding objects from event  "<<e->id()<<" for bunchcrossing "<<bcr;

        workers_[worker]->addPileups(bcr,e,eventNr,vertexoffset);
  }
  void MixingModule::setEventStartInfo(const unsigned int s) {
   playbackInfo_->setEventStartInfo(eventIDs_,fileSeqNrs_,nrEvents_,s); 
  }

  void MixingModule::put(edm::Event &e) {

   if (playbackInfo_) {
      std::auto_ptr<CrossingFramePlaybackInfo> pOut(playbackInfo_);
      e.put(pOut);
    }
  }
  
  void MixingModule::getEventStartInfo(edm::Event & e, const unsigned int s) {
    if (playback_) {
 
      edm::Handle<CrossingFramePlaybackInfo>  playbackInfo_H;
      bool got=e.get((*sel_), playbackInfo_H);
      if (got) {
	playbackInfo_H->getEventStartInfo(eventIDs_,fileSeqNrs_,nrEvents_,s);
      }else{
	LogWarning("MixingModule")<<"\n\nAttention: No CrossingFramePlaybackInfo on the input file, but playback option set!!!!!!!\nAttention: Job is executed without playback, please change the input file if you really want playback!!!!!!!";
	//FIXME: defaults
      }
    }
  }
}//edm
