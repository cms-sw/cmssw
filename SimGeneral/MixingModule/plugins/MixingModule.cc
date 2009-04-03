// File: MixingModule.cc
// Description:  see MixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau
//
//--------------------------------------------

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfo.h"
#include "FWCore/Utilities/interface/FriendlyName.h"
#include "MixingModule.h"
#include "MixingWorker.h"

#include "FWCore/Services/src/Memory.h"
using namespace std;

namespace edm
{

  // Constructor 
  MixingModule::MixingModule(const edm::ParameterSet& ps_mix) : BMixingModule(ps_mix),labelPlayback_(ps_mix.getParameter<std::string>("LabelPlayback"))

  {
    useCurrentProcessOnly_=false;
    if (ps_mix.exists("useCurrentProcessOnly")) {
      useCurrentProcessOnly_=ps_mix.getParameter<bool>("useCurrentProcessOnly");
      LogInfo("MixingModule") <<" using given Parameter 'useCurrentProcessOnly' ="<<useCurrentProcessOnly_;
    }
    if (labelPlayback_.size()>0){
      sel_=new Selector( ModuleLabelSelector(labelPlayback_));
    }
    else {
      sel_=new Selector( MatchAllSelector());
    }

    ParameterSet ps=ps_mix.getParameter<ParameterSet>("mixObjects");
    std::vector<std::string> names = ps.getParameterNames();
    for (std::vector<string>::iterator it=names.begin();it!= names.end();++it)
      {
	  ParameterSet pset=ps.getParameter<ParameterSet>((*it));
          if (!pset.exists("type"))  continue; //to allow replacement by empty pset
	  std::string object = pset.getParameter<std::string>("type");
	  std::vector<InputTag>  tags=pset.getParameter<std::vector<InputTag> >("input");
	
	  //SimTracks
          if (object=="SimTrack") {
            InputTag tag;
	    if (tags.size()>0) tag=tags[0];
            std::string label;
            branchesActivate(object,std::string(""),tag,label);
	    
	    workers_.push_back(new MixingWorker<SimTrack>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,maxNbSources_,tag,checktof_));  
	    produces<CrossingFrame<SimTrack> >(label);
	    LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	    

          }else if (object=="SimVertex") {
            InputTag tag;
	    if (tags.size()>0) tag=tags[0];
            std::string label;
            branchesActivate(object,std::string(""),tag,label);
	    
	    workers_.push_back(new MixingWorker<SimVertex>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,maxNbSources_,tag,checktof_));  
	    produces<CrossingFrame<SimVertex> >(label);
	    LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag "<<tag.encode()<<", label will be "<<label;
	    
	  }

	    else if (object=="HepMCProduct") {
            InputTag tag;
	    if (tags.size()>0) tag=tags[0];
            std::string label;
            branchesActivate(object,std::string(""),tag,label);
	    workers_.push_back(new MixingWorker<HepMCProduct>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,maxNbSources_,tag,checktof_));  
	    produces<CrossingFrame<HepMCProduct> >(label);
	    LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	    

          }else if (object=="PCaloHit") {
            std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
	    for (unsigned int ii=0;ii<subdets.size();ii++) {
	      InputTag tag;
	      if (tags.size()==1) tag=tags[0];
              else if(tags.size()>1) tag=tags[ii]; //FIXME: verify sizes
	      std::string label;
	      branchesActivate(object,subdets[ii],tag,label);
	      workers_.push_back(new MixingWorker<PCaloHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,maxNbSources_,tag,checktof_));  
	      produces<CrossingFrame<PCaloHit> > (label);
	      LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	      
	    }

          }else if (object=="PSimHit") {
	    std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
	    for (unsigned int ii=0;ii<subdets.size();ii++) {
	      InputTag tag;
	      if (tags.size()==1) tag=tags[0];
              else if(tags.size()>1) tag=tags[ii]; //FIXME: verify sizes
	      std::string label;
              branchesActivate(object,subdets[ii],tag,label);
	      if ((subdets[ii].find("HighTof")==std::string::npos) && (subdets[ii].find("LowTof")==std::string::npos)) {
		workers_.push_back(new MixingWorker<PSimHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,maxNbSources_,tag,checktof_));  
		LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	      }else {
		workers_.push_back(new MixingWorker<PSimHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,maxNbSources_,tag,checktof_,true));  
		// here we have to give the opposite selector too (low for high, high for low)
		int slow=(subdets[ii]).find("LowTof");//FIXME: to be done before when creating trackerPids
		int iend=(subdets[ii]).size();
		std::string productInstanceNameOpp;
		if (slow>0) {
		  productInstanceNameOpp=tag.instance().substr(0,iend-6)+"HighTof";
		}else{
		  productInstanceNameOpp=tag.instance().substr(0,iend-7)+"LowTof";
		}
		InputTag tagOpp(tag.label(),productInstanceNameOpp,tag.process());
		workers_[workers_.size()-1]->setOppositeTag(tagOpp);
		workers_[workers_.size()-1]->setCheckTof(checktof_);
		LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	      }
	      produces<CrossingFrame<PSimHit> > (label);
	    }

	  }else LogWarning("MixingModule") <<"You have asked to mix an unknown type of object("<<object<<").\n If you want to include it in mixing, please contact the authors of the MixingModule!";
      }

    sort_all(wantedBranches_);
    for (unsigned int branch=0;branch<wantedBranches_.size();++branch) LogDebug("MixingModule")<<"Will keep branch "<<wantedBranches_[branch]<<" for mixing ";
  
    dropUnwantedBranches(wantedBranches_);
    produces<CrossingFramePlaybackInfo>();
  }
 
  void MixingModule::branchesActivate(std::string object, std::string subdet, InputTag &tag,std::string &label) {
       
    label=tag.label()+tag.instance();
    wantedBranches_.push_back(edm::friendlyname::friendlyName(object) + '_' +
			      tag.label() + '_' +
			      tag.instance());
    
    //if useCurrentProcessOnly, we have to change the input tag
    if (useCurrentProcessOnly_) {
      const std::string processName = edm::Service<edm::service::TriggerNamesService>()->getProcessName();
      tag = InputTag(tag.label(),tag.instance(),processName);
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
    for (unsigned int ii=0;ii<workers_.size();ii++) 
      delete workers_[ii];
  }  

  void MixingModule::addSignals(const edm::Event &e, const edm::EventSetup& setup) { 
    // fill in signal part of CrossingFrame

    LogDebug("MixingModule")<<"===============> adding signals for "<<e.id();
    for (unsigned int ii=0;ii<workers_.size();ii++){ 
      workers_[ii]->addSignals(e);
    }

  }

  void MixingModule::doPileUp(edm::Event &e, const edm::EventSetup& setup)
  { //     we first loop over workers
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
	    merge(bunchCrossing, (pileup_[isource])[bunchCrossing-minBunch_],ii,setup);
	  }	
	}
      }
    }

    // we have to do the ToF transformation for PSimHits once all pileup has been added
     for (unsigned int ii=0;ii<workers_.size();ii++) {
      workers_[ii]->setTof();
      workers_[ii]->put(e);
     }
 }

  void MixingModule::addPileups(const int bcr, EventPrincipal *ep, unsigned int eventNr,unsigned int worker, const edm::EventSetup& setup) {    // fill in pileup part of CrossingFrame

    LogDebug("MixingModule") <<"\n===============> adding objects from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    workers_[worker]->addPileups(bcr,ep,eventNr,vertexoffset);
  }
  void MixingModule::setEventStartInfo(const unsigned int s) {
    playbackInfo_->setEventStartInfo(eventIDs_,fileSeqNrs_,nrEvents_,s); 
  }

  void MixingModule::put(edm::Event &e, const edm::EventSetup& setup) {

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
      }
    }
  }
}//edm
