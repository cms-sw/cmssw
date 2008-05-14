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
  MixingModule::MixingModule(const edm::ParameterSet& ps_mix) : BMixingModule(ps_mix),labelPlayback_(ps_mix.getParameter<std::string>("LabelPlayback"))

  {
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
            if (verifyRegistry(object,std::string(""),tag,label));
	    {
	      workers_.push_back(new MixingWorker<SimTrack>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,maxNbSources_,tag));  
	      produces<CrossingFrame<SimTrack> >(label);
	      LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	    }

          }else if (object=="SimVertex") {
            InputTag tag;
	    if (tags.size()>0) tag=tags[0];
            std::string label;
            if (verifyRegistry(object,std::string(""),tag,label))
	    {
	      workers_.push_back(new MixingWorker<SimVertex>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,maxNbSources_,tag));  
	      produces<CrossingFrame<SimVertex> >(label);
	      LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag "<<tag.encode()<<", label will be "<<label;
	    }
	  }

	    else if (object=="HepMCProduct") {
            InputTag tag;
	    if (tags.size()>0) tag=tags[0];
            std::string label;
            if (verifyRegistry(object,std::string(""),tag,label)){
	    workers_.push_back(new MixingWorker<HepMCProduct>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,maxNbSources_,tag));  
	    produces<CrossingFrame<HepMCProduct> >(label);
	    LogInfo("MixingModule") <<"Will mix"<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	    }

          }else if (object=="PCaloHit") {
            std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
	    for (unsigned int ii=0;ii<subdets.size();ii++) {
	      InputTag tag;
	      if (tags.size()==1) tag=tags[0];
              else if(tags.size()>1) tag=tags[ii]; //FIXME: verify sizes
	      std::string label;
	      if (verifyRegistry(object,subdets[ii],tag,label)){
	      workers_.push_back(new MixingWorker<PCaloHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,maxNbSources_,tag));  
	      produces<CrossingFrame<PCaloHit> > (label);
	      LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	      }
	    }

          }else if (object=="PSimHit") {
	    std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
	    for (unsigned int ii=0;ii<subdets.size();ii++) {
	      InputTag tag;
	      if (tags.size()==1) tag=tags[0];
              else if(tags.size()>1) tag=tags[ii]; //FIXME: verify sizes
	      std::string label;
              if (!verifyRegistry(object,subdets[ii],tag,label)) continue;
	      if ((subdets[ii].find("HighTof")==std::string::npos) && (subdets[ii].find("LowTof")==std::string::npos)) {
		workers_.push_back(new MixingWorker<PSimHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,maxNbSources_,tag));  
		LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	      }else {
		workers_.push_back(new MixingWorker<PSimHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,maxNbSources_,tag,true));  
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
		workers_[workers_.size()-1]->setCheckTof(ps.getUntrackedParameter<bool>("checktof",true));
		LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	      }
	      produces<CrossingFrame<PSimHit> > (label);
	    }

	  }else LogWarning("MixingModule") <<"You have asked to mix an unknown type of object("<<object<<").\n If you want to include it in mixing, please contact the authors of the MixingModule!";
      }

    for (unsigned int branch=0;branch<wantedBranches_.size();++branch) LogDebug("MixingModule")<<"Will keep branch "<<wantedBranches_[branch];
  
    produces<CrossingFramePlaybackInfo>();
  }
 
  bool MixingModule::verifyRegistry(std::string object, std::string subdet, InputTag &tag,std::string &label) {
    // verify that the given product exists in the product registry
    // and create the label to be given to the CrossingFrame

    edm::Service<edm::ConstProductRegistry> reg;
    // Loop over provenance of products in registry.
    std::string lookfor;
    if (object=="HepMCProduct") lookfor="edm::"+object;//exception for HepMCProduct
    else if (object=="edm::HepMCProduct") lookfor=object;
    else  lookfor="std::vector<"+object+">";
    bool found=false;
    for (edm::ProductRegistry::ProductList::const_iterator it = reg->productList().begin();
	 it != reg->productList().end(); ++it) {
      // See FWCore/Framework/interface/BranchDescription.h
      // BranchDescription contains all the information for the product.
      edm::BranchDescription desc = it->second;
      if (desc.className()==lookfor && desc.moduleLabel()==tag.label() && desc.productInstanceName()==tag.instance()) {
	  	    label=desc.moduleLabel()+desc.productInstanceName();
	  found=true;
	  wantedBranches_.push_back(desc.branchName());
	  break; 
	}
    }
    if (!found) {
      LogWarning("MixingModule")<<"!!!!!!!!!Could not find in registry requested object: "<<object<<" with "<<tag<<".\nWill NOT be considered for mixing!!!!!!!!!";
      return false;
    }
    
    return true;
  }

//   Selector * MixingModule::createSelector(InputTag &tag){
//     //FIXME: how to distinguish in input tags between ="" and any?
//     Selector *sel=0;
//     if (tag.label()=="") {
//       if (tag.instance()=="")  sel = new Selector(MatchAllSelector());
//       else sel = new Selector(ProductInstanceNameSelector(tag.instance()));
//     } else {
//       if (tag.instance()=="") sel =new Selector(ModuleLabelSelector(tag.label()));
//       else sel =new Selector(ModuleLabelSelector(tag.label()) && ProductInstanceNameSelector(tag.instance()));
//     }
//     return sel;
//   }

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

  void MixingModule::addSignals(const edm::Event &e) { 
    // fill in signal part of CrossingFrame

    LogDebug("MixingModule")<<"===============> adding signals for "<<e.id();
    for (unsigned int ii=0;ii<workers_.size();ii++){ 
      workers_[ii]->addSignals(e);
    }

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
