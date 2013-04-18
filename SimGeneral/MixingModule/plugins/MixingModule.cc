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
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoExtended.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "MixingModule.h"
#include "MixingWorker.h"

#include "FWCore/Services/src/Memory.h"
using namespace std;

namespace edm
{

  // Constructor 
  MixingModule::MixingModule(const edm::ParameterSet& ps_mix) : 
  BMixingModule(ps_mix),
  labelPlayback_(ps_mix.getParameter<std::string>("LabelPlayback")),
  mixProdStep2_(ps_mix.getParameter<bool>("mixProdStep2")),
  mixProdStep1_(ps_mix.getParameter<bool>("mixProdStep1"))
  {
    if (!mixProdStep1_ && !mixProdStep2_) LogInfo("MixingModule") << " The MixingModule was run in the Standard mode.";
    if (mixProdStep1_) LogInfo("MixingModule") << " The MixingModule was run in the Step1 mode. It produces a mixed secondary source.";
    if (mixProdStep2_) LogInfo("MixingModule") << " The MixingModule was run in the Step2 mode. It uses a mixed secondary source.";
     
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
	  
	  if (mixProdStep2_){
	  //SimTracks
          if (object=="SimTrackPCrossingFrame") {
            InputTag tag;
	    if (tags.size()>0) tag=tags[0];
            std::string label;
            branchesActivate(TypeID(typeid(PCrossingFrame<SimTrack>)).friendlyClassName(),std::string(""),tag,label);
           
	    LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;

	    //---------------------------------
	    // set an appropriate label for the CrossingFrame
	    if (find(names.begin(), names.end(), "mixTracks") != names.end())
            { 
	       edm::ParameterSet psin=ps.getParameter<edm::ParameterSet>("mixTracks");
	       if (!psin.exists("type"))  continue; //to allow replacement by empty pset
	       std::string object = psin.getParameter<std::string>("type");
	       std::vector<InputTag>  tags=psin.getParameter<std::vector<InputTag> >("input");
	
	       InputTag tagCF;
	       if (tags.size()>0) tagCF=tags[0];
               std::string labelCF;
	   
	       branchesActivate(TypeID(typeid(std::vector<SimTrack>)).friendlyClassName(),std::string(""),tagCF,labelCF);
	       workersObjects_.push_back(new MixingWorker<SimTrack>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,labelCF,maxNbSources_,tag,tagCF,mixProdStep2_));  
	       
	       produces<CrossingFrame<SimTrack> >(labelCF);
	    }	    
	    //--------------------------------	    

          }else if (object=="SimVertexPCrossingFrame") {
            InputTag tag;
	    if (tags.size()>0) tag=tags[0];
            std::string label;
            branchesActivate(TypeID(typeid(PCrossingFrame<SimVertex>)).friendlyClassName(),std::string(""),tag,label);

	    LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag "<<tag.encode()<<", label will be "<<label;
	    
	    //---------------------------------
	    // set an appropriate label for the CrossingFrame
	    if (find(names.begin(), names.end(), "mixVertices") != names.end())
            { 
               edm::ParameterSet psin=ps.getParameter<edm::ParameterSet>("mixVertices");
	       if (!psin.exists("type"))  continue; //to allow replacement by empty pset
	       std::string object = psin.getParameter<std::string>("type");
	       std::vector<InputTag>  tags=psin.getParameter<std::vector<InputTag> >("input");
	       
	       InputTag tagCF;
	       if (tags.size()>0) tagCF=tags[0];
               std::string labelCF;
           
               branchesActivate(TypeID(typeid(std::vector<SimVertex>)).friendlyClassName(),std::string(""),tagCF,labelCF);
	       workersObjects_.push_back(new MixingWorker<SimVertex>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,labelCF,maxNbSources_,tag,tagCF,mixProdStep2_));  

	       produces<CrossingFrame<SimVertex> >(labelCF);
	    }
	    //---------------------------------
	  }

	  else if (object=="HepMCProductPCrossingFrame") {
            InputTag tag;
	    if (tags.size()>0) tag=tags[0];
            std::string label;
            branchesActivate(TypeID(typeid(PCrossingFrame<HepMCProduct>)).friendlyClassName(),std::string(""),tag,label);

	    LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	    
	    //---------------------------------
	    // set an appropriate label for the CrossingFrame
	    if (find(names.begin(), names.end(), "mixHepMC") != names.end())
            {   
               edm::ParameterSet psin=ps.getParameter<edm::ParameterSet>("mixHepMC");
	       if (!psin.exists("type"))  continue; //to allow replacement by empty pset
	       std::string object = psin.getParameter<std::string>("type");
	       std::vector<InputTag>  tags=psin.getParameter<std::vector<InputTag> >("input");
	       
	       InputTag tagCF;
	       if (tags.size()>0) tagCF=tags[0];
               std::string labelCF;
	       	      
	       branchesActivate(TypeID(typeid(HepMCProduct)).friendlyClassName(),std::string(""),tagCF,labelCF);
	       workersObjects_.push_back(new MixingWorker<HepMCProduct>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,labelCF,maxNbSources_,tag,tagCF,mixProdStep2_));  

	       produces<CrossingFrame<HepMCProduct> >(labelCF);
	    }
	    //-------------------------------- 	
          }else if (object=="PCaloHitPCrossingFrame") {
            std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
	    for (unsigned int ii=0;ii<subdets.size();ii++) {
	      InputTag tag;
	      if (tags.size()==1) tag=tags[0];
              else if(tags.size()>1) tag=tags[ii];
	      std::string label;
	      branchesActivate(TypeID(typeid(PCrossingFrame<PCaloHit>)).friendlyClassName(),subdets[ii],tag,label);

	      LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	      
	      //---------------------------------
	      // set an appropriate label for the product CrossingFrame
	      if (find(names.begin(), names.end(), "mixCH") != names.end())
              {  
                 edm::ParameterSet psin=ps.getParameter<edm::ParameterSet>("mixCH");
	         if (!psin.exists("type"))  continue; //to allow replacement by empty pset
	         std::string object = psin.getParameter<std::string>("type");
	         std::vector<InputTag>  tags=psin.getParameter<std::vector<InputTag> >("input");
	       
	         InputTag tagCF;
	         if (tags.size()==1) tagCF=tags[0];
		 else if(tags.size()>1) tagCF=tags[ii];
                 std::string labelCF;
        
	         branchesActivate(TypeID(typeid(std::vector<PCaloHit>)).friendlyClassName(),subdets[ii],tagCF,labelCF);
	         workersObjects_.push_back(new MixingWorker<PCaloHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,labelCF,maxNbSources_,tag,tagCF,mixProdStep2_));  

		 produces<CrossingFrame<PCaloHit> > (labelCF);
	      }
	      //--------------------------------
	    }

          }else if (object=="PSimHitPCrossingFrame") {
	    std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
	    for (unsigned int ii=0;ii<subdets.size();ii++) {	      
	      InputTag tag;
	      if (tags.size()==1) tag=tags[0];
              else if(tags.size()>1) tag=tags[ii];
	      std::string label;
              branchesActivate(TypeID(typeid(PCrossingFrame<PSimHit>)).friendlyClassName(),subdets[ii],tag,label);
	   
	      LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	      
	      
	      //---------------------------------
	      // set an appropriate label for the CrossingFrame
	      if (find(names.begin(), names.end(), "mixSH") != names.end())
              { 
                 edm::ParameterSet psin=ps.getParameter<edm::ParameterSet>("mixSH");
	         if (!psin.exists("type"))  continue; //to allow replacement by empty pset
	         std::string object = psin.getParameter<std::string>("type");
	         std::vector<InputTag>  tags=psin.getParameter<std::vector<InputTag> >("input");
	       
	         InputTag tagCF;
	         if (tags.size()==1) tagCF=tags[0];
		 else if(tags.size()>1) tagCF=tags[ii];
                 std::string labelCF;
	         branchesActivate(TypeID(typeid(std::vector<PSimHit>)).friendlyClassName(),subdets[ii],tagCF,labelCF); 		 	   
		 workersObjects_.push_back(new MixingWorker<PSimHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,labelCF,maxNbSources_,tag,tagCF,mixProdStep2_));  	            
		
	         produces<CrossingFrame<PSimHit> > (labelCF);
	      }
	      //-------------------------------
	    }

	  }
	  }//mixProdStep2
	 
	  
	  if (!mixProdStep2_){
	  	
	  InputTag tagCF = InputTag();
	  std::string labelCF = " ";	  
	  
	  //SimTracks
          if (object=="SimTrack") {
            InputTag tag;
	    if (tags.size()>0) tag=tags[0];
            std::string label;

            branchesActivate(TypeID(typeid(std::vector<SimTrack>)).friendlyClassName(),std::string(""),tag,label);
	    workersObjects_.push_back(new MixingWorker<SimTrack>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,labelCF,maxNbSources_,tag,tagCF,mixProdStep2_));  

	    produces<CrossingFrame<SimTrack> >(label);
	    LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	    

          }else if (object=="SimVertex") {
            InputTag tag;
	    if (tags.size()>0) tag=tags[0];
            std::string label;

            branchesActivate(TypeID(typeid(std::vector<SimVertex>)).friendlyClassName(),std::string(""),tag,label);
	    
	    workersObjects_.push_back(new MixingWorker<SimVertex>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,labelCF,maxNbSources_,tag,tagCF,mixProdStep2_));  
	    produces<CrossingFrame<SimVertex> >(label);
	    LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag "<<tag.encode()<<", label will be "<<label;
	    
	  }else if (object=="HepMCProduct") {
            InputTag tag;
	    if (tags.size()>0) tag=tags[0];
            std::string label;

            branchesActivate(TypeID(typeid(HepMCProduct)).friendlyClassName(),std::string(""),tag,label);
	    workersObjects_.push_back(new MixingWorker<HepMCProduct>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,labelCF,maxNbSources_,tag,tagCF,mixProdStep2_));  
            
	    produces<CrossingFrame<HepMCProduct> >(label);
	    LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	    

          }else if (object=="PCaloHit") {
            std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
	    for (unsigned int ii=0;ii<subdets.size();ii++) {
	      InputTag tag;
	      if (tags.size()==1) tag=tags[0];
              else if(tags.size()>1) tag=tags[ii];
	      std::string label;

	      branchesActivate(TypeID(typeid(std::vector<PCaloHit>)).friendlyClassName(),subdets[ii],tag,label);
	      workersObjects_.push_back(new MixingWorker<PCaloHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,labelCF,maxNbSources_,tag,tagCF,mixProdStep2_));  

	      produces<CrossingFrame<PCaloHit> > (label);
	      LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	      
	    }

          }else if (object=="PSimHit") {
	    std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
	    for (unsigned int ii=0;ii<subdets.size();ii++) {
	      InputTag tag;
	      if (tags.size()==1) tag=tags[0];
              else if(tags.size()>1) tag=tags[ii];
	      std::string label;

              branchesActivate(TypeID(typeid(std::vector<PSimHit>)).friendlyClassName(),subdets[ii],tag,label);
              workersObjects_.push_back(new MixingWorker<PSimHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,labelCF,maxNbSources_,tag,tagCF,mixProdStep2_));  

              produces<CrossingFrame<PSimHit> > (label);
              LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	    }
	  }
	  else LogWarning("MixingModule") <<"You have asked to mix an unknown type of object("<<object<<").\n If you want to include it in mixing, please contact the authors of the MixingModule!";
         }//if for mixProdStep2
      }//while over the mixObjects parameters

    sort_all(wantedBranches_);
    for (unsigned int branch=0;branch<wantedBranches_.size();++branch) LogDebug("MixingModule")<<"Will keep branch "<<wantedBranches_[branch]<<" for mixing ";
  
    dropUnwantedBranches(wantedBranches_);

    produces<PileupMixingContent>();
    
    produces<CrossingFramePlaybackInfoExtended>();
  }
 

  void MixingModule::branchesActivate(const std::string &friendlyName, const std::string &subdet, InputTag &tag, std::string &label) {
       
    label=tag.label()+tag.instance();
    wantedBranches_.push_back(friendlyName + '_' +
			      tag.label() + '_' +
			      tag.instance());
        				  
    //if useCurrentProcessOnly, we have to change the input tag
    if (useCurrentProcessOnly_) {
      const std::string processName = edm::Service<edm::service::TriggerNamesService>()->getProcessName();
      tag = InputTag(tag.label(),tag.instance(),processName);
    }    
    
  }

  
  void MixingModule::checkSignal(const edm::Event &e){
    if (workers_.size()==0){
      for (unsigned int ii=0;ii<workersObjects_.size();ii++){
        if (workersObjects_[ii]->checkSignal(e)){
          workers_.push_back(workersObjects_[ii]);
        }
      }
    }
  }
  
  
  void MixingModule::createnewEDProduct() {
    //create playback info
    playbackInfo_=new CrossingFramePlaybackInfoExtended(minBunch_,maxBunch_,maxNbSources_); 

    //and CrossingFrames
    for (unsigned int ii=0;ii<workers_.size();ii++){
      workers_[ii]->createnewEDProduct();
    }  
  }
 

  // Virtual destructor needed.
  MixingModule::~MixingModule() { 
    for (unsigned int ii=0;ii<workersObjects_.size();ii++){ 
      delete workersObjects_[ii];}
    delete sel_;  
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

    std::auto_ptr< PileupMixingContent > PileupMixing_;
    
    std::vector<int> bunchCrossingList;
    std::vector<int> numInteractionList;
    
    //Makin' a list:
    for (int bunchCrossing=minBunch_;bunchCrossing<=maxBunch_;++bunchCrossing) {
      bunchCrossingList.push_back(bunchCrossing);
      if(!doit_[0]) {
        numInteractionList.push_back(0);
      }
      else {
        numInteractionList.push_back(((pileup_[0])[bunchCrossing-minBunch_]).size());
      }
    }

    
    PileupMixing_ = std::auto_ptr< PileupMixingContent >(new PileupMixingContent( bunchCrossingList,
										  numInteractionList));

    e.put(PileupMixing_);


    // we have to do the ToF transformation for PSimHits once all pileup has been added
      for (unsigned int ii=0;ii<workers_.size();ii++) {
        //not apply setTof in Step2 mode because it was done in the Step1
	if (!mixProdStep2_){ 
          workers_[ii]->setTof();
	}
        workers_[ii]->put(e);
      }
 }

  void MixingModule::addPileups(const int bcr, EventPrincipal *ep, unsigned int eventNr,unsigned int worker, const edm::EventSetup& setup) {    
  // fill in pileup part of CrossingFrame
    LogDebug("MixingModule") <<"\n===============> adding objects from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    workers_[worker]->addPileups(bcr,ep,eventNr,vertexoffset);
  }
  
  void MixingModule::setEventStartInfo(const unsigned int s) {
    playbackInfo_->setEventStartInfo(vectorEventIDs_,s); 
  }

  void MixingModule::put(edm::Event &e, const edm::EventSetup& setup) {

    if (playbackInfo_) {
      std::auto_ptr<CrossingFramePlaybackInfoExtended> pOut(playbackInfo_);
      e.put(pOut);
    }
  }
  
  void MixingModule::getEventStartInfo(edm::Event & e, const unsigned int s) {
    if (playback_) {
 
      edm::Handle<CrossingFramePlaybackInfoExtended>  playbackInfo_H;
      bool got=e.get((*sel_), playbackInfo_H); 
      if (got) {
	playbackInfo_H->getEventStartInfo(vectorEventIDs_,s);
      }else{
        throw cms::Exception("MixingProductNotFound") << " No CrossingFramePlaybackInfoExtended on the input file, but playback option set!!!!! Please change the input file if you really want playback!!!!!!"  << endl;    
      }
    }
  }
}//edm
