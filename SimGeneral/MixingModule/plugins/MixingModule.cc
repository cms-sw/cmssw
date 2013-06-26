// File: MixingModule.cc
// Description:  see MixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau, Bill Tanenbaum
//
//--------------------------------------------

#include "MixingModule.h"
#include "MixingWorker.h"

#include "CondFormats/RunInfo/interface/MixingModuleConfig.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoExtended.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

namespace edm {

  // Constructor
  MixingModule::MixingModule(const edm::ParameterSet& ps_mix) :
  BMixingModule(ps_mix),
  inputTagPlayback_(),
  mixProdStep2_(ps_mix.getParameter<bool>("mixProdStep2")),
  mixProdStep1_(ps_mix.getParameter<bool>("mixProdStep1")),
  digiAccumulators_()
  {
    if (!mixProdStep1_ && !mixProdStep2_) LogInfo("MixingModule") << " The MixingModule was run in the Standard mode.";
    if (mixProdStep1_) LogInfo("MixingModule") << " The MixingModule was run in the Step1 mode. It produces a mixed secondary source.";
    if (mixProdStep2_) LogInfo("MixingModule") << " The MixingModule was run in the Step2 mode. It uses a mixed secondary source.";

    useCurrentProcessOnly_=false;
    if (ps_mix.exists("useCurrentProcessOnly")) {
      useCurrentProcessOnly_=ps_mix.getParameter<bool>("useCurrentProcessOnly");
      LogInfo("MixingModule") <<" using given Parameter 'useCurrentProcessOnly' ="<<useCurrentProcessOnly_;
    }
    std::string labelPlayback;    
    if (ps_mix.exists("LabelPlayback")) {
      labelPlayback = ps_mix.getParameter<std::string>("LabelPlayback");
    }
    if (labelPlayback.empty()) {
      labelPlayback = ps_mix.getParameter<std::string>("@module_label");
    }
    inputTagPlayback_ = InputTag(labelPlayback, "");

    ParameterSet ps=ps_mix.getParameter<ParameterSet>("mixObjects");
    std::vector<std::string> names = ps.getParameterNames();
    for(std::vector<std::string>::iterator it=names.begin();it!= names.end();++it) {
      ParameterSet pset=ps.getParameter<ParameterSet>((*it));
      if (!pset.exists("type")) continue; //to allow replacement by empty pset
      std::string object = pset.getParameter<std::string>("type");
      std::vector<InputTag> tags=pset.getParameter<std::vector<InputTag> >("input");

      //if (!mixProdStep2_) {

          InputTag tagCF = InputTag();
          std::string labelCF = " ";

          if (object=="SimTrack") {
            InputTag tag;
            if (tags.size()>0) tag=tags[0];
            std::string label;

            branchesActivate(TypeID(typeid(std::vector<SimTrack>)).friendlyClassName(),std::string(""),tag,label);
            bool makeCrossingFrame = pset.getUntrackedParameter<bool>("makeCrossingFrame", false);
            if(makeCrossingFrame) {
              workersObjects_.push_back(new MixingWorker<SimTrack>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,labelCF,maxNbSources_,tag,tagCF));
              produces<CrossingFrame<SimTrack> >(label);
            }

            LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
            //            std::cout <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label<<std::endl;

          } else if (object=="RecoTrack") {
            InputTag tag;
            if (tags.size()>0) tag=tags[0];
            std::string label;

            branchesActivate(TypeID(typeid(std::vector<reco::Track>)).friendlyClassName(),std::string(""),tag,label);
	    // note: no crossing frame is foreseen to be used for this object type

	    LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
	    //std::cout <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label<<std::endl;

          } else if (object=="SimVertex") {
            InputTag tag;
            if (tags.size()>0) tag=tags[0];
            std::string label;

            branchesActivate(TypeID(typeid(std::vector<SimVertex>)).friendlyClassName(),std::string(""),tag,label);
            bool makeCrossingFrame = pset.getUntrackedParameter<bool>("makeCrossingFrame", false);
            if(makeCrossingFrame) {
              workersObjects_.push_back(new MixingWorker<SimVertex>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,labelCF,maxNbSources_,tag,tagCF));
              produces<CrossingFrame<SimVertex> >(label);
            }

            LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag "<<tag.encode()<<", label will be "<<label;
            //            std::cout <<"Will mix "<<object<<"s with InputTag "<<tag.encode()<<", label will be "<<label<<std::endl;

          } else if (object=="HepMCProduct") {
            InputTag tag;
            if (tags.size()>0) tag=tags[0];
            std::string label;

            branchesActivate(TypeID(typeid(HepMCProduct)).friendlyClassName(),std::string(""),tag,label);
            bool makeCrossingFrame = pset.getUntrackedParameter<bool>("makeCrossingFrame", false);
            if(makeCrossingFrame) {
              workersObjects_.push_back(new MixingWorker<HepMCProduct>(minBunch_,maxBunch_,bunchSpace_,std::string(""),label,labelCF,maxNbSources_,tag,tagCF));
              produces<CrossingFrame<HepMCProduct> >(label);
            }

            LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
            //            std::cout <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label<<std::endl;

          } else if (object=="PCaloHit") {
            std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
            std::vector<std::string> crossingFrames=pset.getUntrackedParameter<std::vector<std::string> >("crossingFrames", std::vector<std::string>());
            sort_all(crossingFrames);
            for (unsigned int ii=0;ii<subdets.size();++ii) {
              InputTag tag;
              if (tags.size()==1) tag=tags[0];
              else if(tags.size()>1) tag=tags[ii];
              std::string label;

              branchesActivate(TypeID(typeid(std::vector<PCaloHit>)).friendlyClassName(),subdets[ii],tag,label);
              if(binary_search_all(crossingFrames, tag.instance())) {
                workersObjects_.push_back(new MixingWorker<PCaloHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,labelCF,maxNbSources_,tag,tagCF));
                produces<CrossingFrame<PCaloHit> >(label);
              }

              LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
              //              std::cout <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label<<std::endl;

            }

          } else if (object=="PSimHit") {
            std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
            std::vector<std::string> crossingFrames=pset.getUntrackedParameter<std::vector<std::string> >("crossingFrames", std::vector<std::string>());
            sort_all(crossingFrames);
            for (unsigned int ii=0;ii<subdets.size();++ii) {
              InputTag tag;
              if (tags.size()==1) tag=tags[0];
              else if(tags.size()>1) tag=tags[ii];
              std::string label;

              branchesActivate(TypeID(typeid(std::vector<PSimHit>)).friendlyClassName(),subdets[ii],tag,label);
              if(binary_search_all(crossingFrames, tag.instance())) {
                workersObjects_.push_back(new MixingWorker<PSimHit>(minBunch_,maxBunch_,bunchSpace_,subdets[ii],label,labelCF,maxNbSources_,tag,tagCF));
                produces<CrossingFrame<PSimHit> >(label);
              }

              LogInfo("MixingModule") <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label;
              //              std::cout <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label<<std::endl;
            }
          } else {
            LogWarning("MixingModule") <<"You have asked to mix an unknown type of object("<<object<<").\n If you want to include it in mixing, please contact the authors of the MixingModule!";
          }
      //} //if for mixProdStep2
    }//while over the mixObjects parameters

    sort_all(wantedBranches_);
    for (unsigned int branch=0;branch<wantedBranches_.size();++branch) LogDebug("MixingModule")<<"Will keep branch "<<wantedBranches_[branch]<<" for mixing ";
dropUnwantedBranches(wantedBranches_);

    produces<PileupMixingContent>();

    produces<CrossingFramePlaybackInfoExtended>();

    // Create and configure digitizers
    createDigiAccumulators(ps_mix);
  }


  void MixingModule::createDigiAccumulators(const edm::ParameterSet& mixingPSet) {
    ParameterSet const& digiPSet = mixingPSet.getParameterSet("digitizers");
    std::vector<std::string> digiNames = digiPSet.getParameterNames();
    for(auto const& digiName : digiNames) {
        ParameterSet const& pset = digiPSet.getParameterSet(digiName);
        std::auto_ptr<DigiAccumulatorMixMod> accumulator = std::auto_ptr<DigiAccumulatorMixMod>(DigiAccumulatorMixModFactory::get()->makeDigiAccumulator(pset, *this));
        // Create appropriate DigiAccumulator
        if(accumulator.get() != 0) {
          digiAccumulators_.push_back(accumulator.release());
        }
    }
  }

  void MixingModule::reload(const edm::EventSetup & setup){
    //change the basic parameters.
    edm::ESHandle<MixingModuleConfig> config;
    setup.get<MixingRcd>().get(config);
    minBunch_=config->minBunch();
    maxBunch_=config->maxBunch();
    bunchSpace_=config->bunchSpace();
    //propagate to change the workers
    for (unsigned int ii=0;ii<workersObjects_.size();++ii){
      workersObjects_[ii]->reload(setup);
    }
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
    if (workers_.empty()){
      for (unsigned int ii=0;ii<workersObjects_.size();++ii){
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
    for (unsigned int ii=0;ii<workers_.size();++ii){
      workers_[ii]->createnewEDProduct();
    }
  }

  // Virtual destructor needed.
  MixingModule::~MixingModule() {
    for (unsigned int ii=0;ii<workersObjects_.size();++ii){
      delete workersObjects_[ii];
    }

    std::vector<DigiAccumulatorMixMod*>::const_iterator accItr = digiAccumulators_.begin();
    std::vector<DigiAccumulatorMixMod*>::const_iterator accEnd = digiAccumulators_.end();
    for (; accItr != accEnd; ++accItr) {
        delete *accItr;
    }
  }

  void MixingModule::addSignals(const edm::Event &e, const edm::EventSetup& setup) {

    LogDebug("MixingModule")<<"===============> adding signals for "<<e.id();

    accumulateEvent(e, setup);
    // fill in signal part of CrossingFrame
    for (unsigned int ii=0;ii<workers_.size();++ii) {
      workers_[ii]->addSignals(e);
    }
  }

  void MixingModule::pileAllWorkers(EventPrincipal const& eventPrincipal,
                                    int bunchCrossing, int eventId,
                                    int& vertexOffset,
                                    const edm::EventSetup& setup) {
    PileUpEventPrincipal pep(eventPrincipal, bunchCrossing, bunchSpace_, eventId, vertexOffset);
    accumulateEvent(pep, setup);

    for (unsigned int ii=0;ii<workers_.size();++ii) {
      LogDebug("MixingModule") <<" merging Event:  id " << eventPrincipal.id();
      //      std::cout <<"PILEALLWORKERS merging Event:  id " << eventPrincipal.id() << std::endl;

        workers_[ii]->addPileups(bunchCrossing,eventPrincipal, eventId, vertexoffset);
    }
  }

  void MixingModule::doPileUp(edm::Event &e, const edm::EventSetup& setup) {
    // Don't allocate because PileUp will do it for us.
    std::vector<edm::EventID> recordEventID;
    edm::Handle<CrossingFramePlaybackInfoExtended>  playbackInfo_H;
    if (playback_) {
      bool got = e.getByLabel(inputTagPlayback_, playbackInfo_H);
      if (!got) {
        throw cms::Exception("MixingProductNotFound") << " No "
          "CrossingFramePlaybackInfoExtended on the input file, but playback "
          "option set!!!!! Please change the input file if you really want "
          "playback!!!!!!"  << std::endl;
      }
    }

    // source[0] is "real" pileup.  Check to see that this is what we are doing.

    std::vector<int> PileupList;
    PileupList.clear();
    TrueNumInteractions_.clear();

    boost::shared_ptr<PileUp> source0 = inputSources_[0];

    if((source0 && source0->doPileUp() ) && !playback_) {
      //    if((!inputSources_[0] || !inputSources_[0]->doPileUp()) && !playback_ ) 

      // Pre-calculate all pileup distributions before we go fishing for events

      source0->CalculatePileup(minBunch_, maxBunch_, PileupList, TrueNumInteractions_);

    }

    //    for (int bunchIdx = minBunch_; bunchIdx <= maxBunch_; ++bunchIdx) {
    //  std::cout << " bunch ID, Pileup, True " << bunchIdx << " " << PileupList[bunchIdx-minBunch_] << " " <<  TrueNumInteractions_[bunchIdx-minBunch_] << std::endl;
    //}

    int KeepTrackOfPileup = 0;

    for (int bunchIdx = minBunch_; bunchIdx <= maxBunch_; ++bunchIdx) {
      for (size_t setBcrIdx=0; setBcrIdx<workers_.size(); ++setBcrIdx) {
        workers_[setBcrIdx]->setBcrOffset();
      }
      for(Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end(); accItr != accEnd; ++accItr) {
        (*accItr)->initializeBunchCrossing(e, setup, bunchIdx);
      }

      for (size_t readSrcIdx=0; readSrcIdx<maxNbSources_; ++readSrcIdx) {
        boost::shared_ptr<PileUp> source = inputSources_[readSrcIdx];   // this looks like we create
                                                                        // new PileUp objects for each
                                                                        // source for each event?
                                                                        // Why?
        for (size_t setSrcIdx=0; setSrcIdx<workers_.size(); ++setSrcIdx) {
          workers_[setSrcIdx]->setSourceOffset(readSrcIdx);
        }

        if (!source || !source->doPileUp()) continue;

        int NumPU_Events = 0;

        if(readSrcIdx ==0 && !playback_) {
           NumPU_Events = PileupList[bunchIdx - minBunch_];
        } else {
           NumPU_Events = 1;
        }  // non-minbias pileup only gets one event for now. Fix later if desired.

        //        int eventId = 0;
        int vertexOffset = 0;

        if (!playback_) {
          inputSources_[readSrcIdx]->readPileUp(e.id(), recordEventID,
            boost::bind(&MixingModule::pileAllWorkers, boost::ref(*this), _1, bunchIdx,
                        _2, vertexOffset, boost::ref(setup)), NumPU_Events
            );
          playbackInfo_->setStartEventId(recordEventID, readSrcIdx, bunchIdx, KeepTrackOfPileup);
          KeepTrackOfPileup+=NumPU_Events;
        } else {
          int dummyId = 0;
          const std::vector<edm::EventID>& playEventID =
            playbackInfo_H->getStartEventId(readSrcIdx, bunchIdx);
          if(readSrcIdx == 0) {
            PileupList.push_back(playEventID.size());
            TrueNumInteractions_.push_back(playEventID.size());
          }
          inputSources_[readSrcIdx]->playPileUp(
            playEventID,
            boost::bind(&MixingModule::pileAllWorkers, boost::ref(*this), _1, bunchIdx,
                        dummyId, vertexOffset, boost::ref(setup))
            );
        }
      }
      for(Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end(); accItr != accEnd; ++accItr) {
        (*accItr)->finalizeBunchCrossing(e, setup, bunchIdx);
      }
    }

    // Keep track of pileup accounting...

    std::auto_ptr<PileupMixingContent> PileupMixing_;

    std::vector<int> numInteractionList;
    std::vector<int> bunchCrossingList;
    std::vector<float> TrueInteractionList;

    //Makin' a list: Basically, we don't care about the "other" sources at this point.
    for (int bunchCrossing=minBunch_;bunchCrossing<=maxBunch_;++bunchCrossing) {
      bunchCrossingList.push_back(bunchCrossing);
      if(!inputSources_[0] || !inputSources_[0]->doPileUp()) {
        numInteractionList.push_back(0);
        TrueInteractionList.push_back(0);
      }
      else {
        numInteractionList.push_back(PileupList[bunchCrossing-minBunch_]);
        TrueInteractionList.push_back((TrueNumInteractions_)[bunchCrossing-minBunch_]);
      }
    }


    PileupMixing_ = std::auto_ptr<PileupMixingContent>(new PileupMixingContent(bunchCrossingList,
                                                                               numInteractionList,
                                                                               TrueInteractionList));

    e.put(PileupMixing_);

    // we have to do the ToF transformation for PSimHits once all pileup has been added
    for (unsigned int ii=0;ii<workers_.size();++ii) {
        workers_[ii]->setTof();
      workers_[ii]->put(e);
    }
 }

  void MixingModule::put(edm::Event &e, const edm::EventSetup& setup) {

    if (playbackInfo_) {
      std::auto_ptr<CrossingFramePlaybackInfoExtended> pOut(playbackInfo_);
      e.put(pOut);
    }
  }

  void MixingModule::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
    for(Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end(); accItr != accEnd; ++accItr) {
      (*accItr)->beginRun(run, setup);
    }
  }

  void MixingModule::endRun(edm::Run const& run, edm::EventSetup const& setup) {
    for(Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end(); accItr != accEnd; ++accItr) {
      (*accItr)->endRun(run, setup);
    }
  }

  void MixingModule::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
    for(Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end(); accItr != accEnd; ++accItr) {
      (*accItr)->beginLuminosityBlock(lumi, setup);
    }
  }

  void MixingModule::endLuminosityBlock(edm::LuminosityBlock const & lumi, edm::EventSetup const& setup) {
    for(Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end(); accItr != accEnd; ++accItr) {
      (*accItr)->endLuminosityBlock(lumi, setup);
    }
  }

  void
  MixingModule::initializeEvent(edm::Event const& event, edm::EventSetup const& setup) {
    for(Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end(); accItr != accEnd; ++accItr) {
      (*accItr)->initializeEvent(event, setup);
    }
  }

  void
  MixingModule::accumulateEvent(edm::Event const& event, edm::EventSetup const& setup) {
    for(Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end(); accItr != accEnd; ++accItr) {
      (*accItr)->accumulate(event, setup);
    }
  }

  void
  MixingModule::accumulateEvent(PileUpEventPrincipal const& event, edm::EventSetup const& setup) {
    for(Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end(); accItr != accEnd; ++accItr) {
      (*accItr)->accumulate(event, setup);
    }
  }

  void
  MixingModule::finalizeEvent(edm::Event& event, edm::EventSetup const& setup) {
    for(Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end(); accItr != accEnd; ++accItr) {
      (*accItr)->finalizeEvent(event, setup);
    }
  }
}//edm
