// File: MixingModule.cc
// Description:  see MixingModule.h
// Author:  Ursula Berthon, LLR Palaiseau, Bill Tanenbaum
//
//--------------------------------------------

#include <functional>
#include <memory>

#include "MixingModule.h"
#include "MixingWorker.h"
#include "Adjuster.h"

#include "CondFormats/RunInfo/interface/MixingModuleConfig.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoExtended.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoNew.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace edm {

  // Constructor
  MixingModule::MixingModule(const edm::ParameterSet& ps_mix, MixingCache::Config const* globalConf)
      : BMixingModule(ps_mix, globalConf),
        inputTagPlayback_(),
        mixProdStep2_(ps_mix.getParameter<bool>("mixProdStep2")),
        mixProdStep1_(ps_mix.getParameter<bool>("mixProdStep1")),
        digiAccumulators_() {
    if (!mixProdStep1_ && !mixProdStep2_)
      LogInfo("MixingModule") << " The MixingModule was run in the Standard mode.";
    if (mixProdStep1_)
      LogInfo("MixingModule") << " The MixingModule was run in the Step1 mode. It produces a mixed secondary source.";
    if (mixProdStep2_)
      LogInfo("MixingModule") << " The MixingModule was run in the Step2 mode. It uses a mixed secondary source.";

    useCurrentProcessOnly_ = false;
    if (ps_mix.exists("useCurrentProcessOnly")) {
      useCurrentProcessOnly_ = ps_mix.getParameter<bool>("useCurrentProcessOnly");
      LogInfo("MixingModule") << " using given Parameter 'useCurrentProcessOnly' =" << useCurrentProcessOnly_;
    }
    std::string labelPlayback;
    if (ps_mix.exists("LabelPlayback")) {
      labelPlayback = ps_mix.getParameter<std::string>("LabelPlayback");
    }
    if (labelPlayback.empty()) {
      labelPlayback = ps_mix.getParameter<std::string>("@module_label");
    }
    if (playback_) {
      inputTagPlayback_ = InputTag(labelPlayback, "", edm::InputTag::kSkipCurrentProcess);
      consumes<CrossingFramePlaybackInfoNew>(inputTagPlayback_);
    }
    wrapLongTimes_ = false;
    if (ps_mix.exists("WrapLongTimes")) {
      wrapLongTimes_ = ps_mix.getParameter<bool>("WrapLongTimes");
    }

    skipSignal_ = false;
    if (ps_mix.exists("skipSignal")) {
      skipSignal_ = ps_mix.getParameter<bool>("skipSignal");
    }

    ParameterSet ps = ps_mix.getParameter<ParameterSet>("mixObjects");
    std::vector<std::string> names = ps.getParameterNames();
    for (std::vector<std::string>::iterator it = names.begin(); it != names.end(); ++it) {
      ParameterSet pset = ps.getParameter<ParameterSet>((*it));
      if (!pset.exists("type"))
        continue;  //to allow replacement by empty pset
      std::string object = pset.getParameter<std::string>("type");
      std::vector<InputTag> tags = pset.getParameter<std::vector<InputTag> >("input");

      //if (!mixProdStep2_) {

      InputTag tagCF = InputTag();
      std::string labelCF = " ";

      if (object == "SimTrack") {
        InputTag tag;
        if (!tags.empty())
          tag = tags[0];
        std::string label;

        branchesActivate(TypeID(typeid(std::vector<SimTrack>)).friendlyClassName(), std::string(""), tag, label);
        adjustersObjects_.push_back(new Adjuster<std::vector<SimTrack> >(tag, consumesCollector(), wrapLongTimes_));
        bool makeCrossingFrame = pset.getUntrackedParameter<bool>("makeCrossingFrame", false);
        if (makeCrossingFrame) {
          workersObjects_.push_back(new MixingWorker<SimTrack>(
              minBunch_, maxBunch_, bunchSpace_, std::string(""), label, labelCF, maxNbSources_, tag, tagCF));
          produces<CrossingFrame<SimTrack> >(label);
        }
        consumes<std::vector<SimTrack> >(tag);

        LogInfo("MixingModule") << "Will mix " << object << "s with InputTag= " << tag.encode() << ", label will be "
                                << label;
        //            std::cout <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label<<std::endl;

      } else if (object == "RecoTrack") {
        InputTag tag;
        if (!tags.empty())
          tag = tags[0];
        std::string label;

        branchesActivate(TypeID(typeid(std::vector<reco::Track>)).friendlyClassName(), std::string(""), tag, label);
        branchesActivate(
            TypeID(typeid(std::vector<reco::TrackExtra>)).friendlyClassName(), std::string(""), tag, label);
        branchesActivate(
            TypeID(typeid(edm::OwnVector<TrackingRecHit, edm::ClonePolicy<TrackingRecHit> >)).friendlyClassName(),
            std::string(""),
            tag,
            label);
        adjustersObjects_.push_back(
            new Adjuster<edm::OwnVector<TrackingRecHit> >(tag, consumesCollector(), wrapLongTimes_));
        // note: no crossing frame is foreseen to be used for this object type

        LogInfo("MixingModule") << "Will mix " << object << "s with InputTag= " << tag.encode() << ", label will be "
                                << label;
        //std::cout <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label<<std::endl;

      } else if (object == "SimVertex") {
        InputTag tag;
        if (!tags.empty())
          tag = tags[0];
        std::string label;

        branchesActivate(TypeID(typeid(std::vector<SimVertex>)).friendlyClassName(), std::string(""), tag, label);
        adjustersObjects_.push_back(new Adjuster<std::vector<SimVertex> >(tag, consumesCollector(), wrapLongTimes_));
        bool makeCrossingFrame = pset.getUntrackedParameter<bool>("makeCrossingFrame", false);
        if (makeCrossingFrame) {
          workersObjects_.push_back(new MixingWorker<SimVertex>(
              minBunch_, maxBunch_, bunchSpace_, std::string(""), label, labelCF, maxNbSources_, tag, tagCF));
          produces<CrossingFrame<SimVertex> >(label);
        }
        consumes<std::vector<SimVertex> >(tag);

        LogInfo("MixingModule") << "Will mix " << object << "s with InputTag " << tag.encode() << ", label will be "
                                << label;
        //            std::cout <<"Will mix "<<object<<"s with InputTag "<<tag.encode()<<", label will be "<<label<<std::endl;

      } else if (object == "HepMCProduct") {
        InputTag tag;
        if (!tags.empty())
          tag = tags[0];
        std::string label;

        branchesActivate(TypeID(typeid(HepMCProduct)).friendlyClassName(), std::string(""), tag, label);
        bool makeCrossingFrame = pset.getUntrackedParameter<bool>("makeCrossingFrame", false);
        if (makeCrossingFrame) {
          workersObjects_.push_back(new MixingWorker<HepMCProduct>(
              minBunch_, maxBunch_, bunchSpace_, std::string(""), label, labelCF, maxNbSources_, tag, tagCF, tags));
          produces<CrossingFrame<HepMCProduct> >(label);
        }
        consumes<HepMCProduct>(tag);

        LogInfo("MixingModule") << "Will mix " << object << "s with InputTag= " << tag.encode() << ", label will be "
                                << label;
        //            std::cout <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label<<std::endl;
        for (size_t i = 1; i < tags.size(); ++i) {
          InputTag fallbackTag = tags[i];
          std::string fallbackLabel;
          branchesActivate(
              TypeID(typeid(HepMCProduct)).friendlyClassName(), std::string(""), fallbackTag, fallbackLabel);
          mayConsume<HepMCProduct>(fallbackTag);
        }

      } else if (object == "PCaloHit") {
        std::vector<std::string> subdets = pset.getParameter<std::vector<std::string> >("subdets");
        std::vector<std::string> crossingFrames =
            pset.getUntrackedParameter<std::vector<std::string> >("crossingFrames", std::vector<std::string>());
        sort_all(crossingFrames);
        for (unsigned int ii = 0; ii < subdets.size(); ++ii) {
          InputTag tag;
          if (tags.size() == 1)
            tag = tags[0];
          else if (tags.size() > 1)
            tag = tags[ii];
          std::string label;

          branchesActivate(TypeID(typeid(std::vector<PCaloHit>)).friendlyClassName(), subdets[ii], tag, label);
          adjustersObjects_.push_back(new Adjuster<std::vector<PCaloHit> >(tag, consumesCollector(), wrapLongTimes_));
          if (binary_search_all(crossingFrames, tag.instance())) {
            workersObjects_.push_back(new MixingWorker<PCaloHit>(
                minBunch_, maxBunch_, bunchSpace_, subdets[ii], label, labelCF, maxNbSources_, tag, tagCF));
            produces<CrossingFrame<PCaloHit> >(label);
            consumes<std::vector<PCaloHit> >(tag);
          }

          LogInfo("MixingModule") << "Will mix " << object << "s with InputTag= " << tag.encode() << ", label will be "
                                  << label;
          //              std::cout <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label<<std::endl;
        }

      } else if (object == "PSimHit") {
        std::vector<std::string> subdets = pset.getParameter<std::vector<std::string> >("subdets");
        std::vector<std::string> crossingFrames =
            pset.getUntrackedParameter<std::vector<std::string> >("crossingFrames", std::vector<std::string>());
        sort_all(crossingFrames);
        std::vector<std::string> pcrossingFrames =
            pset.getUntrackedParameter<std::vector<std::string> >("pcrossingFrames", std::vector<std::string>());
        sort_all(pcrossingFrames);
        for (unsigned int ii = 0; ii < subdets.size(); ++ii) {
          InputTag tag;
          if (tags.size() == 1)
            tag = tags[0];
          else if (tags.size() > 1)
            tag = tags[ii];
          std::string label;

          branchesActivate(TypeID(typeid(std::vector<PSimHit>)).friendlyClassName(), subdets[ii], tag, label);
          adjustersObjects_.push_back(new Adjuster<std::vector<PSimHit> >(tag, consumesCollector(), wrapLongTimes_));
          if (binary_search_all(crossingFrames, tag.instance())) {
            bool makePCrossingFrame = binary_search_all(pcrossingFrames, tag.instance());
            workersObjects_.push_back(new MixingWorker<PSimHit>(minBunch_,
                                                                maxBunch_,
                                                                bunchSpace_,
                                                                subdets[ii],
                                                                label,
                                                                labelCF,
                                                                maxNbSources_,
                                                                tag,
                                                                tagCF,
                                                                makePCrossingFrame));
            produces<CrossingFrame<PSimHit> >(label);
            if (makePCrossingFrame) {
              produces<PCrossingFrame<PSimHit> >(label);
            }
            consumes<std::vector<PSimHit> >(tag);
          }

          LogInfo("MixingModule") << "Will mix " << object << "s with InputTag= " << tag.encode() << ", label will be "
                                  << label;
          //              std::cout <<"Will mix "<<object<<"s with InputTag= "<<tag.encode()<<", label will be "<<label<<std::endl;
        }
      } else {
        LogWarning("MixingModule")
            << "You have asked to mix an unknown type of object(" << object
            << ").\n If you want to include it in mixing, please contact the authors of the MixingModule!";
      }
      //} //if for mixProdStep2
    }  //while over the mixObjects parameters

    sort_all(wantedBranches_);
    for (unsigned int branch = 0; branch < wantedBranches_.size(); ++branch)
      LogDebug("MixingModule") << "Will keep branch " << wantedBranches_[branch] << " for mixing ";

    dropUnwantedBranches(wantedBranches_);

    produces<PileupMixingContent>();

    produces<CrossingFramePlaybackInfoNew>();

    edm::ConsumesCollector iC(consumesCollector());
    if (globalConf->configFromDB_) {
      configToken_ = esConsumes<edm::Transition::BeginLuminosityBlock>();
    }
    // Create and configure digitizers
    createDigiAccumulators(ps_mix, iC);
  }

  void MixingModule::createDigiAccumulators(const edm::ParameterSet& mixingPSet, edm::ConsumesCollector& iC) {
    ParameterSet const& digiPSet = mixingPSet.getParameterSet("digitizers");
    std::vector<std::string> digiNames = digiPSet.getParameterNames();
    for (auto const& digiName : digiNames) {
      ParameterSet const& pset = digiPSet.getParameterSet(digiName);
      if (pset.existsAs<edm::InputTag>("HepMCProductLabel")) {
        consumes<HepMCProduct>(pset.getParameter<edm::InputTag>("HepMCProductLabel"));
      }
      std::unique_ptr<DigiAccumulatorMixMod> accumulator = std::unique_ptr<DigiAccumulatorMixMod>(
          DigiAccumulatorMixModFactory::get()->makeDigiAccumulator(pset, producesCollector(), iC));
      // Create appropriate DigiAccumulator
      if (accumulator.get() != nullptr) {
        digiAccumulators_.push_back(accumulator.release());
      }
    }
  }

  void MixingModule::reload(const edm::EventSetup& setup) {
    //change the basic parameters.
    auto const& config = setup.getData(configToken_);
    minBunch_ = config.minBunch();
    maxBunch_ = config.maxBunch();
    bunchSpace_ = config.bunchSpace();
    //propagate to change the workers
    for (unsigned int ii = 0; ii < workersObjects_.size(); ++ii) {
      workersObjects_[ii]->reload(minBunch_, maxBunch_, bunchSpace_);
    }
  }

  void MixingModule::branchesActivate(const std::string& friendlyName,
                                      const std::string& subdet,
                                      InputTag& tag,
                                      std::string& label) {
    label = tag.label() + tag.instance();
    wantedBranches_.push_back(friendlyName + '_' + tag.label() + '_' + tag.instance());

    //if useCurrentProcessOnly, we have to change the input tag
    if (useCurrentProcessOnly_) {
      const std::string processName = edm::Service<edm::service::TriggerNamesService>()->getProcessName();
      tag = InputTag(tag.label(), tag.instance(), processName);
    }
  }

  void MixingModule::checkSignal(const edm::Event& e) {
    if (adjusters_.empty()) {
      for (auto const& adjuster : adjustersObjects_) {
        if (skipSignal_ or adjuster->checkSignal(e)) {
          adjusters_.push_back(adjuster);
        }
      }
    }
    if (workers_.empty()) {
      for (auto const& worker : workersObjects_) {
        if (skipSignal_ or worker->checkSignal(e)) {
          workers_.push_back(worker);
        }
      }
    }
  }

  void MixingModule::createnewEDProduct() {
    //create playback info
    playbackInfo_ = new CrossingFramePlaybackInfoNew(minBunch_, maxBunch_, maxNbSources_);
    //and CrossingFrames
    for (unsigned int ii = 0; ii < workers_.size(); ++ii) {
      workers_[ii]->createnewEDProduct();
    }
  }

  // Virtual destructor needed.
  MixingModule::~MixingModule() {
    for (auto& worker : workersObjects_) {
      delete worker;
    }

    for (auto& adjuster : adjustersObjects_) {
      delete adjuster;
    }

    for (auto& digiAccumulator : digiAccumulators_) {
      delete digiAccumulator;
    }
  }

  void MixingModule::addSignals(const edm::Event& e, const edm::EventSetup& setup) {
    if (skipSignal_) {
      return;
    }

    LogDebug("MixingModule") << "===============> adding signals for " << e.id();

    accumulateEvent(e, setup);
    // fill in signal part of CrossingFrame
    for (unsigned int ii = 0; ii < workers_.size(); ++ii) {
      workers_[ii]->addSignals(e);
    }
  }

  bool MixingModule::pileAllWorkers(EventPrincipal const& eventPrincipal,
                                    ModuleCallingContext const* mcc,
                                    int bunchCrossing,
                                    int eventId,
                                    int& vertexOffset,
                                    const edm::EventSetup& setup,
                                    StreamID const& streamID) {
    InternalContext internalContext(eventPrincipal.id(), mcc);
    ParentContext parentContext(&internalContext);
    ModuleCallingContext moduleCallingContext(&moduleDescription());
    ModuleContextSentry moduleContextSentry(&moduleCallingContext, parentContext);

    setupPileUpEvent(setup);

    for (auto const& adjuster : adjusters_) {
      adjuster->doOffset(bunchSpace_, bunchCrossing, eventPrincipal, &moduleCallingContext, eventId, vertexOffset);
    }
    PileUpEventPrincipal pep(eventPrincipal, &moduleCallingContext, bunchCrossing);

    accumulateEvent(pep, setup, streamID);

    for (auto const& worker : workers_) {
      LogDebug("MixingModule") << " merging Event:  id " << eventPrincipal.id();
      //      std::cout <<"PILEALLWORKERS merging Event:  id " << eventPrincipal.id() << std::endl;

      worker->addPileups(eventPrincipal, &moduleCallingContext, eventId);
    }

    return true;
  }

  void MixingModule::doPileUp(edm::Event& e, const edm::EventSetup& setup) {
    using namespace std::placeholders;

    // Don't allocate because PileUp will do it for us.
    std::vector<edm::SecondaryEventIDAndFileInfo> recordEventID;
    std::vector<size_t> sizes;
    sizes.reserve(maxNbSources_ * (maxBunch_ + 1 - minBunch_));
    size_t playbackCounter = 0U;
    edm::Handle<CrossingFramePlaybackInfoNew> playbackInfo_H;
    edm::Handle<CrossingFramePlaybackInfoExtended> oldFormatPlaybackInfo_H;
    bool oldFormatPlayback = false;
    if (playback_) {
      bool got = e.getByLabel(inputTagPlayback_, playbackInfo_H);
      if (!got) {
        bool gotOld = e.getByLabel(inputTagPlayback_, oldFormatPlaybackInfo_H);
        if (!gotOld) {
          throw cms::Exception("MixingProductNotFound")
              << " No "
                 "CrossingFramePlaybackInfoNew on the input file, but playback "
                 "option set!!!!! Please change the input file if you really want "
                 "playback!!!!!!"
              << std::endl;
        }
        oldFormatPlayback = true;
      }
    }

    // source[0] is "real" pileup.  Check to see that this is what we are doing.

    std::vector<int> PileupList;
    PileupList.clear();
    TrueNumInteractions_.clear();

    std::shared_ptr<PileUp> source0 = inputSources_[0];

    if ((source0 && source0->doPileUp(0)) && !playback_) {
      //    if((!inputSources_[0] || !inputSources_[0]->doPileUp()) && !playback_ )

      // Pre-calculate all pileup distributions before we go fishing for events

      source0->CalculatePileup(minBunch_, maxBunch_, PileupList, TrueNumInteractions_, e.streamID());
    }

    // pre-populate Pileup information
    // necessary for luminosity-dependent effects during hit accumulation

    std::vector<int> numInteractionList;
    std::vector<int> bunchCrossingList;
    std::vector<float> TrueInteractionList;
    std::vector<edm::EventID> eventInfoList;  // will be empty if we pre-populate, but it's not used in digitizers

    if (!playback_) {
      //Makin' a list: Basically, we don't care about the "other" sources at this point.
      for (int bunchCrossing = minBunch_; bunchCrossing <= maxBunch_; ++bunchCrossing) {
        bunchCrossingList.push_back(bunchCrossing);
        if (!inputSources_[0] || !inputSources_[0]->doPileUp(0)) {
          numInteractionList.push_back(0);
          TrueInteractionList.push_back(0);
        } else {
          numInteractionList.push_back(PileupList[bunchCrossing - minBunch_]);
          TrueInteractionList.push_back((TrueNumInteractions_)[bunchCrossing - minBunch_]);
        }
      }
    } else {  // have to read PU information from playback info
      for (int bunchIdx = minBunch_; bunchIdx <= maxBunch_; ++bunchIdx) {
        bunchCrossingList.push_back(bunchIdx);
        for (size_t readSrcIdx = 0; readSrcIdx < maxNbSources_; ++readSrcIdx) {
          if (oldFormatPlayback) {
            std::vector<edm::EventID> const& playEventID =
                oldFormatPlaybackInfo_H->getStartEventId(readSrcIdx, bunchIdx);
            size_t numberOfEvents = playEventID.size();
            if (readSrcIdx == 0) {
              PileupList.push_back(numberOfEvents);
              TrueNumInteractions_.push_back(numberOfEvents);
              numInteractionList.push_back(numberOfEvents);
              TrueInteractionList.push_back(numberOfEvents);
            }
          } else {
            size_t numberOfEvents = playbackInfo_H->getNumberOfEvents(bunchIdx, readSrcIdx);
            if (readSrcIdx == 0) {
              PileupList.push_back(numberOfEvents);
              TrueNumInteractions_.push_back(numberOfEvents);
              numInteractionList.push_back(numberOfEvents);
              TrueInteractionList.push_back(numberOfEvents);
            }
          }
        }
      }
    }

    for (Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end();
         accItr != accEnd;
         ++accItr) {
      (*accItr)->StorePileupInformation(
          bunchCrossingList, numInteractionList, TrueInteractionList, eventInfoList, bunchSpace_);
    }

    //    for (int bunchIdx = minBunch_; bunchIdx <= maxBunch_; ++bunchIdx) {
    //  std::cout << " bunch ID, Pileup, True " << bunchIdx << " " << PileupList[bunchIdx-minBunch_] << " " <<  TrueNumInteractions_[bunchIdx-minBunch_] << std::endl;
    //}

    for (int bunchIdx = minBunch_; bunchIdx <= maxBunch_; ++bunchIdx) {
      for (size_t setBcrIdx = 0; setBcrIdx < workers_.size(); ++setBcrIdx) {
        workers_[setBcrIdx]->setBcrOffset();
      }
      for (Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end();
           accItr != accEnd;
           ++accItr) {
        (*accItr)->initializeBunchCrossing(e, setup, bunchIdx);
      }

      for (size_t readSrcIdx = 0; readSrcIdx < maxNbSources_; ++readSrcIdx) {
        std::shared_ptr<PileUp> source = inputSources_[readSrcIdx];  // this looks like we create
                                                                     // new PileUp objects for each
                                                                     // source for each event?
                                                                     // Why?
        for (size_t setSrcIdx = 0; setSrcIdx < workers_.size(); ++setSrcIdx) {
          workers_[setSrcIdx]->setSourceOffset(readSrcIdx);
        }

        if (!source || !source->doPileUp(bunchIdx)) {
          sizes.push_back(0U);
          if (playback_ && !oldFormatPlayback) {
            playbackCounter += playbackInfo_H->getNumberOfEvents(bunchIdx, readSrcIdx);
          }
          continue;
        }

        //        int eventId = 0;
        int vertexOffset = 0;

        ModuleCallingContext const* mcc = e.moduleCallingContext();
        if (!playback_) {
          // non-minbias pileup only gets one event for now. Fix later if desired.
          int numberOfEvents = (readSrcIdx == 0 ? PileupList[bunchIdx - minBunch_] : 1);
          sizes.push_back(numberOfEvents);
          inputSources_[readSrcIdx]->readPileUp(e.id(),
                                                recordEventID,
                                                std::bind(&MixingModule::pileAllWorkers,
                                                          std::ref(*this),
                                                          _1,
                                                          mcc,
                                                          bunchIdx,
                                                          _2,
                                                          vertexOffset,
                                                          std::ref(setup),
                                                          e.streamID()),
                                                numberOfEvents,
                                                e.streamID());
        } else if (oldFormatPlayback) {
          std::vector<edm::EventID> const& playEventID = oldFormatPlaybackInfo_H->getStartEventId(readSrcIdx, bunchIdx);
          size_t numberOfEvents = playEventID.size();
          if (readSrcIdx == 0) {
            PileupList.push_back(numberOfEvents);
            TrueNumInteractions_.push_back(numberOfEvents);
          }
          sizes.push_back(numberOfEvents);
          std::vector<EventID>::const_iterator begin = playEventID.begin();
          std::vector<EventID>::const_iterator end = playEventID.end();
          inputSources_[readSrcIdx]->playOldFormatPileUp(begin,
                                                         end,
                                                         recordEventID,
                                                         std::bind(&MixingModule::pileAllWorkers,
                                                                   std::ref(*this),
                                                                   _1,
                                                                   mcc,
                                                                   bunchIdx,
                                                                   _2,
                                                                   vertexOffset,
                                                                   std::ref(setup),
                                                                   e.streamID()));
        } else {
          size_t numberOfEvents = playbackInfo_H->getNumberOfEvents(bunchIdx, readSrcIdx);
          if (readSrcIdx == 0) {
            PileupList.push_back(numberOfEvents);
            TrueNumInteractions_.push_back(numberOfEvents);
          }
          sizes.push_back(numberOfEvents);
          std::vector<SecondaryEventIDAndFileInfo>::const_iterator begin = playbackInfo_H->getEventId(playbackCounter);
          playbackCounter += numberOfEvents;
          std::vector<SecondaryEventIDAndFileInfo>::const_iterator end = playbackInfo_H->getEventId(playbackCounter);
          inputSources_[readSrcIdx]->playPileUp(begin,
                                                end,
                                                recordEventID,
                                                std::bind(&MixingModule::pileAllWorkers,
                                                          std::ref(*this),
                                                          _1,
                                                          mcc,
                                                          bunchIdx,
                                                          _2,
                                                          vertexOffset,
                                                          std::ref(setup),
                                                          e.streamID()));
        }
      }
      for (Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end();
           accItr != accEnd;
           ++accItr) {
        (*accItr)->finalizeBunchCrossing(e, setup, bunchIdx);
      }
    }

    // Save playback information
    for (auto const item : recordEventID) {
      eventInfoList.emplace_back(item.eventID());
    }

    // setInfo swaps recordEventID, so recordEventID is useless (empty) after the call.
    playbackInfo_->setInfo(recordEventID, sizes);

    // Keep track of pileup accounting...

    std::unique_ptr<PileupMixingContent> PileupMixing_;

    PileupMixing_ = std::make_unique<PileupMixingContent>(
        bunchCrossingList, numInteractionList, TrueInteractionList, eventInfoList, bunchSpace_);

    e.put(std::move(PileupMixing_));

    // we have to do the ToF transformation for PSimHits once all pileup has been added
    for (unsigned int ii = 0; ii < workers_.size(); ++ii) {
      workers_[ii]->setTof();
      workers_[ii]->put(e);
    }
  }

  void MixingModule::put(edm::Event& e, const edm::EventSetup& setup) {
    if (playbackInfo_) {
      std::unique_ptr<CrossingFramePlaybackInfoNew> pOut(playbackInfo_);
      e.put(std::move(pOut));
    }
  }

  void MixingModule::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
    for (Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end();
         accItr != accEnd;
         ++accItr) {
      (*accItr)->beginRun(run, setup);
    }
    BMixingModule::beginRun(run, setup);
  }

  void MixingModule::endRun(edm::Run const& run, edm::EventSetup const& setup) {
    for (Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end();
         accItr != accEnd;
         ++accItr) {
      (*accItr)->endRun(run, setup);
    }
    BMixingModule::endRun(run, setup);
  }

  void MixingModule::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
    for (Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end();
         accItr != accEnd;
         ++accItr) {
      (*accItr)->beginLuminosityBlock(lumi, setup);
    }
    BMixingModule::beginLuminosityBlock(lumi, setup);
  }

  void MixingModule::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
    for (Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end();
         accItr != accEnd;
         ++accItr) {
      (*accItr)->endLuminosityBlock(lumi, setup);
    }
    BMixingModule::endLuminosityBlock(lumi, setup);
  }

  void MixingModule::initializeEvent(edm::Event const& event, edm::EventSetup const& setup) {
    for (Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end();
         accItr != accEnd;
         ++accItr) {
      (*accItr)->initializeEvent(event, setup);
    }
  }

  void MixingModule::accumulateEvent(edm::Event const& event, edm::EventSetup const& setup) {
    for (Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end();
         accItr != accEnd;
         ++accItr) {
      (*accItr)->accumulate(event, setup);
    }
  }

  void MixingModule::accumulateEvent(PileUpEventPrincipal const& event,
                                     edm::EventSetup const& setup,
                                     edm::StreamID const& streamID) {
    for (Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end();
         accItr != accEnd;
         ++accItr) {
      (*accItr)->accumulate(event, setup, streamID);
    }
  }

  void MixingModule::finalizeEvent(edm::Event& event, edm::EventSetup const& setup) {
    for (Accumulators::const_iterator accItr = digiAccumulators_.begin(), accEnd = digiAccumulators_.end();
         accItr != accEnd;
         ++accItr) {
      (*accItr)->finalizeEvent(event, setup);
    }
  }
}  // namespace edm
