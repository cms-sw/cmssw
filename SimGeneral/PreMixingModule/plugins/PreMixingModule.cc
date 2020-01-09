/** \class PreMixingModule
 *
 * PreMixingModule is the EDProducer subclass that overlays premixed
 * MC events on top of MC. It is similar to DataMixingModule, but
 * tailored for premixing use case.
 *
 ************************************************************/
#include "Mixing/Base/interface/BMixingModule.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/InternalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoNew.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"
#include "PreMixingPileupCopy.h"

#include <CLHEP/Random/RandomEngine.h>

#include <functional>
#include <vector>

namespace edm {
  class PreMixingModule : public BMixingModule {
  public:
    PreMixingModule(const edm::ParameterSet& ps, MixingCache::Config const* globalConf);

    ~PreMixingModule() override = default;

    void checkSignal(const edm::Event& e) override{};
    void createnewEDProduct() override {}
    void addSignals(const edm::Event& e, const edm::EventSetup& ES) override;
    void doPileUp(edm::Event& e, const edm::EventSetup& ES) override;
    void put(edm::Event& e, const edm::EventSetup& ES) override;

    void initializeEvent(edm::Event const& e, edm::EventSetup const& eventSetup) override;
    void beginRun(edm::Run const& run, edm::EventSetup const& eventSetup) override;
    void beginLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) override;
    void endLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) override;
    void endRun(const edm::Run& r, const edm::EventSetup& setup) override;

  private:
    class AdjustPileupDistribution {
    public:
      AdjustPileupDistribution(const edm::ParameterSet& ps)
          : firstRun_(ps.getParameter<unsigned int>("firstRun")),
            firstBinPileup_(ps.getParameter<unsigned int>("firstBinPileup")),
            pileupProbabilities_(ps.getParameter<std::vector<double>>("pileupProbabilities")) {
        for (double p : pileupProbabilities_) {
          if (p < 0. or p > 1.) {
            throw cms::Exception("Configuration") << "Invalid probability value " << p << " for firstRun " << firstRun_
                                                  << ". The probability must be >= 0. and <= 1.";
          }
        }
      }

      edm::RunNumber_t firstRun() const { return firstRun_; }
      double probability(float pileup) const {
        unsigned int bin = static_cast<unsigned int>(pileup);
        if (bin < firstBinPileup_ or bin >= firstBinPileup_ + pileupProbabilities_.size()) {
          edm::LogWarning("PreMixingModule")
              << "Got pileup event with true pileup " << pileup
              << " that is outside of the configured pileup adjustment bounds [" << firstBinPileup_ << ", "
              << firstBinPileup_ + pileupProbabilities_.size() - 1 << "]. Using probability 0.";
          return 0.;
        }
        return pileupProbabilities_[bin - firstBinPileup_];
      }

    private:
      edm::RunNumber_t firstRun_;
      unsigned int firstBinPileup_;
      std::vector<double> pileupProbabilities_;
    };

    bool pileWorker(const edm::EventPrincipal&,
                    int bcr,
                    int EventId,
                    const edm::EventSetup& ES,
                    ModuleCallingContext const*,
                    AdjustPileupDistribution const* pileupAdjuster);

    PreMixingPileupCopy puWorker_;
    bool addedPileup_ = false;

    std::vector<AdjustPileupDistribution> pileupAdjusters_;
    std::vector<std::unique_ptr<PreMixingWorker>> workers_;
  };

  PreMixingModule::PreMixingModule(const edm::ParameterSet& ps, MixingCache::Config const* globalConf)
      : BMixingModule(ps, globalConf),
        puWorker_(ps.getParameter<edm::ParameterSet>("workers").getParameter<edm::ParameterSet>("pileup"),
                  producesCollector(),
                  consumesCollector()),
        pileupAdjusters_(
            edm::vector_transform(ps.getParameter<std::vector<edm::ParameterSet>>("adjustPileupDistribution"),
                                  [](const auto& ps) { return AdjustPileupDistribution(ps); })) {
    std::sort(pileupAdjusters_.begin(), pileupAdjusters_.end(), [](const auto& a, const auto& b) {
      return a.firstRun() < b.firstRun();
    });

    const auto& workers = ps.getParameter<edm::ParameterSet>("workers");
    std::vector<std::string> names = workers.getParameterNames();

    // Hack to keep the random number sequence unchanged for migration
    // from DataMixingModule to PreMixingModule. To be removed in a
    // subsequent PR doing only that.
    {
      std::vector<std::string> tmp;
      auto hack = [&](const std::string& name) {
        auto i = std::find(names.begin(), names.end(), name);
        if (i != names.end()) {
          tmp.push_back(*i);
          names.erase(i);
        }
      };
      hack("ecal");
      hack("hcal");
      hack("strip");
      hack("pixel");
      std::copy(names.begin(), names.end(), std::back_inserter(tmp));
      names = std::move(tmp);
    }

    for (const auto& name : names) {
      if (name == "pileup") {
        continue;
      }
      const auto& pset = workers.getParameter<edm::ParameterSet>(name);
      std::string type = pset.getParameter<std::string>("workerType");
      workers_.emplace_back(
          PreMixingWorkerFactory::get()->create(type, pset, producesCollector(), consumesCollector()));
    }
  }

  void PreMixingModule::initializeEvent(const edm::Event& e, const edm::EventSetup& ES) {
    for (auto& w : workers_) {
      w->initializeEvent(e, ES);
    }
  }

  void PreMixingModule::beginRun(edm::Run const& run, const edm::EventSetup& ES) {
    BMixingModule::beginRun(run, ES);
    for (auto& w : workers_) {
      w->beginRun(run, ES);
    }
  }

  void PreMixingModule::endRun(edm::Run const& run, const edm::EventSetup& ES) {
    for (auto& w : workers_) {
      w->endRun();
    }
    BMixingModule::endRun(run, ES);
  }

  void PreMixingModule::addSignals(const edm::Event& e, const edm::EventSetup& ES) {
    // fill in maps of hits

    LogDebug("PreMixingModule") << "===============> adding MC signals for " << e.id();

    for (auto& w : workers_) {
      w->addSignals(e, ES);
    }

    addedPileup_ = false;
  }

  bool PreMixingModule::pileWorker(const EventPrincipal& ep,
                                   int bcr,
                                   int eventNr,
                                   const edm::EventSetup& ES,
                                   edm::ModuleCallingContext const* mcc,
                                   AdjustPileupDistribution const* pileupAdjuster) {
    InternalContext internalContext(ep.id(), mcc);
    ParentContext parentContext(&internalContext);
    ModuleCallingContext moduleCallingContext(&moduleDescription());
    ModuleContextSentry moduleContextSentry(&moduleCallingContext, parentContext);

    PileUpEventPrincipal pep(ep, &moduleCallingContext, bcr);

    if (pileupAdjuster) {
      float trueNumInteractions = puWorker_.getTrueNumInteractions(pep);
      double prob = pileupAdjuster->probability(static_cast<unsigned int>(trueNumInteractions));
      edm::Service<edm::RandomNumberGenerator> rng;
      CLHEP::HepRandomEngine& engine = rng->getEngine(ep.streamID());
      if (engine.flat() > prob) {
        // engine.flat() should give a double in ]0,1[ range
        // the choice above means that "prob = 1-ulp" is treatead as 1
        return false;
      }
    }

    LogDebug("PreMixingModule") << "\n===============> adding pileups from event  " << ep.id() << " for bunchcrossing "
                                << bcr;

    // Note:  setupPileUpEvent may modify the run and lumi numbers of the EventPrincipal to match that of the primary event.
    setupPileUpEvent(ES);

    // check and see if we need to copy the pileup information from
    // secondary stream to the output stream
    // We only have the pileup event here, so pick the first time and store the info
    if (!addedPileup_) {
      puWorker_.addPileupInfo(pep);
      addedPileup_ = true;
    }

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    for (auto& w : workers_) {
      w->addPileups(pep, ES);
    }

    return true;
  }

  void PreMixingModule::doPileUp(edm::Event& e, const edm::EventSetup& ES) {
    using namespace std::placeholders;

    std::vector<edm::SecondaryEventIDAndFileInfo> recordEventID;
    std::vector<int> PileupList;
    TrueNumInteractions_.clear();

    ModuleCallingContext const* mcc = e.moduleCallingContext();

    AdjustPileupDistribution const* pileupAdjuster = nullptr;
    if (not pileupAdjusters_.empty()) {
      // Find the adjustment settings for the run of the signal event
      // the container should be small-enough to not really gain
      // anything with binary search
      auto it = std::find_if(pileupAdjusters_.rbegin(),
                             pileupAdjusters_.rend(),
                             [iRun = e.id().run()](const auto& elem) { return elem.firstRun() <= iRun; });
      if (it == pileupAdjusters_.rend()) {
        throw cms::Exception("LogicError") << "Encountered run " << e.id().run()
                                           << ", but the first run available in the pileup adjustment configuration is "
                                           << pileupAdjusters_.front().firstRun() << ". Please fix the configuration.";
      }
      pileupAdjuster = &*it;
    }

    for (int bunchCrossing = minBunch_; bunchCrossing <= maxBunch_; ++bunchCrossing) {
      for (unsigned int isource = 0; isource < maxNbSources_; ++isource) {
        std::shared_ptr<PileUp> source = inputSources_[isource];
        if (!source || !(source->doPileUp(bunchCrossing)))
          continue;

        if (isource == 0)
          source->CalculatePileup(minBunch_, maxBunch_, PileupList, TrueNumInteractions_, e.streamID());

        int NumPU_Events = 0;
        if (isource == 0) {
          NumPU_Events = PileupList[bunchCrossing - minBunch_];
        } else {
          // non-minbias pileup only gets one event for now. Fix later if desired.
          NumPU_Events = 1;
        }

        for (auto& w : workers_) {
          w->initializeBunchCrossing(e, ES, bunchCrossing);
        }

        source->readPileUp(
            e.id(),
            recordEventID,
            std::bind(
                &PreMixingModule::pileWorker, std::ref(*this), _1, bunchCrossing, _2, std::cref(ES), mcc, pileupAdjuster),
            NumPU_Events,
            e.streamID());

        for (auto& w : workers_) {
          w->finalizeBunchCrossing(e, ES, bunchCrossing);
        }
      }
    }
  }

  void PreMixingModule::put(edm::Event& e, const edm::EventSetup& ES) {
    // individual workers...
    // move pileup first so we have access to the information for the put step
    const auto& ps = puWorker_.getPileupSummaryInfo();
    int bunchSpacing = puWorker_.getBunchSpacing();

    for (auto& w : workers_) {
      w->put(e, ES, ps, bunchSpacing);
    }

    puWorker_.putPileupInfo(e);
  }

  void PreMixingModule::beginLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) {
    BMixingModule::beginLuminosityBlock(l1, c);
    for (auto& w : workers_) {
      w->beginLuminosityBlock(l1, c);
    }
  }

  void PreMixingModule::endLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) {
    BMixingModule::endLuminosityBlock(l1, c);
  }
}  // namespace edm

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Framework/interface/MakerMacros.h"
using edm::PreMixingModule;
DEFINE_FWK_MODULE(PreMixingModule);
