#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/Associations/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Associator.h"
#include "SimDataFormats/Associations/interface/StubAssociation.h"

#include <vector>
#include <deque>
#include <map>
#include <utility>
#include <set>
#include <iterator>

namespace tt {

  /*! \class  tt::StubAssociator
   *  \brief  Class to associate reconstrucable TrackingParticles with TTStubs and vice versa
   *          It may associate multiple TPs with a TTStub and can therefore be used to associate
   *          TTTracks with TrackingParticles. This EDProducer creates two StubAssociation EDProducts,
   *          one using only TP that are "reconstructable" (produce stubs in a min. number of layers)
   *          and one using TP that are also "use for the tracking efficiency measurement".
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class StubAssociator : public edm::stream::EDProducer<> {
  public:
    explicit StubAssociator(const edm::ParameterSet&);
    ~StubAssociator() override = default;

  private:
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void produce(edm::Event&, const edm::EventSetup&) override;
    // ED input token of TTStubs
    edm::EDGetTokenT<TTStubDetSetVec> getTokenTTStubDetSetVec_;
    // ED input token of TTClusterAssociation
    edm::EDGetTokenT<TTClusterAssMap> getTokenTTClusterAssMap_;
    // ED output token for stub association for fake rate
    edm::EDPutTokenT<StubAssociation> putTokenFake_;
    // ED output token for stub association duplicate rate
    edm::EDPutTokenT<StubAssociation> putTokenDup_;
    // ED output token for stub association for tracking efficiency
    edm::EDPutTokenT<StubAssociation> putTokenEff_;
    // Setup token
    edm::ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // Associator token
    edm::ESGetToken<Associator, SetupRcd> esGetTokenAssociator_;
    // pt cut in GeV
    double minPt_;
    // half lumi region size in cm
    double maxZ0_;
    // cut on impact parameter in cm
    double maxD0_;
    // cut on vertex pos r in cm
    double maxVertR_;
    // cut on vertex pos z in cm
    double maxVertZ_;
    //
    bool looseMatching_;
    // selector to partly select TPs for efficiency measurements
    TrackingParticleSelector tpSelector_;
  };

  StubAssociator::StubAssociator(const edm::ParameterSet& iConfig)
      : minPt_(iConfig.getParameter<double>("MinPt")),
        maxZ0_(iConfig.getParameter<double>("MaxZ0")),
        maxD0_(iConfig.getParameter<double>("MaxD0")),
        maxVertR_(iConfig.getParameter<double>("MaxVertR")),
        maxVertZ_(iConfig.getParameter<double>("MaxVertZ")),
        looseMatching_(iConfig.getParameter<bool>("LooseMatching")) {
    // book in- and output ed products
    const auto& ttStubDetSetVec = iConfig.getParameter<edm::InputTag>("InputTagTTStubDetSetVec");
    const auto& ttClusterAssMap = iConfig.getParameter<edm::InputTag>("InputTagTTClusterAssMap");
    const auto& branchFake = iConfig.getParameter<std::string>("BranchFake");
    const auto& branchDup = iConfig.getParameter<std::string>("BranchDup");
    const auto& branchEff = iConfig.getParameter<std::string>("BranchEff");
    getTokenTTStubDetSetVec_ = consumes(ttStubDetSetVec);
    getTokenTTClusterAssMap_ = consumes(ttClusterAssMap);
    putTokenFake_ = produces(branchFake);
    putTokenDup_ = produces(branchDup);
    putTokenEff_ = produces(branchEff);
    // book ES product
    esGetTokenAssociator_ = esConsumes();
  }

  void StubAssociator::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
    // configure TrackingParticleSelector
    constexpr double ptMax = 9.e9;
    constexpr double maxEta_ = 9.e9;
    constexpr int minHit = 0;
    constexpr bool signalOnly = true;
    constexpr bool intimeOnly = true;
    constexpr bool chargedOnly = true;
    constexpr bool stableOnly = false;
    tpSelector_ = TrackingParticleSelector(
        minPt_, ptMax, -maxEta_, maxEta_, maxVertR_, maxVertZ_, minHit, signalOnly, intimeOnly, chargedOnly, stableOnly);
  }

  void StubAssociator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // helper class to associate TTStubs and TrackingParticle
    const Associator* associator = &iSetup.getData(esGetTokenAssociator_);
    // associate TTStubs with TrackingParticles
    edm::Handle<TTStubDetSetVec> handle;
    iEvent.getByToken<TTStubDetSetVec>(getTokenTTStubDetSetVec_, handle);
    const TTClusterAssMap& ttClusterAssMap = iEvent.get(getTokenTTClusterAssMap_);
    std::map<TPPtr, std::set<TTStubRef>> mapTPPtrsTTStubRefs;
    for (TTStubDetSetVec::const_iterator ttModule = handle->begin(); ttModule != handle->end(); ttModule++) {
      for (TTStubDetSet::const_iterator ttStub = ttModule->begin(); ttStub != ttModule->end(); ttStub++) {
        const TTStubRef ttStubRef = makeRefTo(handle, ttStub);
        std::set<TPPtr> tpPtrs;
        for (unsigned int iClus = 0; iClus < 2; iClus++)
          for (const TPPtr& tpPtr : ttClusterAssMap.findTrackingParticlePtrs(ttStubRef->clusterRef(iClus)))
            if (tpPtr.isNonnull())
              tpPtrs.insert(tpPtr);
        for (const TPPtr& tpPtr : tpPtrs)
          mapTPPtrsTTStubRefs[tpPtr].insert(ttStubRef);
      }
    }
    // associate TTStubs with primary TrackingParticles
    std::map<TPPtr, std::set<TTStubRef>> mapPrimaryTPPtrsTTStubRefs;
    if (looseMatching_) {
      for (auto& p : mapTPPtrsTTStubRefs) {
        const TPPtr primary = associator->getPrimaryTP(p.first);
        std::set<TTStubRef>& ttStubRefs = mapPrimaryTPPtrsTTStubRefs[primary];
        ttStubRefs.insert(p.second.begin(), p.second.end());
      }
    }
    // associate loosly reconstructable TrackingParticles with TTStubs
    StubAssociation forFake;
    if (looseMatching_) {
      for (const auto& p : mapPrimaryTPPtrsTTStubRefs) {
        // require min layers
        const std::vector<TTStubRef> ttStubRefs(p.second.begin(), p.second.end());
        if (associator->reconstructable(ttStubRefs))
          forFake.insert(p.first, ttStubRefs);
      }
    }
    // associate appreciated TPs with TTStubs
    StubAssociation forDup;
    StubAssociation forEff;
    for (auto& p : mapTPPtrsTTStubRefs) {
      // require min layers
      const std::vector<TTStubRef> ttStubRefs(p.second.begin(), p.second.end());
      if (!associator->reconstructable(ttStubRefs))
        continue;
      if (!looseMatching_)
        forFake.insert(p.first, ttStubRefs);
      forDup.insert(p.first, ttStubRefs);
      // require parameter space and signal only
      if (!tpSelector_(*p.first))
        continue;
      // require additional parameter space
      if ((std::abs(p.first->d0()) > maxD0_) || (std::abs(p.first->z0()) > maxZ0_))
        continue;
      // fill selected TP
      forEff.insert(p.first, ttStubRefs);
    }
    // store StubAssociations
    iEvent.emplace(putTokenFake_, std::move(forFake));
    iEvent.emplace(putTokenDup_, std::move(forDup));
    iEvent.emplace(putTokenEff_, std::move(forEff));
  }

}  // namespace tt

DEFINE_FWK_MODULE(tt::StubAssociator);
