#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimTracker/TrackTriggerAssociation/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <vector>
#include <map>
#include <utility>
#include <set>
#include <algorithm>
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
    ~StubAssociator() override {}

  private:
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void produce(edm::Event&, const edm::EventSetup&) override;
    // helper classe to store configurations
    const Setup* setup_;
    // ED input token of TTStubs
    edm::EDGetTokenT<TTStubDetSetVec> getTokenTTStubDetSetVec_;
    // ED input token of TTClusterAssociation
    edm::EDGetTokenT<TTClusterAssMap> getTokenTTClusterAssMap_;
    // ED output token for recosntructable stub association
    edm::EDPutTokenT<StubAssociation> putTokenReconstructable_;
    // ED output token for selected stub association
    edm::EDPutTokenT<StubAssociation> putTokenSelection_;
    // Setup token
    edm::ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    //
    StubAssociation::Config iConfig_;
    // required number of associated stub layers to a TP to consider it reconstruct-able
    int minLayers_;
    // required number of associated ps stub layers to a TP to consider it reconstruct-able
    int minLayersPS_;
    // pt cut in GeV
    double minPt_;
    // max eta for TP with z0 = 0
    double maxEta0_;
    // half lumi region size in cm
    double maxZ0_;
    // cut on impact parameter in cm
    double maxD0_;
    // cut on vertex pos r in cm
    double maxVertR_;
    // cut on vertex pos z in cm
    double maxVertZ_;
    // cut on TP zT
    double maxZT_;
    // selector to partly select TPs for efficiency measurements
    TrackingParticleSelector tpSelector_;
  };

  StubAssociator::StubAssociator(const edm::ParameterSet& iConfig)
      : minLayers_(iConfig.getParameter<int>("MinLayers")),
        minLayersPS_(iConfig.getParameter<int>("MinLayersPS")),
        minPt_(iConfig.getParameter<double>("MinPt")),
        maxEta0_(iConfig.getParameter<double>("MaxEta0")),
        maxZ0_(iConfig.getParameter<double>("MaxZ0")),
        maxD0_(iConfig.getParameter<double>("MaxD0")),
        maxVertR_(iConfig.getParameter<double>("MaxVertR")),
        maxVertZ_(iConfig.getParameter<double>("MaxVertZ")) {
    iConfig_.minLayersGood_ = iConfig.getParameter<int>("MinLayersGood");
    iConfig_.minLayersGoodPS_ = iConfig.getParameter<int>("MinLayersGoodPS");
    iConfig_.maxLayersBad_ = iConfig.getParameter<int>("MaxLayersBad");
    iConfig_.maxLayersBadPS_ = iConfig.getParameter<int>("MaxLayersBadPS");
    // book in- and output ed products
    getTokenTTStubDetSetVec_ =
        consumes<TTStubDetSetVec>(iConfig.getParameter<edm::InputTag>("InputTagTTStubDetSetVec"));
    getTokenTTClusterAssMap_ =
        consumes<TTClusterAssMap>(iConfig.getParameter<edm::InputTag>("InputTagTTClusterAssMap"));
    putTokenReconstructable_ = produces<StubAssociation>(iConfig.getParameter<std::string>("BranchReconstructable"));
    putTokenSelection_ = produces<StubAssociation>(iConfig.getParameter<std::string>("BranchSelection"));
    // book ES product
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd, edm::Transition::BeginRun>();
  }

  void StubAssociator::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
    setup_ = &iSetup.getData(esGetTokenSetup_);
    maxZT_ = std::sinh(maxEta0_) * setup_->chosenRofZ();
    // configure TrackingParticleSelector
    constexpr double ptMax = 9.e9;
    constexpr int minHit = 0;
    constexpr bool signalOnly = true;
    constexpr bool intimeOnly = true;
    constexpr bool chargedOnly = true;
    constexpr bool stableOnly = false;
    const double maxEta = std::asinh((maxZT_ + maxZ0_) / setup_->chosenRofZ());
    tpSelector_ = TrackingParticleSelector(
        minPt_, ptMax, -maxEta, maxEta, maxVertR_, maxVertZ_, minHit, signalOnly, intimeOnly, chargedOnly, stableOnly);
  }

  void StubAssociator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // associate TTStubs with TrackingParticles
    edm::Handle<TTStubDetSetVec> handleTTStubDetSetVec;
    iEvent.getByToken<TTStubDetSetVec>(getTokenTTStubDetSetVec_, handleTTStubDetSetVec);
    edm::Handle<TTClusterAssMap> handleTTClusterAssMap;
    iEvent.getByToken<TTClusterAssMap>(getTokenTTClusterAssMap_, handleTTClusterAssMap);
    std::map<TPPtr, std::vector<TTStubRef>> mapTPPtrsTTStubRefs;
    auto isNonnull = [](const TPPtr& tpPtr) { return tpPtr.isNonnull(); };
    for (TTStubDetSetVec::const_iterator ttModule = handleTTStubDetSetVec->begin();
         ttModule != handleTTStubDetSetVec->end();
         ttModule++) {
      for (TTStubDetSet::const_iterator ttStub = ttModule->begin(); ttStub != ttModule->end(); ttStub++) {
        const TTStubRef ttStubRef = makeRefTo(handleTTStubDetSetVec, ttStub);
        std::set<TPPtr> tpPtrs;
        for (unsigned int iClus = 0; iClus < 2; iClus++) {
          const std::vector<TPPtr>& assocPtrs =
              handleTTClusterAssMap->findTrackingParticlePtrs(ttStubRef->clusterRef(iClus));
          std::copy_if(assocPtrs.begin(), assocPtrs.end(), std::inserter(tpPtrs, tpPtrs.begin()), isNonnull);
        }
        for (const TPPtr& tpPtr : tpPtrs)
          mapTPPtrsTTStubRefs[tpPtr].push_back(ttStubRef);
      }
    }
    // associate reconstructable TrackingParticles with TTStubs
    StubAssociation reconstructable(iConfig_, setup_);
    StubAssociation selection(iConfig_, setup_);
    for (const auto& p : mapTPPtrsTTStubRefs) {
      // require min layers
      std::set<int> hitPattern, hitPatternPS;
      for (const TTStubRef& ttStubRef : p.second) {
        const int layerId = setup_->layerId(ttStubRef);
        hitPattern.insert(layerId);
        if (setup_->psModule(ttStubRef))
          hitPatternPS.insert(layerId);
      }
      if (static_cast<int>(hitPattern.size()) < minLayers_ || static_cast<int>(hitPatternPS.size()) < minLayersPS_)
        continue;
      reconstructable.insert(p.first, p.second);
      // require parameter space
      const double zT = p.first->z0() + p.first->tanl() * setup_->chosenRofZ();
      if ((std::abs(p.first->d0()) > maxD0_) || (std::abs(p.first->z0()) > maxZ0_) || (std::abs(zT) > maxZT_))
        continue;
      // require signal only and min pt
      if (tpSelector_(*p.first))
        selection.insert(p.first, p.second);
    }
    iEvent.emplace(putTokenReconstructable_, std::move(reconstructable));
    iEvent.emplace(putTokenSelection_, std::move(selection));
  }

}  // namespace tt

DEFINE_FWK_MODULE(tt::StubAssociator);
