#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimTracker/TrackTriggerAssociation/interface/TTTypes.h"
#include "SimTracker/TrackTriggerAssociation/interface/StubAssociation.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iterator>
#include <cmath>

using namespace std;
using namespace edm;

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
  class StubAssociator : public global::EDProducer<> {
  public:
    explicit StubAssociator(const ParameterSet&);

  private:
    void produce(StreamID, Event&, const EventSetup&) const override;
    // ED input token of TTStubs
    EDGetTokenT<TTStubDetSetVec> getTokenTTStubDetSetVec_;
    // ED input token of TTClusterAssociation
    EDGetTokenT<TTClusterAssMap> getTokenTTClusterAssMap_;
    // ED output token for recosntructable stub association
    EDPutTokenT<StubAssociation> putTokenReconstructable_;
    // ED output token for selected stub association
    EDPutTokenT<StubAssociation> putTokenSelection_;
    // Setup token
    ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
  };

  StubAssociator::StubAssociator(const ParameterSet& iConfig) {
    // book in- and output ed products
    getTokenTTStubDetSetVec_ = consumes<TTStubDetSetVec>(iConfig.getParameter<InputTag>("InputTagTTStubDetSetVec"));
    getTokenTTClusterAssMap_ = consumes<TTClusterAssMap>(iConfig.getParameter<InputTag>("InputTagTTClusterAssMap"));
    putTokenReconstructable_ = produces<StubAssociation>(iConfig.getParameter<string>("BranchReconstructable"));
    putTokenSelection_ = produces<StubAssociation>(iConfig.getParameter<string>("BranchSelection"));
    // book ES product
    esGetTokenSetup_ = esConsumes<Setup, SetupRcd>();
  }

  void StubAssociator::produce(StreamID, Event& iEvent, const EventSetup& iSetup) const {
    auto const& setup = iSetup.getData(esGetTokenSetup_);

    // associate TTStubs with TrackingParticles
    Handle<TTStubDetSetVec> handleTTStubDetSetVec = iEvent.getHandle(getTokenTTStubDetSetVec_);
    auto const& ttClusterAssMap = iEvent.get(getTokenTTClusterAssMap_);

    map<TPPtr, vector<TTStubRef>> mapTPPtrsTTStubRefs;
    auto isNonnull = [](const TPPtr& tpPtr) { return tpPtr.isNonnull(); };
    for (TTStubDetSetVec::const_iterator ttModule = handleTTStubDetSetVec->begin();
         ttModule != handleTTStubDetSetVec->end();
         ttModule++) {
      for (TTStubDetSet::const_iterator ttStub = ttModule->begin(); ttStub != ttModule->end(); ttStub++) {
        const TTStubRef ttStubRef = makeRefTo(handleTTStubDetSetVec, ttStub);
        set<TPPtr> tpPtrs;
        for (unsigned int iClus = 0; iClus < 2; iClus++) {
          const vector<TPPtr>& assocPtrs = ttClusterAssMap.findTrackingParticlePtrs(ttStubRef->clusterRef(iClus));
          copy_if(assocPtrs.begin(), assocPtrs.end(), inserter(tpPtrs, tpPtrs.begin()), isNonnull);
        }
        for (const TPPtr& tpPtr : tpPtrs)
          mapTPPtrsTTStubRefs[tpPtr].push_back(ttStubRef);
      }
    }
    // associate reconstructable TrackingParticles with TTStubs
    StubAssociation reconstructable(&setup);
    StubAssociation selection(&setup);
    for (const auto& p : mapTPPtrsTTStubRefs) {
      if (!setup.useForReconstructable(*p.first) || !setup.reconstructable(p.second))
        continue;
      reconstructable.insert(p.first, p.second);
      if (setup.useForAlgEff(*p.first))
        selection.insert(p.first, p.second);
    }
    iEvent.emplace(putTokenReconstructable_, std::move(reconstructable));
    iEvent.emplace(putTokenSelection_, std::move(selection));
  }

}  // namespace tt

DEFINE_FWK_MODULE(tt::StubAssociator);
