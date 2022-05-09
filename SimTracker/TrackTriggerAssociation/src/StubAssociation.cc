#include "SimTracker/TrackTriggerAssociation/interface/StubAssociation.h"

#include <map>
#include <vector>
#include <utility>
#include <numeric>

using namespace std;

namespace tt {

  // insert a TPPtr and its associated collection of TTstubRefs into the underlayering maps
  void StubAssociation::insert(const TPPtr& tpPtr, const vector<TTStubRef>& ttSTubRefs) {
    mapTPPtrsTTStubRefs_.insert({tpPtr, ttSTubRefs});
    for (const TTStubRef& ttSTubRef : ttSTubRefs)
      mapTTStubRefsTPPtrs_[ttSTubRef].push_back(tpPtr);
  }

  // returns collection of TPPtrs associated to given TTstubRef
  vector<TPPtr> StubAssociation::findTrackingParticlePtrs(const TTStubRef& ttStubRef) const {
    const auto it = mapTTStubRefsTPPtrs_.find(ttStubRef);
    const vector<TPPtr> res = it != mapTTStubRefsTPPtrs_.end() ? it->second : emptyTPPtrs_;
    return res;
  }

  // returns collection of TTStubRefs associated to given TPPtr
  vector<TTStubRef> StubAssociation::findTTStubRefs(const TPPtr& tpPtr) const {
    const auto it = mapTPPtrsTTStubRefs_.find(tpPtr);
    return it != mapTPPtrsTTStubRefs_.end() ? it->second : emptyTTStubRefs_;
  }

  // Get all TPs that are matched to these stubs in at least 'tpMinLayers' layers and 'tpMinLayersPS' ps layers
  vector<TPPtr> StubAssociation::associate(const vector<TTStubRef>& ttStubRefs) const {
    // count associated layer for each TP
    map<TPPtr, set<int>> m;
    map<TPPtr, set<int>> mPS;
    for (const TTStubRef& ttStubRef : ttStubRefs) {
      for (const TPPtr& tpPtr : findTrackingParticlePtrs(ttStubRef)) {
        const int layerId = setup_->layerId(ttStubRef);
        m[tpPtr].insert(layerId);
        if (setup_->psModule(ttStubRef))
          mPS[tpPtr].insert(layerId);
      }
    }
    // count matched TPs
    auto sum = [this](int& sum, const pair<TPPtr, set<int>>& p) {
      return sum += ((int)p.second.size() < setup_->tpMinLayers() ? 0 : 1);
    };
    const int nTPs = accumulate(m.begin(), m.end(), 0, sum);
    vector<TPPtr> tpPtrs;
    tpPtrs.reserve(nTPs);
    // fill and return matched TPs
    for (const auto& p : m)
      if ((int)p.second.size() >= setup_->tpMinLayers() && (int)mPS[p.first].size() >= setup_->tpMinLayersPS())
        tpPtrs.push_back(p.first);
    return tpPtrs;
  }

  // Get all TPs that are matched to these stubs in at least 'tpMinLayers' layers and 'tpMinLayersPS' ps layers with not more then 'tpMaxBadStubs2S' not associated 2S stubs and not more then 'tpMaxBadStubsPS' associated PS stubs
  std::vector<TPPtr> StubAssociation::associateFinal(const std::vector<TTStubRef>& ttStubRefs) const {
    // Get all TPs that are matched to these stubs in at least 'tpMinLayers' layers and 'tpMinLayersPS' ps layers
    vector<TPPtr> tpPtrs = associate(ttStubRefs);
    // remove TPs with more then 'tpMaxBadStubs2S' not associated 2S stubs and more then 'tpMaxBadStubsPS' not associated PS stubs
    auto check = [this, &ttStubRefs](const TPPtr& tpPtr) {
      int bad2S(0);
      int badPS(0);
      for (const TTStubRef& ttStubRef : ttStubRefs) {
        const vector<TPPtr>& tpPtrs = findTrackingParticlePtrs(ttStubRef);
        if (find(tpPtrs.begin(), tpPtrs.end(), tpPtr) == tpPtrs.end())
          setup_->psModule(ttStubRef) ? badPS++ : bad2S++;
      }
      if (badPS > setup_->tpMaxBadStubsPS() || bad2S > setup_->tpMaxBadStubs2S())
        return true;
      return false;
    };
    tpPtrs.erase(remove_if(tpPtrs.begin(), tpPtrs.end(), check), tpPtrs.end());
    return tpPtrs;
  }

}  // namespace tt
