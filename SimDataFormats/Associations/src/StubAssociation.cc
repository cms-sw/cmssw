#include "SimDataFormats/Associations/interface/StubAssociation.h"

#include <vector>
#include <deque>

namespace tt {

  // insert a TPPtr and its associated collection of TTstubRefs into the underlayering maps
  void StubAssociation::insert(const TPPtr& tpPtr, const std::vector<TTStubRef>& ttSTubRefs) {
    mapTPPtrsTTStubRefs_.insert({tpPtr, ttSTubRefs});
    for (const TTStubRef& ttSTubRef : ttSTubRefs)
      mapTTStubRefsTPPtrs_[ttSTubRef].push_back(tpPtr);
  }

  // returns collection of TPPtrs associated to given TTstubRef
  const std::vector<TPPtr>& StubAssociation::findTrackingParticlePtrs(const TTStubRef& ttStubRef) const {
    const auto it = mapTTStubRefsTPPtrs_.find(ttStubRef);
    return it != mapTTStubRefsTPPtrs_.end() ? it->second : emptyTPPtrs_;
  }

  // returns collection of TTStubRefs associated to given TPPtr
  const std::vector<TTStubRef>& StubAssociation::findTTStubRefs(const TPPtr& tpPtr) const {
    const auto it = mapTPPtrsTTStubRefs_.find(tpPtr);
    return it != mapTPPtrsTTStubRefs_.end() ? it->second : emptyTTStubRefs_;
  }

}  // namespace tt
