#ifndef SimDataFormats_Associations_StubAssociation_h
#define SimDataFormats_Associations_StubAssociation_h

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/Associations/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <vector>
#include <deque>
#include <map>

namespace tt {

  /*! \class  tt::StubAssociation
   *  \brief  Class to store maps to associate TrackingParticles with TTStubs and vice versa.
   *          It may associate multiple TPs with a TTStub and can therefore be used to associate
   *          TTTracks with TrackingParticles.
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class StubAssociation {
  public:
    // deafault constructor
    StubAssociation() {}
    // destructor
    ~StubAssociation() = default;
    // insert a TPPtr and its associated collection of TTstubRefs into the underlayering maps
    void insert(const TPPtr& tpPtr, const std::vector<TTStubRef>& ttSTubRefs);
    // returns map containing TTStubRef and their associated collection of TPPtrs
    const std::map<TTStubRef, std::vector<TPPtr>>& getTTStubToTrackingParticlesMap() const {
      return mapTTStubRefsTPPtrs_;
    }
    // returns map containing TPPtr and their associated collection of TTStubRefs
    const std::map<TPPtr, std::vector<TTStubRef>>& getTrackingParticleToTTStubsMap() const {
      return mapTPPtrsTTStubRefs_;
    }
    // returns collection of TPPtrs associated to given TTstubRef
    const std::vector<TPPtr>& findTrackingParticlePtrs(const TTStubRef& ttStubRef) const;
    // returns collection of TTStubRefs associated to given TPPtr
    const std::vector<TTStubRef>& findTTStubRefs(const TPPtr& tpPtr) const;
    // total number of stubs associated with TPs
    int numStubs() const { return mapTTStubRefsTPPtrs_.size(); };
    // total number of TPs associated with stubs
    int numTPs() const { return mapTPPtrsTTStubRefs_.size(); };

  private:
    // map containing TTStubRef and their associated collection of TPPtrs
    std::map<TTStubRef, std::vector<TPPtr>> mapTTStubRefsTPPtrs_;
    // map containing TPPtr and their associated collection of TTStubRefs
    std::map<TPPtr, std::vector<TTStubRef>> mapTPPtrsTTStubRefs_;
    // empty container of TPPtr
    const std::vector<TPPtr> emptyTPPtrs_;
    // empty container of TTStubRef
    const std::vector<TTStubRef> emptyTTStubRefs_;
  };

}  // namespace tt

#endif
