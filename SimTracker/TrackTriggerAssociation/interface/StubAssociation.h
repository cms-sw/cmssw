#ifndef SimTracker_TrackTriggerAssociation_StubAssociation_h
#define SimTracker_TrackTriggerAssociation_StubAssociation_h

#include "SimTracker/TrackTriggerAssociation/interface/TTTypes.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <vector>
#include <map>

namespace tt {

  /*! \class  tt::StubAssociation
   *  \brief  Class to associate reconstrucable TrackingParticles with TTStubs and vice versa.
   *          It may associate multiple TPs with a TTStub and can therefore be used to associate
   *          TTTracks with TrackingParticles.
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class StubAssociation {
  public:
    StubAssociation() { setup_ = nullptr; }
    StubAssociation(const Setup* setup) : setup_(setup) {}
    ~StubAssociation() {}

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
    std::vector<TPPtr> findTrackingParticlePtrs(const TTStubRef& ttStubRef) const;
    // returns collection of TTStubRefs associated to given TPPtr
    std::vector<TTStubRef> findTTStubRefs(const TPPtr& tpPtr) const;
    // total number of stubs associated with TPs
    int numStubs() const { return mapTTStubRefsTPPtrs_.size(); };
    // total number of TPs associated with stubs
    int numTPs() const { return mapTPPtrsTTStubRefs_.size(); };
    // Get all TPs that are matched to these stubs in at least 'tpMinLayers' layers and 'tpMinLayersPS' ps layers
    std::vector<TPPtr> associate(const std::vector<TTStubRef>& ttStubRefs) const;
    // Get all TPs that are matched to these stubs in at least 'tpMinLayers' layers and 'tpMinLayersPS' ps layers with not more then 'tpMaxBadStubs2S' not associated 2S stubs and not more then 'tpMaxBadStubsPS' associated PS stubs
    std::vector<TPPtr> associateFinal(const std::vector<TTStubRef>& ttStubRefs) const;

  private:
    // stores, calculates and provides run-time constants
    const Setup* setup_;
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