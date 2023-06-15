#ifndef SimG4Core_TrackWithHistory_H
#define SimG4Core_TrackWithHistory_H

#include "G4Track.hh"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "G4Allocator.hh"

class G4VProcess;
class G4PrimaryParticle;
/** The part of the information about a SimTrack that we need from
 *  a G4Track
 */

class TrackWithHistory {
public:
  /** The constructor is called at time, 
     *  when some of the information may not available yet.
     */
  TrackWithHistory(const G4Track *g4track, int pID);
  TrackWithHistory(const G4PrimaryParticle*, int trackID, const math::XYZVectorD& pos, double time);
  ~TrackWithHistory() = default;

  inline void *operator new(size_t);
  inline void operator delete(void *TrackWithHistory);

  void setToBeSaved() { saved_ = true; }
  int trackID() const { return trackID_; }
  int particleID() const { return particleID_; }
  int parentID() const { return parentID_; }
  int genParticleID() const { return genParticleID_; }
  int vertexID() const { return vertexID_; }
  const math::XYZVectorD &momentum() const { return momentum_; }
  double totalEnergy() const { return totalEnergy_; }
  const math::XYZVectorD &vertexPosition() const { return vertexPosition_; }
  double globalTime() const { return globalTime_; }
  double localTime() const { return localTime_; }
  double properTime() const { return properTime_; }
  const G4VProcess *creatorProcess() const { return creatorProcess_; }
  double weight() const { return weight_; }
  void setTrackID(int i) { trackID_ = i; }
  void setParentID(int i) { parentID_ = i; }
  void setVertexID(int i) { vertexID_ = i; }
  void setGenParticleID(int i) { genParticleID_ = i; }
  bool storeTrack() const { return storeTrack_; }
  bool saved() const { return saved_; }

  // Boundary crossing variables
  void setCrossedBoundaryPosMom(int id,
                                const math::XYZTLorentzVectorF& position,
                                const math::XYZTLorentzVectorF& momentum) {
    crossedBoundary_ = true;
    idAtBoundary_ = id;
    positionAtBoundary_ = position;
    momentumAtBoundary_ = momentum;
  }
  bool crossedBoundary() const { return crossedBoundary_; }
  const math::XYZTLorentzVectorF &getPositionAtBoundary() const { return positionAtBoundary_; }
  const math::XYZTLorentzVectorF &getMomentumAtBoundary() const { return momentumAtBoundary_; }
  int getIDAtBoundary() const { return idAtBoundary_; }

  // tracker surface
  const math::XYZVectorD& trackerSurfacePosition() const { return tkSurfacePosition_; }
  const math::XYZTLorentzVectorD& trackerSurfaceMomentum() const { return tkSurfaceMomentum_; }
  void setSurfacePosMom(const math::XYZVectorD& pos,
                        const math::XYZTLorentzVectorD& mom) {
    tkSurfacePosition_ = pos;
    tkSurfaceMomentum_ = mom;
  }

private:
  int trackID_;
  int particleID_;
  int parentID_;
  int genParticleID_{-1};
  int vertexID_{-1};
  math::XYZVectorD momentum_;
  double totalEnergy_;
  math::XYZVectorD vertexPosition_;
  double globalTime_;
  double localTime_;
  double properTime_;
  const G4VProcess *creatorProcess_;
  double weight_;
  bool storeTrack_;
  bool saved_{false};

  bool crossedBoundary_{false};
  int idAtBoundary_{-1};
  math::XYZTLorentzVectorF positionAtBoundary_{math::XYZTLorentzVectorF(0.f, 0.f, 0.f, 0.f)};
  math::XYZTLorentzVectorF momentumAtBoundary_{math::XYZTLorentzVectorF(0.f, 0.f, 0.f, 0.f)};
  math::XYZVectorD tkSurfacePosition_{math::XYZVectorD(0., 0., 0.)};
  math::XYZTLorentzVectorD tkSurfaceMomentum_{math::XYZTLorentzVectorD(0., 0., 0., 0.)};
};

extern G4ThreadLocal G4Allocator<TrackWithHistory> *fpTrackWithHistoryAllocator;

inline void *TrackWithHistory::operator new(size_t) {
  if (!fpTrackWithHistoryAllocator)
    fpTrackWithHistoryAllocator = new G4Allocator<TrackWithHistory>;
  return (void *)fpTrackWithHistoryAllocator->MallocSingle();
}

inline void TrackWithHistory::operator delete(void *aTwH) {
  fpTrackWithHistoryAllocator->FreeSingle((TrackWithHistory *)aTwH);
}

#endif
