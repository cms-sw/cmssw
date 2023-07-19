#ifndef SimG4Core_TrackWithHistory_H
#define SimG4Core_TrackWithHistory_H

#include "G4Track.hh"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "G4Allocator.hh"

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
  TrackWithHistory(const G4PrimaryParticle *, int trackID, const math::XYZVectorD &pos, const double time);
  ~TrackWithHistory() = default;

  inline void *operator new(std::size_t);
  inline void operator delete(void *TrackWithHistory);

  int trackID() const { return trackID_; }
  int particleID() const { return pdgID_; }
  int parentID() const { return parentID_; }
  int genParticleID() const { return genParticleID_; }
  int vertexID() const { return vertexID_; }
  int processType() const { return procType_; }
  int getIDAtBoundary() const { return idAtBoundary_; }

  void setTrackID(int i) { trackID_ = i; }
  void setParentID(int i) { parentID_ = i; }
  void setVertexID(int i) { vertexID_ = i; }
  void setGenParticleID(int i) { genParticleID_ = i; }

  double totalEnergy() const { return totalEnergy_; }
  double time() const { return time_; }
  double weight() const { return weight_; }
  void setToBeSaved() { saved_ = true; }
  bool storeTrack() const { return storeTrack_; }
  bool saved() const { return saved_; }
  bool crossedBoundary() const { return crossedBoundary_; }

  const math::XYZVectorD &momentum() const { return momentum_; }
  const math::XYZVectorD &vertexPosition() const { return vertexPosition_; }

  // Boundary crossing variables
  void setCrossedBoundaryPosMom(int id,
                                const math::XYZTLorentzVectorF &position,
                                const math::XYZTLorentzVectorF &momentum) {
    crossedBoundary_ = true;
    idAtBoundary_ = id;
    positionAtBoundary_ = position;
    momentumAtBoundary_ = momentum;
  }
  const math::XYZTLorentzVectorF &getPositionAtBoundary() const { return positionAtBoundary_; }
  const math::XYZTLorentzVectorF &getMomentumAtBoundary() const { return momentumAtBoundary_; }

  // tracker surface
  const math::XYZVectorD &trackerSurfacePosition() const { return tkSurfacePosition_; }
  const math::XYZTLorentzVectorD &trackerSurfaceMomentum() const { return tkSurfaceMomentum_; }
  void setSurfacePosMom(const math::XYZVectorD &pos, const math::XYZTLorentzVectorD &mom) {
    tkSurfacePosition_ = pos;
    tkSurfaceMomentum_ = mom;
  }

private:
  int trackID_;
  int pdgID_;
  int parentID_;
  int genParticleID_{-1};
  int vertexID_{-1};
  int idAtBoundary_{-1};
  int procType_{0};
  double totalEnergy_;
  double time_;  // lab system
  double weight_;
  math::XYZVectorD momentum_;
  math::XYZVectorD vertexPosition_;
  math::XYZTLorentzVectorF positionAtBoundary_{math::XYZTLorentzVectorF(0.f, 0.f, 0.f, 0.f)};
  math::XYZTLorentzVectorF momentumAtBoundary_{math::XYZTLorentzVectorF(0.f, 0.f, 0.f, 0.f)};
  math::XYZVectorD tkSurfacePosition_{math::XYZVectorD(0., 0., 0.)};
  math::XYZTLorentzVectorD tkSurfaceMomentum_{math::XYZTLorentzVectorD(0., 0., 0., 0.)};
  bool storeTrack_{false};
  bool saved_{false};
  bool crossedBoundary_{false};
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
