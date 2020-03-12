#ifndef SimG4Core_TrackWithHistory_H
#define SimG4Core_TrackWithHistory_H

#include "G4Track.hh"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "G4Allocator.hh"

class G4VProcess;
class G4TrackToParticleID;
/** The part of the information about a SimTrack that we need from
 *  a G4Track
 */

class TrackWithHistory {
public:
  /** The constructor is called at PreUserTrackingAction time, 
     *  when some of the information is not available yet.
     */
  TrackWithHistory(const G4Track *g4track);
  ~TrackWithHistory() {}

  inline void *operator new(size_t);
  inline void operator delete(void *TrackWithHistory);

  void save() { saved_ = true; }
  unsigned int trackID() const { return trackID_; }
  int particleID() const { return particleID_; }
  int parentID() const { return parentID_; }
  int genParticleID() const { return genParticleID_; }
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
  void setGenParticleID(int i) { genParticleID_ = i; }
  bool storeTrack() const { return storeTrack_; }
  bool saved() const { return saved_; }
  /** Internal consistency check (optional).
     *  Method called at PostUserTrackingAction time, to check
     *  if the information is consistent with that provided
     *  to the constructor.
     */
  void checkAtEnd(const G4Track *);

private:
  unsigned int trackID_;
  int particleID_;
  int parentID_;
  int genParticleID_;
  math::XYZVectorD momentum_;
  double totalEnergy_;
  math::XYZVectorD vertexPosition_;
  double globalTime_;
  double localTime_;
  double properTime_;
  const G4VProcess *creatorProcess_;
  double weight_;
  bool storeTrack_;
  bool saved_;
  int extractGenID(const G4Track *gt) const;
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
