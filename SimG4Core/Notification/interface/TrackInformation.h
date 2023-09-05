#ifndef SimG4Core_TrackInformation_H
#define SimG4Core_TrackInformation_H

#include "G4VUserTrackInformation.hh"
#include "G4Allocator.hh"
#include "G4Track.hh"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

class TrackInformation : public G4VUserTrackInformation {
public:
  TrackInformation(){};
  ~TrackInformation() override = default;
  inline void *operator new(size_t);
  inline void operator delete(void *TrackInformation);

  bool storeTrack() const { return storeTrack_; }
  /// can only be set to true, cannot be reset to false!
  void setStoreTrack() {
    storeTrack_ = true;
    isInHistory_ = true;
  }

  bool isPrimary() const { return isPrimary_; }
  void setPrimary(bool v) { isPrimary_ = v; }

  bool hasHits() const { return hasHits_; }
  void setHasHits(bool v) { hasHits_ = v; }

  bool isGeneratedSecondary() const { return isGeneratedSecondary_; }
  void setGeneratedSecondary(bool v) { isGeneratedSecondary_ = v; }

  bool isInHistory() const { return isInHistory_; }
  void putInHistory() { isInHistory_ = true; }

  bool isAncestor() const { return flagAncestor_; }
  void setAncestor() { flagAncestor_ = true; }

  int mcTruthID() const { return mcTruthID_; }
  void setMCTruthID(int id) { mcTruthID_ = id; }

  // Calo section
  int getIDonCaloSurface() const { return idOnCaloSurface_; }
  void setIDonCaloSurface(int id, int ical, int last, int pdgID, double p) {
    idOnCaloSurface_ = id;
    idCaloVolume_ = ical;
    idLastVolume_ = last;
    caloSurfaceParticlePID_ = pdgID;
    caloSurfaceParticleP_ = p;
  }
  int getIDCaloVolume() const { return idCaloVolume_; }
  int getIDLastVolume() const { return idLastVolume_; }
  bool caloIDChecked() const { return caloIDChecked_; }
  void setCaloIDChecked(bool f) { caloIDChecked_ = f; }
  int caloSurfaceParticlePID() const { return caloSurfaceParticlePID_; }
  void setCaloSurfaceParticlePID(int id) { caloSurfaceParticlePID_ = id; }
  double caloSurfaceParticleP() const { return caloSurfaceParticleP_; }
  void setCaloSurfaceParticleP(double p) { caloSurfaceParticleP_ = p; }

  // Boundary crossing variables
  void setCrossedBoundary(const G4Track *track);
  bool crossedBoundary() const { return crossedBoundary_; }
  const math::XYZTLorentzVectorF &getPositionAtBoundary() const { return positionAtBoundary_; }
  const math::XYZTLorentzVectorF &getMomentumAtBoundary() const { return momentumAtBoundary_; }
  bool startedInFineVolume() const { return startedInFineVolume_; }
  void setStartedInFineVolume(bool flag = true) {
    startedInFineVolume_ = flag;
    startedInFineVolumeIsSet_ = true;
  }
  bool startedInFineVolumeIsSet() { return startedInFineVolumeIsSet_; }

  // Generator information
  int genParticlePID() const { return genParticlePID_; }
  void setGenParticlePID(int id) { genParticlePID_ = id; }
  double genParticleP() const { return genParticleP_; }
  void setGenParticleP(double p) { genParticleP_ = p; }

  // remember the PID of particle entering the CASTOR detector. This is needed
  // in order to scale the hadronic response
  bool hasCastorHit() const { return hasCastorHit_; }
  void setCastorHitPID(const int pid) {
    hasCastorHit_ = true;
    castorHitPID_ = pid;
  }
  int getCastorHitPID() const { return castorHitPID_; }

  // methods for MTD info management
  //
  void setFromTtoBTL() { mtdStatus_ |= 1 << 0; }  // 1st bit
  bool isFromTtoBTL() const { return (mtdStatus_ >> 0) & 1; }
  void setFromBTLtoT() { mtdStatus_ |= 1 << 1; }  // 2nd bit
  bool isFromBTLtoT() const { return (mtdStatus_ >> 1) & 1; }
  void setBTLlooper() { mtdStatus_ |= 1 << 2; }  // 3th bit
  bool isBTLlooper() const { return (mtdStatus_ >> 2) & 1; }
  void setInTrkFromBackscattering() { mtdStatus_ |= 1 << 3; }  // 4th bit
  bool isInTrkFromBackscattering() const { return (mtdStatus_ >> 3) & 1; }
  void setExtSecondary() { mtdStatus_ |= 1 << 4; }  //5th bit
  bool isExtSecondary() const { return (mtdStatus_ >> 4) & 1; }

  void Print() const override;

private:
  bool storeTrack_{false};
  bool isPrimary_{false};
  bool hasHits_{false};
  bool isGeneratedSecondary_{false};
  bool isInHistory_{false};
  bool flagAncestor_{false};
  bool caloIDChecked_{false};
  bool crossedBoundary_{false};
  bool startedInFineVolume_{false};
  bool startedInFineVolumeIsSet_{false};
  bool hasCastorHit_{false};
  int idOnCaloSurface_{0};
  int idCaloVolume_{-1};
  int idLastVolume_{-1};
  int genParticlePID_{-1};
  int mcTruthID_{-1};
  int caloSurfaceParticlePID_{0};
  int castorHitPID_{0};
  uint8_t mtdStatus_{0};
  double genParticleP_{0.};
  double caloSurfaceParticleP_{0.};
  math::XYZTLorentzVectorF positionAtBoundary_;
  math::XYZTLorentzVectorF momentumAtBoundary_;
};

extern G4ThreadLocal G4Allocator<TrackInformation> *fpTrackInformationAllocator;

inline void *TrackInformation::operator new(size_t) {
  if (!fpTrackInformationAllocator)
    fpTrackInformationAllocator = new G4Allocator<TrackInformation>;
  return (void *)fpTrackInformationAllocator->MallocSingle();
}

inline void TrackInformation::operator delete(void *trkInfo) {
  fpTrackInformationAllocator->FreeSingle((TrackInformation *)trkInfo);
}

#endif
