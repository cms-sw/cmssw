#ifndef SimTrack_H
#define SimTrack_H

#include "SimDataFormats/Track/interface/CoreSimTrack.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Utilities/interface/Exception.h"

class SimTrack : public CoreSimTrack {
public:
  typedef CoreSimTrack Core;

  /// constructor
  SimTrack();
  SimTrack(int ipart, const math::XYZTLorentzVectorD& p);

  /// full constructor (pdg type, momentum, time,
  /// index of parent vertex in final vector
  /// index of corresponding gen part in final vector)
  SimTrack(int ipart, const math::XYZTLorentzVectorD& p, int iv, int ig);

  SimTrack(int ipart,
           const math::XYZTLorentzVectorD& p,
           int iv,
           int ig,
           const math::XYZVectorD& tkp,
           const math::XYZTLorentzVectorD& tkm);

  /// constructor from transient
  SimTrack(const CoreSimTrack& t, int iv, int ig);

  /// index of the vertex in the Event container (-1 if no vertex)
  int vertIndex() const { return ivert; }
  bool noVertex() const { return ivert == -1; }

  /// index of the corresponding Generator particle in the Event container (-1 if no Genpart)
  bool isPrimary() const { return (trackInfo_ >> 1) & 1; }
  int genpartIndex() const { return isPrimary() ? igenpart : -1; }
  bool noGenpart() const { return isPrimary() ? igenpart == -1 : true; }

  const math::XYZVectorD& trackerSurfacePosition() const { return tkposition; }

  const math::XYZTLorentzVectorD& trackerSurfaceMomentum() const { return tkmomentum; }

  inline void setTkPosition(const math::XYZVectorD& pos) { tkposition = pos; }

  inline void setTkMomentum(const math::XYZTLorentzVectorD& mom) { tkmomentum = mom; }

  inline void setVertexIndex(const int v) { ivert = v; }

  void setCrossedBoundaryVars(bool crossedBoundary,
                              int idAtBoundary,
                              math::XYZTLorentzVectorF positionAtBoundary,
                              math::XYZTLorentzVectorF momentumAtBoundary) {
    if (crossedBoundary)
      trackInfo_ |= (1 << 2);
    idAtBoundary_ = idAtBoundary;
    positionAtBoundary_ = positionAtBoundary;
    momentumAtBoundary_ = momentumAtBoundary;
  }
  bool crossedBoundary() const { return (trackInfo_ >> 2) & 1; }
  const math::XYZTLorentzVectorF& getPositionAtBoundary() const { return positionAtBoundary_; }
  const math::XYZTLorentzVectorF& getMomentumAtBoundary() const { return momentumAtBoundary_; }
  int getIDAtBoundary() const { return idAtBoundary_; }

  bool isFromBackScattering() const { return trackInfo_ & 1; }
  void setFromBackScattering() { trackInfo_ |= 1; }

  void setIsPrimary() { trackInfo_ |= (1 << 1); }
  void setGenParticleID(const int idx) { igenpart = idx; }
  int getPrimaryID() const { return igenpart; }
  uint8_t getTrackInfo() const { return trackInfo_; }

private:
  int ivert;
  int igenpart;

  math::XYZVectorD tkposition;
  math::XYZTLorentzVectorD tkmomentum;

  int idAtBoundary_;
  math::XYZTLorentzVectorF positionAtBoundary_;
  math::XYZTLorentzVectorF momentumAtBoundary_;
  uint8_t trackInfo_;
  // explanation of trackInfo bits:
  // 00000001 = simTrack is from backscattering
  // 00000010 = simTrack is of a primary particle
  // 00000100 = simTrack crossed the boundary
};

#include <iosfwd>
std::ostream& operator<<(std::ostream& o, const SimTrack& t);

#endif
