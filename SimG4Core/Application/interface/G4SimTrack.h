#ifndef SimG4Core_G4SimTrack_H
#define SimG4Core_G4SimTrack_H

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector> 
#include <cmath>

class G4SimTrack
{
public:
  G4SimTrack() {}
  
  G4SimTrack(int iid, int ipart, const math::XYZVectorD & ip, double ie)
    : id_(iid), ipart_(ipart), ip_(ip), ie_(ie),ivert_(-1),igenpart_(-1), parentID_(-1),
      parentMomentum_(math::XYZVectorD(0.,0.,0.)),tkSurfacePosition_(math::XYZVectorD(0.,0.,0.)),
      tkSurfaceMomentum_(math::XYZTLorentzVectorD(0.,0.,0.,0.)){} 
  
  G4SimTrack(int iid, int ipart, const math::XYZVectorD& ip, double ie,
	     int iv,  int ig, const math::XYZVectorD& ipmom) 
    : id_(iid),ipart_(ipart),ip_(ip),ie_(ie),ivert_(iv),igenpart_(ig),parentMomentum_(ipmom),
      tkSurfacePosition_(math::XYZVectorD(0.,0.,0.)),tkSurfaceMomentum_(math::XYZTLorentzVectorD(0.,0.,0.,0.)) {}

  G4SimTrack(int iid, int ipart, const math::XYZVectorD& ip, double ie,
	     int iv,  int ig, const math::XYZVectorD& ipmom, 
	     const math::XYZVectorD & tkpos, const math::XYZTLorentzVectorD & tkmom) 
    : id_(iid),ipart_(ipart),ip_(ip),ie_(ie),ivert_(iv),igenpart_(ig),parentMomentum_(ipmom),
      tkSurfacePosition_(tkpos),tkSurfaceMomentum_(tkmom) {}

  ~G4SimTrack() {}

  int id() const { return id_; }
  int part() const { return ipart_; }
  const math::XYZVectorD & momentum() const { return ip_; }
  double energy() const { return ie_; }
  int ivert() const { return ivert_; }
  int igenpart() const { return igenpart_; }
  // parent momentum at interaction
  const math::XYZVectorD & parentMomentum() const { return parentMomentum_; } 
  // Information at level of tracker surface
  const math::XYZVectorD & trackerSurfacePosition() const {return tkSurfacePosition_;}
  const math::XYZTLorentzVectorD & trackerSurfaceMomentum() const {return tkSurfaceMomentum_;}
  // parent track ID (only stored if parent momentum at interaction
  // is stored, else = -1)
  int parentID() const { return parentID_; }
private:
  int id_;
  int ipart_;
  math::XYZVectorD ip_;
  double ie_;
  int ivert_; 
  int igenpart_; 
  int parentID_;
  math::XYZVectorD parentMomentum_;
  math::XYZVectorD tkSurfacePosition_;
  math::XYZTLorentzVectorD tkSurfaceMomentum_;
};

#endif
