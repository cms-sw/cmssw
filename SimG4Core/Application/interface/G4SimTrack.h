#ifndef SimG4Core_G4SimTrack_H
#define SimG4Core_G4SimTrack_H

#include "CLHEP/Vector/ThreeVector.h"

class G4SimTrack
{
public:
    G4SimTrack() {}
    G4SimTrack(int iid, int ipart, const Hep3Vector & ip, double ie) 
	: id_(iid),ipart_(ipart),ip_(ip),ie_(ie),ivert_(-1),igenpart_(-1),
	  parentID_(-1),parentMomentum_(0.) {}
    G4SimTrack(int iid, int ipart, const Hep3Vector & ip, double ie,
	       int iv,  int ig,    const Hep3Vector & ipmom) 
	: id_(iid),ipart_(ipart),ip_(ip),ie_(ie),ivert_(iv),igenpart_(ig),
	  parentMomentum_(ipmom) {}
    ~G4SimTrack() {}
    const int id() const { return id_; }
    const int part() const { return ipart_; }
    const Hep3Vector & momentum() const { return ip_; }
    const double energy() const { return ie_; }
    int const ivert() const { return ivert_; }
    int const igenpart() const { return igenpart_; }
    // parent momentum at interaction
    const Hep3Vector & parentMomentum() const { return parentMomentum_; } 
    // parent track ID (only stored if parent momentum at interaction
    // is stored, else = -1)
    const int parentID() const { return parentID_; }
private:
    int id_;
    int ipart_;
    Hep3Vector ip_;
    double ie_;
    int ivert_; 
    int igenpart_; 
    int parentID_;
    Hep3Vector parentMomentum_;
};

#endif
