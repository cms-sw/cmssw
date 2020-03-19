///////////////////////////////////////////////////////////////////////////////
// File: CaloG4Hit.h
// Date: 10.02 Taken from CMSCaloHit
//
// Hit class for Calorimeters (Ecal, Hcal, ...)
//
// One Hit object should be created
// -for each new particle entering the calorimeter
// -for each detector unit (= cristal or fiber or scintillator layer)
// -for each nanosecond of the shower development
//
// This implies that all hit objects created for a given shower
// have the same value for
// - Entry (= local coordinates of the entrance point of the particle
//            in the unit where the shower starts)
// - the TrackID (= Identification number of the incident particle)
// - the IncidentEnergy (= energy of that particle)
//
// Modified:
//
///////////////////////////////////////////////////////////////////////////////
#ifndef PPS_TotemRPG4Hit_h
#define PPS_TotemRPG4Hit_h 1

#include "G4VHit.hh"
#include "G4ThreeVector.hh"
#include "DataFormats/Math/interface/Point3D.h"
#include <iostream>

class TotemRPG4Hit : public G4VHit {
public:
  TotemRPG4Hit();
  ~TotemRPG4Hit() override = default;
  TotemRPG4Hit(const TotemRPG4Hit &right);
  const TotemRPG4Hit &operator=(const TotemRPG4Hit &right);
  int operator==(const TotemRPG4Hit &) { return 0; }

  void Draw() override {}
  void Print() override;

public:
  G4ThreeVector entry() const;
  void setEntry(G4ThreeVector xyz);
  G4ThreeVector exit() const;
  void setExit(G4ThreeVector xyz);

  void setLocalEntry(const G4ThreeVector &theLocalEntryPoint);
  void setLocalExit(const G4ThreeVector &theLocalExitPoint);
  G4ThreeVector localEntry() const;
  G4ThreeVector localExit() const;

  double incidentEnergy() const;
  void setIncidentEnergy(double e);

  unsigned int trackID() const;
  void setTrackID(int i);

  int unitID() const;
  void setUnitID(unsigned int i);

  double timeSlice() const;
  void setTimeSlice(double d);
  int timeSliceID() const;

  double p() const;
  double tof() const;
  double energyLoss() const;
  int particleType() const;

  void setP(double e);
  void setTof(double e);
  void setEnergyLoss(double e);
  void setParticleType(short i);

  void addEnergyLoss(double e);

  double thetaAtEntry() const;
  double phiAtEntry() const;

  void setThetaAtEntry(double t);
  void setPhiAtEntry(double f);

  double x() const;
  void setX(double t);
  double y() const;
  double z() const;
  void setY(double t);
  void setZ(double t);

  int parentId() const;
  double vx() const;
  double vy() const;
  double vz() const;

  void setParentId(int p);
  void setVx(double p);
  void setVy(double p);
  void setVz(double p);

  void setPx(double p);
  void setPy(double p);
  void setPz(double p);

  double px() const;
  double py() const;
  double pz() const;

private:
  G4ThreeVector entry_;        //Entry point
  G4ThreeVector exit_;         //Exit point
  G4ThreeVector local_entry_;  //local entry point
  G4ThreeVector local_exit_;   //local exit point
  double theIncidentEnergy_;   //Energy of the primary particle
  int theTrackID_;             //Identification number of the primary
                               //particle
  uint32_t theUnitID_;         //Totem Unit Number
  double theTimeSlice_;        //Time Slice Identification

  double theX_;
  double theY_;
  double theZ_;
  double thePabs_;
  double theTof_;
  double theEnergyLoss_;
  int theParticleType_;

  double theThetaAtEntry_;
  double thePhiAtEntry_;
  G4ThreeVector theEntryPoint_;
  G4ThreeVector theExitPoint_;

  int theParentId_;
  double theVx_;
  double theVy_;
  double theVz_;

  double thePx_, thePy_, thePz_;
};

std::ostream &operator<<(std::ostream &, const TotemRPG4Hit &);

#endif  //PPS_TotemRPG4Hit_h
