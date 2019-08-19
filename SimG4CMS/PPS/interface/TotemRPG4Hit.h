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
  G4ThreeVector getEntry() const;
  void setEntry(G4ThreeVector xyz);
  G4ThreeVector getExit() const;
  void setExit(G4ThreeVector xyz);

  void setLocalEntry(const G4ThreeVector &theLocalEntryPoint);
  void setLocalExit(const G4ThreeVector &theLocalExitPoint);
  G4ThreeVector getLocalEntry() const;
  G4ThreeVector getLocalExit() const;

  double getIncidentEnergy() const;
  void setIncidentEnergy(double e);

  unsigned int getTrackID() const;
  void setTrackID(int i);

  int getUnitID() const;
  void setUnitID(unsigned int i);

  double getTimeSlice() const;
  void setTimeSlice(double d);
  int getTimeSliceID() const;

  double getPabs() const;
  double getTof() const;
  double getEnergyLoss() const;
  int getParticleType() const;

  void setPabs(double e);
  void setTof(double e);
  void setEnergyLoss(double e);
  void setParticleType(short i);

  void addEnergyLoss(double e);

  double getThetaAtEntry() const;
  double getPhiAtEntry() const;

  void setThetaAtEntry(double t);
  void setPhiAtEntry(double f);

  double getX() const;
  void setX(double t);
  double getY() const;
  double getZ() const;
  void setY(double t);
  void setZ(double t);

  int getParentId() const;
  double getVx() const;
  double getVy() const;
  double getVz() const;

  void setParentId(int p);
  void setVx(double p);
  void setVy(double p);
  void setVz(double p);

  void set_p_x(double p);
  void set_p_y(double p);
  void set_p_z(double p);

  double get_p_x() const;
  double get_p_y() const;
  double get_p_z() const;

private:
  G4ThreeVector entry;        //Entry point
  G4ThreeVector exit;         //Exit point
  G4ThreeVector local_entry;  //local entry point
  G4ThreeVector local_exit;   //local exit point
  double theIncidentEnergy;   //Energy of the primary particle
  int theTrackID;             //Identification number of the primary
                              //particle
  uint32_t theUnitID;         //Totem Unit Number
  double theTimeSlice;        //Time Slice Identification

  double theX;
  double theY;
  double theZ;
  double thePabs;
  double theTof;
  double theEnergyLoss;
  int theParticleType;

  double theThetaAtEntry;
  double thePhiAtEntry;
  G4ThreeVector theEntryPoint;
  G4ThreeVector theExitPoint;

  int theParentId;
  double theVx;
  double theVy;
  double theVz;

  double p_x, p_y, p_z;
};

std::ostream &operator<<(std::ostream &, const TotemRPG4Hit &);

#endif  //PPS_TotemRPG4Hit_h
