///////////////////////////////////////////////////////////////////////////////
// File: FP420G4Hit.h
// Date: 02.2006
//
// Package:     FP420
// Class  :     FP420G4Hit
//
///////////////////////////////////////////////////////////////////////////////
#ifndef FP420G4Hit_h
#define FP420G4Hit_h

#include "G4VHit.hh"
#include <iostream>

#include "G4Step.hh"
//#include "G4StepPoint.hh"

class FP420G4Hit : public G4VHit {
public:
  FP420G4Hit();
  ~FP420G4Hit() override;
  FP420G4Hit(const FP420G4Hit& right);
  const FP420G4Hit& operator=(const FP420G4Hit& right);
  int operator==(const FP420G4Hit&) { return 0; }

  void Draw() override {}
  void Print() override;

public:
  G4ThreeVector getEntry() const;
  void setEntry(const G4ThreeVector& xyz);

  G4ThreeVector getEntryLocalP() const;
  void setEntryLocalP(const G4ThreeVector& xyz1);

  G4ThreeVector getExitLocalP() const;
  void setExitLocalP(const G4ThreeVector& xyz1);

  double getEM() const;
  void setEM(double e);

  double getHadr() const;
  void setHadr(double e);

  double getIncidentEnergy() const;
  void setIncidentEnergy(double e);

  //G4int          getTrackID() const;
  unsigned int getTrackID() const;
  void setTrackID(int i);

  unsigned int getUnitID() const;
  void setUnitID(unsigned int i);

  double getTimeSlice() const;
  void setTimeSlice(double d);
  int getTimeSliceID() const;

  void addEnergyDeposit(double em, double hd);
  void addEnergyDeposit(const FP420G4Hit& aHit);

  double getEnergyDeposit() const;

  float getPabs() const;
  float getTof() const;
  float getEnergyLoss() const;
  int getParticleType() const;

  void setPabs(float e);
  void setTof(float e);
  void addEnergyLoss(float e);
  void setEnergyLoss(float e);
  void setParticleType(short i);

  float getThetaAtEntry() const;
  float getPhiAtEntry() const;

  void setThetaAtEntry(float t);
  void setPhiAtEntry(float f);

  float getX() const;
  void setX(float t);
  float getY() const;
  float getZ() const;
  void setY(float t);
  void setZ(float t);

  int getParentId() const;
  float getVx() const;
  float getVy() const;
  float getVz() const;

  void setParentId(int p);
  void setVx(float p);
  void setVy(float p);
  void setVz(float p);

private:
  G4ThreeVector entry;       //Entry point
  G4ThreeVector entrylp;     //Entry  local point
  G4ThreeVector exitlp;      //Exit  local point
  double elem;               //EnergyDeposit of EM particles
  double hadr;               //EnergyDeposit of HD particles
  double theIncidentEnergy;  //Energy of the primary particle
  G4int theTrackID;          //Identification number of the primary
                             //particle
  double theTimeSlice;       //Time Slice Identification

  int theUnitID;  //FP420 Unit Number

  float theX;
  float theY;
  float theZ;
  float thePabs;
  float theTof;
  float theEnergyLoss;
  int theParticleType;

  float theThetaAtEntry;
  float thePhiAtEntry;

  int theParentId;
  float theVx;
  float theVy;
  float theVz;
};

std::ostream& operator<<(std::ostream&, const FP420G4Hit&);

#endif
