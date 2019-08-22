///////////////////////////////////////////////////////////////////////////////
// File: BscG4Hit.h
// Date: 02.2006
//
// Package:     Bsc
// Class  :     BscG4Hit
//
///////////////////////////////////////////////////////////////////////////////
#ifndef BscG4Hit_h
#define BscG4Hit_h

#include "G4VHit.hh"
#include "G4ThreeVector.hh"
#include <cstdint>
#include <iostream>

class BscG4Hit : public G4VHit {
public:
  BscG4Hit();
  ~BscG4Hit() override;
  BscG4Hit(const BscG4Hit& right);
  const BscG4Hit& operator=(const BscG4Hit& right);
  int operator==(const BscG4Hit&) { return 0; }

  void Draw() override {}
  void Print() override;

public:
  const G4ThreeVector& getEntry() const { return entry; };
  void setEntry(const G4ThreeVector& xyz);

  const G4ThreeVector& getEntryLocalP() const { return entrylp; };
  void setEntryLocalP(const G4ThreeVector& xyz) { entrylp = xyz; };

  const G4ThreeVector& getExitLocalP() const { return exitlp; };
  void setExitLocalP(const G4ThreeVector& xyz) { exitlp = xyz; };

  float getEM() const { return elem; };
  void setEM(float e) {
    elem = e;
    theEnergyLoss = elem + hadr;
  };

  float getHadr() const { return hadr; };
  void setHadr(float e) {
    hadr = e;
    theEnergyLoss = elem + hadr;
  };

  float getIncidentEnergy() const { return theIncidentEnergy; };
  void setIncidentEnergy(float e) { theIncidentEnergy = e; };

  int getTrackID() const { return theTrackID; };
  void setTrackID(int id) { theTrackID = id; };

  uint32_t getUnitID() const { return theUnitID; };
  void setUnitID(uint32_t id) { theUnitID = id; };

  double getTimeSlice() const { return theTimeSlice; };
  void setTimeSlice(double d) { theTimeSlice = d; };
  int getTimeSliceID() const { return (int)theTimeSlice; };
  void addEnergyDeposit(float em, float hd);
  void addEnergyDeposit(const BscG4Hit& aHit);

  float getEnergyDeposit() const { return theEnergyLoss; };

  float getPabs() const { return thePabs; };
  float getTof() const { return theTof; };
  float getEnergyLoss() const { return theEnergyLoss; };
  int getParticleType() const { return theParticleType; };

  void setPabs(float e) { thePabs = e; };
  void setTof(float e) { theTof = e; };
  void setEnergyLoss(float e) { theEnergyLoss = e; };
  void setParticleType(int i) { theParticleType = i; };

  float getThetaAtEntry() const { return theThetaAtEntry; };
  float getPhiAtEntry() const { return thePhiAtEntry; };

  void setThetaAtEntry(float t) { theThetaAtEntry = t; };
  void setPhiAtEntry(float f) { thePhiAtEntry = f; };

  float getX() const { return theX; };
  void setX(float t) { theX = t; };
  float getY() const { return theY; };
  float getZ() const { return theZ; };
  void setY(float t) { theY = t; };
  void setZ(float t) { theZ = t; };

  void setHitPosition(const G4ThreeVector&);
  void setVertexPosition(const G4ThreeVector&);

  int getParentId() const { return theParentId; };
  int getProcessId() const { return theProcessId; };
  float getVx() const { return theVx; };
  float getVy() const { return theVy; };
  float getVz() const { return theVz; };

  void setParentId(int p) { theParentId = p; };
  void setProcessId(int p) { theProcessId = p; };
  void setVx(float p) { theVx = p; };
  void setVy(float p) { theVy = p; };
  void setVz(float p) { theVz = p; };

private:
  G4ThreeVector entry;      //Entry point
  G4ThreeVector entrylp;    //Entry  local point
  G4ThreeVector exitlp;     //Exit  local point
  float elem;               //EnergyDeposit of EM particles
  float hadr;               //EnergyDeposit of HD particles
  float theIncidentEnergy;  //Energy of the primary particle
  int theTrackID;           //Geant4 track ID
  double theTimeSlice;      //Time Slice Identification

  int theUnitID;  // Unit Number

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
  int theProcessId;
  float theVx;
  float theVy;
  float theVz;
};

std::ostream& operator<<(std::ostream&, const BscG4Hit&);

#endif
