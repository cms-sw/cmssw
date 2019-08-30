///////////////////////////////////////////////////////////////////////////////
// Author
// Seyed Mohsen Etesami setesami@cern.ch
// Feb 2016
///////////////////////////////////////////////////////////////////////////////
#ifndef PPS_PPSDiamondG4Hit_h
#define PPS_PPSDiamondG4Hit_h

#include "G4VHit.hh"
#include "G4ThreeVector.hh"
#include <iostream>

class PPSDiamondG4Hit : public G4VHit {
public:
  PPSDiamondG4Hit();
  ~PPSDiamondG4Hit() override;
  PPSDiamondG4Hit(const PPSDiamondG4Hit &right);
  const PPSDiamondG4Hit &operator=(const PPSDiamondG4Hit &right);
  int operator==(const PPSDiamondG4Hit &) { return 0; }

  void Draw() override {}
  void Print() override;

public:
  const G4ThreeVector &getEntry() const;
  void setEntry(const G4ThreeVector &xyz);
  const G4ThreeVector &getExit() const;
  void setExit(const G4ThreeVector &xyz);

  void setLocalEntry(const G4ThreeVector &theLocalEntryPoint);
  void setLocalExit(const G4ThreeVector &theLocalExitPoint);
  const G4ThreeVector &getLocalEntry() const;
  const G4ThreeVector &getLocalExit() const;

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
  double getY() const;
  double getZ() const;

  void setX(double t);
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

  void setPx(double p);
  void setPy(double p);
  void setPz(double p);
  double getPx() const;
  double getPy() const;
  double getPz() const;

  double getGlobalTimehit() const;
  void setGlobalTimehit(double h);

private:
  G4ThreeVector entry_;        //Entry point
  G4ThreeVector exit_;         //Exit point
  G4ThreeVector local_entry_;  //local entry point
  G4ThreeVector local_exit_;   //local exit point
  double theIncidentEnergy_;   //Energy of the primary particle
  int theTrackID_;             //Identification number of the primary particle
  uint32_t theUnitID_;         //PPS DetectorId
  double theTimeSlice_;        //Time Slice Identification
  double theGlobaltimehit_;
  double theX_;
  double theY_;
  double theZ_;
  double thePabs_;
  double theTof_;
  double theEnergyLoss_;
  int theParticleType_;
  int theParentId_;
  double theVx_;
  double theVy_;
  double theVz_;
  double thePx_;
  double thePy_;
  double thePz_;
  double theThetaAtEntry_;
  double thePhiAtEntry_;
};

std::ostream &operator<<(std::ostream &, const PPSDiamondG4Hit &);

#endif  //PPSDiamondG4Hit_h
