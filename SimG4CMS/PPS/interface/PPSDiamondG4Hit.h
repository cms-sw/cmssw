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
  const G4ThreeVector& getEntry() const;
  void setEntry(const G4ThreeVector& xyz);
  const G4ThreeVector& getExit() const;
  void setExit(const G4ThreeVector& xyz);

  void setLocalEntry(const G4ThreeVector &theLocalEntryPoint);
  void setLocalExit(const G4ThreeVector &theLocalExitPoint);
  const G4ThreeVector& getLocalEntry() const;
  const G4ThreeVector& getLocalExit() const;

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

  double getGlobalTimehit() const;
  void setGlobalTimehit(double h);

private:
  G4ThreeVector entry;        //Entry point
  G4ThreeVector exit;         //Exit point
  G4ThreeVector local_entry;  //local entry point
  G4ThreeVector local_exit;   //local exit point
  double theIncidentEnergy;   //Energy of the primary particle
  int theTrackID;             //Identification number of the primary particle
  uint32_t theUnitID;         //PPS DetectorId
  double theTimeSlice;        //Time Slice Identification
  double theGlobaltimehit;
  double theX;
  double theY;
  double theZ;
  double thePabs;
  double theTof;
  double theEnergyLoss;
  int theParticleType;
  int theParentId;
  double theVx;
  double theVy;
  double theVz;
  double p_x, p_y, p_z;
  double theThetaAtEntry;
  double thePhiAtEntry;
};

std::ostream &operator<<(std::ostream &, const PPSDiamondG4Hit &);

#endif  //PPSDiamondG4Hit_h
