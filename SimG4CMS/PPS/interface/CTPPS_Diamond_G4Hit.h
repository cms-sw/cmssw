///////////////////////////////////////////////////////////////////////////////
// Author 
// Seyed Mohsen Etesami setesami@cern.ch 
// Feb 2016
///////////////////////////////////////////////////////////////////////////////
#ifndef CTPPS_CTPPS_Diamond_G4Hit_h
#define CTPPS_CTPPS_Diamond_G4Hit_h 

#include "G4VHit.hh"
#include <CLHEP/Vector/ThreeVector.h>
#include <boost/cstdint.hpp>
#include <iostream>

using CLHEP::Hep3Vector;

class CTPPS_Diamond_G4Hit : public G4VHit {
  
public:
  CTPPS_Diamond_G4Hit();
  ~CTPPS_Diamond_G4Hit() override;
  CTPPS_Diamond_G4Hit(const CTPPS_Diamond_G4Hit &right);
  const CTPPS_Diamond_G4Hit& operator=(const CTPPS_Diamond_G4Hit &right);
  int operator==(const CTPPS_Diamond_G4Hit &){return 0;}
  
  void Draw() override{}
  void Print() override;
  
public:
  Hep3Vector getEntry() const;
  void setEntry(Hep3Vector xyz);
  Hep3Vector getExit() const;
  void setExit(Hep3Vector xyz);
  
  void setLocalEntry(const Hep3Vector &theLocalEntryPoint);
  void setLocalExit(const Hep3Vector &theLocalExitPoint);
  Hep3Vector getLocalEntry() const;
  Hep3Vector getLocalExit() const;
  
  double getIncidentEnergy() const;
  void setIncidentEnergy (double e);
  
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
  Hep3Vector entry;             //Entry point
  Hep3Vector exit;    //Exit point
  Hep3Vector local_entry;    //local entry point
  Hep3Vector local_exit;     //local exit point
  double theIncidentEnergy; //Energy of the primary particle
  int theTrackID;        //Identification number of the primary particle
  uint32_t theUnitID;         //CTPPS DetectorId
  double theTimeSlice;      //Time Slice Identification
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

std::ostream& operator<<(std::ostream&, const CTPPS_Diamond_G4Hit&);

#endif  //CTPPS_Diamond_G4Hit_h
