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
  BscG4Hit(const BscG4Hit &right);
  const BscG4Hit& operator=(const BscG4Hit &right);
  int operator==(const BscG4Hit &){return 0;}
  
  void         Draw() override{}
  void         Print() override;
  
public:
  
  G4ThreeVector   getEntry() const;
  void         setEntry(const G4ThreeVector& xyz);
  
  G4ThreeVector    getEntryLocalP() const;
  void         setEntryLocalP(const G4ThreeVector& xyz1);

  G4ThreeVector    getExitLocalP() const;
  void         setExitLocalP(const G4ThreeVector& xyz1);

  float        getEM() const;
  void         setEM (float e);
  
  float        getHadr() const;
  void         setHadr (float e);
  
  float        getIncidentEnergy() const;
  void         setIncidentEnergy (float e);
  
  int          getTrackID() const;
  void         setTrackID (int i);
  
  uint32_t     getUnitID() const;
  void         setUnitID (uint32_t i);
  
  double       getTimeSlice() const;     
  void         setTimeSlice(double d);
  int          getTimeSliceID() const;     
  
  void         addEnergyDeposit(float em, float hd);
  void         addEnergyDeposit(const BscG4Hit& aHit);
  
  float        getEnergyDeposit() const;
  
  float getPabs() const;
  float getTof() const;
  float getEnergyLoss() const;
  int getParticleType() const;

  void setPabs(float e);
  void setTof(float e);
  void setEnergyLoss(float e);
  void setParticleType(int i);

  float getThetaAtEntry() const;   
  float getPhiAtEntry() const;

  void setThetaAtEntry(float t);
  void setPhiAtEntry(float f) ;

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
  
    G4ThreeVector entry;             //Entry point
    G4ThreeVector entrylp;    //Entry  local point
    G4ThreeVector exitlp;    //Exit  local point
  float       elem;              //EnergyDeposit of EM particles
  float       hadr;              //EnergyDeposit of HD particles
  float       theIncidentEnergy; //Energy of the primary particle
  int          theTrackID;        //Identification number of the primary
                                  //particle
  double       theTimeSlice;      //Time Slice Identification

  int theUnitID;         //Bsc Unit Number

  float theX;
  float theY;
  float theZ;
  float thePabs  ;
    float theTof ;
    float theEnergyLoss   ;
    int theParticleType ;


  float theThetaAtEntry ;
    float thePhiAtEntry    ;

    int theParentId;
    float theVx;
    float theVy;
    float theVz;
};

std::ostream& operator<<(std::ostream&, const BscG4Hit&);

#endif

