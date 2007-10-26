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
#include <CLHEP/Vector/ThreeVector.h>
#include <boost/cstdint.hpp>
#include <iostream>

#include "G4Step.hh"
//#include "G4StepPoint.hh"

class BscG4Hit : public G4VHit {
  
public:
  
  BscG4Hit();
  ~BscG4Hit();
  BscG4Hit(const BscG4Hit &right);
  const BscG4Hit& operator=(const BscG4Hit &right);
  int operator==(const BscG4Hit &){return 0;}
  
  void         Draw(){}
  void         Print();
  
public:
  
  G4ThreeVector   getEntry() const;
  void         setEntry(G4ThreeVector xyz);
  
  G4ThreeVector    getEntryLocalP() const;
  void         setEntryLocalP(G4ThreeVector xyz1);

  G4ThreeVector    getExitLocalP() const;
  void         setExitLocalP(G4ThreeVector xyz1);

  double       getEM() const;
  void         setEM (double e);
  
  double       getHadr() const;
  void         setHadr (double e);
  
  double       getIncidentEnergy() const;
  void         setIncidentEnergy (double e);
  
  G4int          getTrackID() const;
  void         setTrackID (int i);
  
  unsigned int getUnitID() const;
  void         setUnitID (unsigned int i);
  
  double       getTimeSlice() const;     
  void         setTimeSlice(double d);
  int          getTimeSliceID() const;     
  
  void         addEnergyDeposit(double em, double hd);
  void         addEnergyDeposit(const BscG4Hit& aHit);
  
  double       getEnergyDeposit() const;
  
  float getPabs() const;
  float getTof() const;
  float getEnergyLoss() const;
  int getParticleType() const;

 void setPabs(float e)      ;
  void setTof(float e)  ;
  void setEnergyLoss(float e) ;
  void setParticleType(short i) ;

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
  double       elem;              //EnergyDeposit of EM particles
  double       hadr;              //EnergyDeposit of HD particles
  double       theIncidentEnergy; //Energy of the primary particle
  G4int          theTrackID;        //Identification number of the primary
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

