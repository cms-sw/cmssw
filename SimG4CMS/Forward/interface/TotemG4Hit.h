#ifndef Forward_TotemG4Hit_h
#define Forward_TotemG4Hit_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemG4Hit
//
/**\class TotemG4Hit TotemG4Hit.h SimG4CMS/Forward/interface/TotemG4Hit.h
 
 Description: Transient Hit class for Totem taken from those for Calorimeters
 
 Usage: One Hit object should be created
   -for each new particle entering the calorimeter
   -for each detector unit (= cristal or fiber or scintillator layer)
   -for each nanosecond of the shower development

   This implies that all hit objects created for a given shower
   have the same value for
   - Entry (= local coordinates of the entrance point of the particle
              in the unit where the shower starts) 
   - the TrackID (= Identification number of the incident particle)
   - the IncidentEnergy (= energy of that particle)
 
*/
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemG4Hit.h,v 1.2 2007/11/20 12:37:19 fabiocos Exp $
//
 
// system include files

// user include files

#include "G4VHit.hh"
#include "DataFormats/Math/interface/Point3D.h"
#include <boost/cstdint.hpp>
#include <iostream>

class TotemG4Hit : public G4VHit {
  
public:

  // ---------- Constructor and destructor -----------------
  TotemG4Hit();
  ~TotemG4Hit();
  TotemG4Hit(const TotemG4Hit &right);

  // ---------- operators ----------------------------------
  const TotemG4Hit& operator=(const TotemG4Hit &right);
  int operator==(const TotemG4Hit &){return 0;}

  // ---------- member functions ---------------------------
  void         Draw(){}
  void         Print();

  math::XYZPoint   getEntry() const;
  void         setEntry(double x, double y, double z)      {entry.SetCoordinates(x,y,z);}
  
  double       getEM() const;
  void         setEM (double e);
  
  double       getHadr() const;
  void         setHadr (double e);
  
  double       getIncidentEnergy() const;
  void         setIncidentEnergy (double e);
  
  int          getTrackID() const;
  void         setTrackID (int i);
  
  uint32_t     getUnitID() const;
  void         setUnitID (uint32_t i);
  
  double       getTimeSlice() const;     
  void         setTimeSlice(double d);
  int          getTimeSliceID() const;     
  
  void         addEnergyDeposit(double em, double hd);
  void         addEnergyDeposit(const TotemG4Hit& aHit);
  
  double       getEnergyDeposit() const;
  
  float        getPabs() const;
  float        getTof() const;
  float        getEnergyLoss() const;
  int          getParticleType() const;

  void         setPabs(float e);
  void         setTof(float e);
  void         setEnergyLoss(float e) ;
  void         setParticleType(short i) ;

  float        getThetaAtEntry() const;   
  float        getPhiAtEntry() const;

  void         setThetaAtEntry(float t);
  void         setPhiAtEntry(float f) ;

  float        getX() const;
  float        getY() const;
  float        getZ() const;
  void         setX(float t);
  void         setY(float t);
  void         setZ(float t);

  int          getParentId() const;
  float        getVx() const;
  float        getVy() const;
  float        getVz() const;

  void         setParentId(int p);
  void         setVx(float p);
  void         setVy(float p);
  void         setVz(float p);

private:
  
  math::XYZPoint   entry;             //Entry point
  double       elem;              //EnergyDeposit of EM particles
  double       hadr;              //EnergyDeposit of HD particles
  double       theIncidentEnergy; //Energy of the primary particle
  int          theTrackID;        //Identification number of the primary
                                  //particle
  uint32_t     theUnitID;         //Totem Unit Number
  double       theTimeSlice;      //Time Slice Identification


  float        theX;
  float        theY;
  float        theZ;
  float        thePabs;
  float        theTof;
  float        theEnergyLoss;
  int          theParticleType;

  float        theThetaAtEntry;
  float        thePhiAtEntry;
  math::XYZPoint   theEntryPoint;
  math::XYZPoint   theExitPoint;

  int          theParentId;
  float        theVx;
  float        theVy;
  float        theVz;

};

std::ostream& operator<<(std::ostream&, const TotemG4Hit&);

#endif

