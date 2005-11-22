///////////////////////////////////////////////////////////////////////////////
// File: CaloG4Hit.h
// Date: 10.02 Taken from CMSCaloHit
// 
// Hit class for Calorimeters (Ecal, Hcal, ...)
//
// One Hit object should be created
// -for each new particle entering the calorimeter
// -for each detector unit (= crystal or fibre or scintillator layer)
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
#ifndef CaloG4Hit_h
#define CaloG4Hit_h 1

#include "SimG4CMS/Calo/interface/CaloHitID.h"
#include "G4VHit.hh"
#include <CLHEP/Vector/ThreeVector.h>
#include <iostream>

class CaloG4Hit : public G4VHit {
  
public:
  
  CaloG4Hit();
  ~CaloG4Hit();
  CaloG4Hit(const CaloG4Hit &right);
  const CaloG4Hit& operator=(const CaloG4Hit &right);
  bool operator==(const CaloG4Hit &){return false;}
  
  void         Draw(){}
  void         Print();
  
public:
  
  Hep3Vector   getEntry() const              {return entry;}
  void         setEntry(Hep3Vector xyz)      {entry = xyz;}
  
  Hep3Vector   getEntryLocal() const         {return entryLocal;}
  void         setEntryLocal(Hep3Vector xyz) {entryLocal = xyz;}
  
  Hep3Vector   getPosition() const           {return pos;}
  void         setPosition(Hep3Vector xyz)   {pos = xyz;}
  
  double       getEM() const                 {return elem;}
  void         setEM (double e)              {elem = e;}
  
  double       getHadr() const               {return hadr;}
  void         setHadr (double e)            {hadr = e;}
  
  double       getIncidentEnergy() const     {return theIncidentEnergy;}
  void         setIncidentEnergy (double e)  {theIncidentEnergy = e;}
  
  int          getTrackID() const            {return hitID.trackID();}
  uint32_t     getUnitID() const             {return hitID.unitID();}
  double       getTimeSlice() const          {return hitID.timeSlice();}
  int          getTimeSliceID() const        {return hitID.timeSliceID();}

  CaloHitID    getID() const                 {return hitID;}
  void         setID (uint32_t i, double d, int j) {hitID.setID(i,d,j);}
  void         setID (CaloHitID id)          {hitID = id;}
  
  void         addEnergyDeposit(double em, double hd);
  void         addEnergyDeposit(const CaloG4Hit& aHit);
  
  double       getEnergyDeposit() const      {return (elem+hadr);}
  
private:
  
  Hep3Vector   entry;             //Entry point (Global coordinate)
  Hep3Vector   entryLocal;        //Entry point (Local  coordinate)
  Hep3Vector   pos;               //Position    (Global coordinate)
  double       elem;              //EnergyDeposit of EM particles
  double       hadr;              //EnergyDeposit of HD particles
  double       theIncidentEnergy; //Energy of the primary particle
  CaloHitID    hitID;             //Identification number of the hit given by
                                  //primary particle, Cell ID, Time of the hit
};

class CaloG4HitLess {
public:
  bool operator()( const CaloG4Hit* a, const CaloG4Hit* b) {
    if (a->getTrackID() < b->getTrackID()) {
      return true;
    } else if (a->getTrackID() > b->getTrackID()) {
      return false;
    } else if (a->getUnitID() < b->getUnitID()) {
      return true;
    } else if (a->getUnitID() > b->getUnitID()) {
      return false;
    } else {
      return (a->getTimeSliceID() < b->getTimeSliceID());
    }
  }
};

class CaloG4HitEqual {
public:
  bool operator()( const CaloG4Hit* a, const CaloG4Hit* b) {
    return (a->getTrackID()     == b->getTrackID() && 
	    a->getUnitID()      == b->getUnitID() && 
	    a->getTimeSliceID() == b->getTimeSliceID());
  }
};

std::ostream& operator<<(std::ostream&, const CaloG4Hit&);

#endif
