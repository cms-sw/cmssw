#ifndef SimG4CMS_CaloG4Hit_h
#define SimG4CMS_CaloG4Hit_h 1
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

#include "SimG4CMS/Calo/interface/CaloHitID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/Point3D.h"
#include <iostream>

#include "G4Allocator.hh"
#include "G4VHit.hh"

class CaloG4Hit : public G4VHit {
  
public:
  
  CaloG4Hit();
  virtual ~CaloG4Hit();
  CaloG4Hit(const CaloG4Hit &right);
  const CaloG4Hit& operator=(const CaloG4Hit &right);
  bool operator==(const CaloG4Hit &){return false;}
  inline void * operator new(size_t);
  inline void   operator delete(void * CaloG4Hit);

  void         Draw(){}
  void         Print();
  
public:
  
  math::XYZPoint getEntry() const              {return entry;}
  void           setEntry(double x, double y, double z)      {entry.SetCoordinates(x,y,z);}
  
  math::XYZPoint getEntryLocal() const         {return entryLocal;}
  void           setEntryLocal(double x, double y, double z) {entryLocal.SetCoordinates(x,y,z);}
  
  math::XYZPoint getPosition() const           {return pos;}
  void           setPosition(double x, double y, double z)   {pos.SetCoordinates(x,y,z);}
  
  double         getEM() const                 {return elem;}
  void           setEM (double e)              {elem = e;}
  
  double         getHadr() const               {return hadr;}
  void           setHadr (double e)            {hadr = e;}
  
  double         getIncidentEnergy() const     {return theIncidentEnergy;}
  void           setIncidentEnergy (double e)  {theIncidentEnergy = e;}
  
  int            getTrackID() const            {return hitID.trackID();}
  uint32_t       getUnitID() const             {return hitID.unitID();}
  double         getTimeSlice() const          {return hitID.timeSlice();}
  int            getTimeSliceID() const        {return hitID.timeSliceID();}
  uint16_t       getDepth() const              {return hitID.depth();}

  CaloHitID      getID() const                 {return hitID;}
  void           setID (uint32_t i, double d, int j, uint16_t k=0) { hitID.setID(i,d,j,k);}
  void           setID (const CaloHitID& id)          {hitID = id;}
  
  void           addEnergyDeposit(double em, double hd);
  void           addEnergyDeposit(const CaloG4Hit& aHit);
  
  double         getEnergyDeposit() const      {return (elem+hadr);}
  
private:
  
  math::XYZPoint   entry;             //Entry point (Global coordinate)
  math::XYZPoint   entryLocal;        //Entry point (Local  coordinate)
  math::XYZPoint   pos;               //Position    (Global coordinate)
  double           elem;              //EnergyDeposit of EM particles
  double           hadr;              //EnergyDeposit of HD particles
  double           theIncidentEnergy; //Energy of the primary particle
  CaloHitID        hitID;             //Identification number of the hit given 
                                      //by primary particle, Cell ID, Time of 
                                      //the hit
};

class CaloG4HitLess {
public:
  bool operator()( const CaloG4Hit* a, const CaloG4Hit* b) {
    if (a->getTrackID() != b->getTrackID()) {
      return (a->getTrackID() < b->getTrackID());
    } else if (a->getUnitID() != b->getUnitID()) {
      return (a->getUnitID() < b->getUnitID());
    } else if (a->getDepth() != b->getDepth()) {
      return (a->getDepth() < b->getDepth());
    } else {
      return (a->getTimeSliceID() < b->getTimeSliceID());
    }
  }
};

class CaloG4HitEqual {
public:
  bool operator()( const CaloG4Hit* a, const CaloG4Hit* b) {
    return (a->getTrackID()     == b->getTrackID() && 
	    a->getUnitID()      == b->getUnitID()  && 
	    a->getDepth()       == b->getDepth()   &&
	    a->getTimeSliceID() == b->getTimeSliceID());
  }
};

extern G4ThreadLocal G4Allocator<CaloG4Hit> *fpCaloG4HitAllocator;

inline void * CaloG4Hit::operator new(size_t) {
  if (!fpCaloG4HitAllocator) fpCaloG4HitAllocator = 
    new G4Allocator<CaloG4Hit>;
  return (void*)fpCaloG4HitAllocator->MallocSingle();
}

inline void CaloG4Hit::operator delete(void * aHit) {  
  fpCaloG4HitAllocator->FreeSingle((CaloG4Hit*) aHit); 
}

std::ostream& operator<<(std::ostream&, const CaloG4Hit&);

#endif
