#ifndef SimG4CMS_CaloGVHit_h
#define SimG4CMS_CaloGVHit_h 1
///////////////////////////////////////////////////////////////////////////////
// File: CaloGVHit.h
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

class CaloGVHit {
  
public:
  
  CaloGVHit();
  ~CaloGVHit();
  CaloGVHit(const CaloGVHit &right);
  const CaloGVHit& operator=(const CaloGVHit &right);
  bool operator==(const CaloGVHit &){return false;}
  
public:
  
  double         getEM() const                 {return elem;}
  void           setEM (double e)              {elem = e;}
  
  double         getHadr() const               {return hadr;}
  void           setHadr (double e)            {hadr = e;}
  
  int            getTrackID() const            {return hitID.trackID();}
  uint32_t       getUnitID() const             {return hitID.unitID();}
  double         getTimeSlice() const          {return hitID.timeSlice();}
  int            getTimeSliceID() const        {return hitID.timeSliceID();}
  uint16_t       getDepth() const              {return hitID.depth();}

  CaloHitID      getID() const                 {return hitID;}
  void           setID (uint32_t i, double d, int j, uint16_t k=0) {
    hitID.setID(i,d,j,k);}
  void           setID (const CaloHitID& id)   {hitID = id;}
  
  void           addEnergyDeposit(double em, double hd);
  void           addEnergyDeposit(const CaloGVHit& aHit);
  
  double         getEnergyDeposit() const      {return (elem+hadr);}
  
private:
  
  double           elem;              //EnergyDeposit of EM particles
  double           hadr;              //EnergyDeposit of HD particles
  CaloHitID        hitID;             //Identification number of the hit given 
                                      //by primary particle, Cell ID, Time of 
                                      //the hit
};

class CaloGVHitLess {
public:
  bool operator()( const CaloGVHit* a, const CaloGVHit* b) {
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

class CaloGVHitEqual {
public:
  bool operator()( const CaloGVHit* a, const CaloGVHit* b) {
    return (a->getTrackID()     == b->getTrackID() && 
	    a->getUnitID()      == b->getUnitID()  && 
	    a->getDepth()       == b->getDepth()   &&
	    a->getTimeSliceID() == b->getTimeSliceID());
  }
};

std::ostream& operator<<(std::ostream&, const CaloGVHit&);

#endif
