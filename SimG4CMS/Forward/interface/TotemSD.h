#ifndef Forward_TotemSD_h
#define Forward_TotemSD_h
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemSD
//
/**\class TotemSD TotemSD.h SimG4CMS/Forward/interface/TotemSD.h
 
 Description: Stores hits of Totem in appropriate  container
 
 Usage:
    Used in sensitive detector builder 
 
*/
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemSD.h,v 1.2 2007/05/08 21:27:29 sunanda Exp $
//
 
// system include files

// user include files

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "SimG4CMS/Forward/interface/TotemG4Hit.h"
#include "SimG4CMS/Forward/interface/TotemG4HitCollection.h"
#include "SimG4CMS/Forward/interface/TotemVDetectorOrganization.h"
 
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
 
#include <string>

class TrackingSlaveSD;

class TotemSD : public SensitiveTkDetector,
		public Observer<const BeginOfEvent*>,
		public Observer<const EndOfEvent*> {

public:

  TotemSD(std::string, const DDCompactView &, SensitiveDetectorCatalog &, 
	  edm::ParameterSet const &, const SimTrackManager*);
  virtual ~TotemSD();

  virtual bool   ProcessHits(G4Step *,G4TouchableHistory *);
  virtual uint32_t setDetUnitId(G4Step*);

  virtual void   Initialize(G4HCofThisEvent * HCE);
  virtual void   EndOfEvent(G4HCofThisEvent * eventHC);
  virtual void   clear();
  virtual void   DrawAll();
  virtual void   PrintAll();

  void fillHits(edm::PSimHitContainer&, std::string use);

private:

  void           update(const BeginOfEvent *);
  void           update(const ::EndOfEvent *);
  virtual void   clearHits();

private:

  G4ThreeVector  SetToLocal(G4ThreeVector globalPoint);
  void           GetStepInfo(G4Step* aStep);
  bool           HitExists();
  void           CreateNewHit();
  void           CreateNewHitEvo();
  G4ThreeVector  PosizioEvo(G4ThreeVector,double ,double ,double, double,int&);
  void           UpdateHit();
  void           StoreHit(TotemG4Hit*);
  void           ResetForNewPrimary();
  void           Summarize();

private:

  TrackingSlaveSD*            slave;
  TotemVDetectorOrganization* numberingScheme;

  // Data relative to primary particle (the one which triggers a shower)
  // These data are common to all Hits of a given shower.
  // One shower is made of several hits which differ by the
  // unit ID (cristal/fiber/scintillator) and the Time slice ID.

  G4ThreeVector               entrancePoint;
  float                       incidentEnergy;
  G4int                       primID  ; //@@ ID of the primary particle.

  std::string                 name;
  G4int                       hcID;
  TotemG4HitCollection*       theHC; 
  const SimTrackManager*      theManager;

  int                         tsID; 
  TotemG4Hit*                 currentHit;
  G4Track*                    theTrack;
  G4VPhysicalVolume*          currentPV;
  uint32_t                    unitID, previousUnitID;
  int                         primaryID, tSliceID;  
  double                      tSlice;

  G4StepPoint*                preStepPoint; 
  G4StepPoint*                postStepPoint; 
  float                       edeposit;
  G4ThreeVector               hitPoint;

  G4ThreeVector               Posizio;
  float                       Pabs;
  float                       Tof;
  float                       Eloss;	
  short                       ParticleType; 

  float                       ThetaAtEntry;
  float                       PhiAtEntry;

  int                         ParentId;
  float                       Vx,Vy,Vz;

  int                         eventno;
};

#endif
