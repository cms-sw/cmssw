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
class SimTrackManager;

class TotemSD : public SensitiveTkDetector,
		public Observer<const BeginOfEvent*>{

public:

  TotemSD(const std::string&, const DDCompactView &, const SensitiveDetectorCatalog &,
	  edm::ParameterSet const &, const SimTrackManager*);
  ~TotemSD() override;

  bool   ProcessHits(G4Step *,G4TouchableHistory *) override;
  uint32_t setDetUnitId(const G4Step*) override;

  void   Initialize(G4HCofThisEvent * HCE) override;
  void   EndOfEvent(G4HCofThisEvent * eventHC) override;
  void   PrintAll() override;

  void   fillHits(edm::PSimHitContainer&, const std::string&) override;
  void   clearHits() override;

protected:

  void   update(const BeginOfEvent *) override;

private:

  G4ThreeVector  SetToLocal(const G4ThreeVector& globalPoint);
  void           GetStepInfo(const G4Step* aStep);
  bool           HitExists();
  void           CreateNewHit();
  void           CreateNewHitEvo();
  G4ThreeVector  PosizioEvo(const G4ThreeVector&,double ,double ,double, double,int&);
  void           UpdateHit();
  void           StoreHit(TotemG4Hit*);
  void           ResetForNewPrimary();

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

  const G4StepPoint*          preStepPoint; 
  const G4StepPoint*          postStepPoint; 
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
};

#endif
