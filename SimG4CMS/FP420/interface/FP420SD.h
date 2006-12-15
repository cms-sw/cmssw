//
#ifndef FP420_FP420SD_h
#define FP420_FP420SD_h
//

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"

#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

// last
//#include "SimG4Core/Application/interface/SimTrackManager.h"
#include "SimG4CMS/Calo/interface/CaloSD.h"


//#include "SimG4Core/Notification/interface/TrackWithHistory.h"
//#include "SimG4Core/Notification/interface/TrackContainer.h"

#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"

  
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"

//#include <CLHEP/Vector/ThreeVector.h>
//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <map>
#include <string>
 


//AZ:
class TrackingSlaveSD;
//class SimTrackManager;


//-------------------------------------------------------------------

class FP420SD : public SensitiveTkDetector,
		public Observer<const BeginOfEvent*>,
		public Observer<const EndOfEvent*> {
 public:
  
  FP420SD(std::string, const DDCompactView &,
  	  edm::ParameterSet const &,
  	  const SimTrackManager*
	  );

//-------------------------------------------------------------------
/*
class FP420SD : public CaloSD {

public:    
  FP420SD(G4String, const DDCompactView &, edm::ParameterSet const &,const SimTrackManager*);
*/
//-------------------------------------------------------------------




  virtual ~FP420SD();
  
  virtual bool ProcessHits(G4Step *,G4TouchableHistory *);
  virtual uint32_t  setDetUnitId(G4Step*);

  virtual void Initialize(G4HCofThisEvent * HCE);
  virtual void EndOfEvent(G4HCofThisEvent * eventHC);
  virtual void clear();
  virtual void DrawAll();
  virtual void PrintAll();

  virtual double getEnergyDeposit(G4Step* step);
  //protected:
  //    Collection       hits_;
    void fillHits(edm::PSimHitContainer&, std::string use);
  
  
 private:
  void           update(const BeginOfEvent *);
  void           update(const ::EndOfEvent *);
  virtual void   clearHits();
  
  //void SetNumberingScheme(FP420NumberingScheme* scheme);
  
  
  
  //  int eventno;
 private:
  
  G4ThreeVector SetToLocal(G4ThreeVector global);
  G4ThreeVector SetToLocalExit(G4ThreeVector globalPoint);
  void          GetStepInfo(G4Step* aStep);
  G4bool        HitExists();
  void          CreateNewHit();
  void          UpdateHit();
  void          StoreHit(FP420G4Hit*);
  void          ResetForNewPrimary();
  void          Summarize();
  
  
 private:
  
  //AZ:
  TrackingSlaveSD* slave;
  FP420NumberingScheme * numberingScheme;
  
  G4ThreeVector entrancePoint, exitPoint;
  G4ThreeVector theEntryPoint ;
  G4ThreeVector theExitPoint  ;
  
  float                incidentEnergy;
  G4int                primID  ; 
  
  //  G4String             name;
  std::string             name;
  G4int                    hcID;
  FP420G4HitCollection*       theHC; 
  const SimTrackManager*      theManager;
 
  G4int                    tsID; 
  FP420G4Hit*               currentHit;
  G4Track*                 theTrack;
  G4VPhysicalVolume*         currentPV;
  // unsigned int         unitID, previousUnitID;
  uint32_t             unitID, previousUnitID;
  G4int                primaryID, tSliceID;  
  G4double             tSlice;
  
  G4StepPoint*         preStepPoint; 
  G4StepPoint*         postStepPoint; 
  float                edeposit;
  
  G4ThreeVector        hitPoint;
  //  G4ThreeVector    Position;
  G4ThreeVector        hitPointExit;
  G4ThreeVector        hitPointLocal;
  G4ThreeVector        hitPointLocalExit;
  float Pabs;
  float Tof;
  float Eloss;	
  short ParticleType; 
  
  float ThetaAtEntry;
  float PhiAtEntry;
  
  int ParentId;
  float Vx,Vy,Vz;
  float X,Y,Z;
  
  
  //
  // Hist
  //
  int eventno;
  
 protected:
  
  float                edepositEM, edepositHAD;
};

#endif // FP420SD_h




