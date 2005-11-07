///////////////////////////////////////////////////////////////////////////////
// File: CaloSD.h
// Description: Stores hits of calorimetric type in appropriate container
// Use in your sensitive detector builder:
//    CaloSD* caloSD = new CaloSD(SDname, new CaloNumberingScheme());
///////////////////////////////////////////////////////////////////////////////
#ifndef CaloSD_h
#define CaloSD_h

#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"


// To be replaced by something else 
/* #include "Utilities/Notification/interface/TimerProxy.h" */
/* #include "Utilities/UI/interface/Verbosity.h" */
 
#include "G4VPhysicalVolume.hh"
#include "G4Track.hh"
#ifdef G4v7
#include "G4VGFlashSensitiveDetector.hh"
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using namespace std;
class G4Step;
class G4HCofThisEvent;
class EventAction;
class CaloNumberingScheme;
class CaloSlaveSD;
#ifdef G4v7
class G4GFlashSpot;
#endif

class CaloSD : 
#ifdef G4v7
public G4VGFlashSensitiveDetector,
#endif
public SensitiveCaloDetector
	       /* 	       private Observer<const BeginOfEvent *>, */
	       /* 	       private Observer<const EndOfEvent *>, */
	       /*                public LazyObserver<const EventAction*> { */
{
 public:    
  typedef ::EndOfEvent MyEndOfEvent;
  typedef map<vector<int>,CaloG4Hit*> MyMap;
  
  CaloSD(G4String aSDname, const DDCompactView & cpv,
	 edm::ParameterSet const & p);
  virtual ~CaloSD();
  virtual bool ProcessHits(G4Step * step,G4TouchableHistory * tHistory);
#ifdef G4v7
  virtual bool ProcessHits(G4GFlashSpot*aSpot,G4TouchableHistory*);
#endif
  virtual double getEnergyDeposit(G4Step* step); 
  virtual int    SetDetUnitId(G4Step* step);
  
  virtual void   Initialize(G4HCofThisEvent * HCE);
  //  void           lazyUpDate(const EventAction *);
  virtual void   EndOfEvent(G4HCofThisEvent * eventHC);
  virtual void   clear();
  virtual void   DrawAll();
  virtual void   PrintAll();

  void setNumberingScheme(CaloNumberingScheme* scheme);
  void fillHits(edm::PCaloHitContainer&,std::string n);

protected:

  void          getStepInfo(G4Step* aStep);
  G4ThreeVector setToLocal(G4ThreeVector, const G4VTouchable*);
  G4bool        hitExists();
  G4bool        checkHit();
  void          createNewHit();
  void          upDateHit();
  void          resetForNewPrimary(G4ThreeVector, double);
  double        getAttenuation(G4Step* aStep, double birk1, double birk2);

private:

  void          upDate(const BeginOfEvent *);
  void          upDate(const MyEndOfEvent * );
  virtual void  clearHits();

  void          storeHit(CaloG4Hit*);
  bool          saveHit(CaloG4Hit*);
  void          summarize();

protected:
  
  // Data relative to primary particle (the one which triggers a shower)
  // These data are common to all Hits of a given shower.
  // One shower is made of several hits which differ by the
  // unit ID (crystal/fibre/scintillator) and the Time slice ID.

  G4ThreeVector        entrancePoint;
  G4ThreeVector        entranceLocal;
  float                incidentEnergy;
  int                  primIDSaved; //   ID of the last saved primary

  CaloHitID            currentID, previousID; 
  G4Track*             theTrack;

  G4StepPoint*         preStepPoint; 
  float                edepositEM, edepositHAD;

  double               energyCut;
  int                  checkHits;
  bool                 useMap;

//  TimerProxy           theHitTimer;

private:

  CaloSlaveSD*          slave;
  CaloNumberingScheme * numberingScheme;

  G4String             name;
  int                  hcID;
  CaloG4HitCollection* theHC; 
  map<CaloHitID,CaloG4Hit*> hitMap;

  CaloG4Hit*           currentHit;
  vector<CaloG4Hit*>   hitvec;
//  const EventAction*   eventAction;

//  static UserVerbosity cout;

};

#endif // CaloSD_h
