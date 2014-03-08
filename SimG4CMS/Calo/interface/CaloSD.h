#ifndef SimG4CMS_CaloSD_h
#define SimG4CMS_CaloSD_h
///////////////////////////////////////////////////////////////////////////////
// File: CaloSD.h
// Description: Stores hits of calorimetric type in appropriate container
// Use in your sensitive detector builder:
//    CaloSD* caloSD = new CaloSD(SDname, new CaloNumberingScheme());
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "SimG4CMS/Calo/interface/CaloMeanResponse.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// To be replaced by something else 
/* #include "Utilities/Notification/interface/TimerProxy.h" */
 
#include "G4VPhysicalVolume.hh"
#include "G4Track.hh"
#include "G4VGFlashSensitiveDetector.hh"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

class G4Step;
class G4HCofThisEvent;
class CaloSlaveSD;
class G4GFlashSpot;

class CaloSD : public SensitiveCaloDetector, 
               public G4VGFlashSensitiveDetector,
               public Observer<const BeginOfRun *>,    
               public Observer<const BeginOfEvent *>,
               public Observer<const BeginOfTrack *>,
               public Observer<const EndOfTrack *>,
               public Observer<const EndOfEvent *> {

public:    
  
  CaloSD(G4String  aSDname, const DDCompactView & cpv,
         SensitiveDetectorCatalog & clg, 
         edm::ParameterSet const & p, const SimTrackManager*,
	 double timeSlice=1.,bool ignoreTkID=false);
  virtual ~CaloSD();
  virtual bool     ProcessHits(G4Step * step,G4TouchableHistory * tHistory);
  virtual bool     ProcessHits(G4GFlashSpot*aSpot,G4TouchableHistory*);
  virtual double   getEnergyDeposit(G4Step* step); 
  virtual uint32_t setDetUnitId(G4Step* step)=0;
  
  virtual void     Initialize(G4HCofThisEvent * HCE);
  virtual void     EndOfEvent(G4HCofThisEvent * eventHC);
  virtual void     clear();
  virtual void     DrawAll();
  virtual void     PrintAll();

  void             fillHits(edm::PCaloHitContainer&,std::string n);

protected:

  virtual G4bool   getStepInfo(G4Step* aStep);
  G4ThreeVector    setToLocal(G4ThreeVector, const G4VTouchable*);
  G4ThreeVector    setToGlobal(G4ThreeVector, const G4VTouchable*);
  G4bool           hitExists();
  G4bool           checkHit();
  CaloG4Hit*       createNewHit();
  void             updateHit(CaloG4Hit*);
  void             resetForNewPrimary(G4ThreeVector, double);
  double           getAttenuation(G4Step* aStep, double birk1, double birk2,
                                  double birk3);

  virtual void     update(const BeginOfRun *);
  virtual void     update(const BeginOfEvent *);
  virtual void     update(const BeginOfTrack * trk);
  virtual void     update(const EndOfTrack * trk);
  virtual void     update(const ::EndOfEvent *);
  virtual void     clearHits();
  virtual void     initRun();
  virtual bool     filterHit(CaloG4Hit*, double);

  virtual int      getTrackID(G4Track*);
  virtual uint16_t getDepth(G4Step*);   
  double           getResponseWt(G4Track*);
  int              getNumberOfHits();

private:

  void             storeHit(CaloG4Hit*);
  bool             saveHit(CaloG4Hit*);
  void             summarize();
  void             cleanHitCollection();

protected:
  
  // Data relative to primary particle (the one which triggers a shower)
  // These data are common to all Hits of a given shower.
  // One shower is made of several hits which differ by the
  // unit ID (crystal/fibre/scintillator) and the Time slice ID.

  G4ThreeVector                   entrancePoint;
  G4ThreeVector                   entranceLocal;
  G4ThreeVector                   posGlobal;
  float                           incidentEnergy;
  int                             primIDSaved; //  ID of the last saved primary

  CaloHitID                       currentID, previousID; 
  G4Track*                        theTrack;

  G4StepPoint*                    preStepPoint; 
  float                           edepositEM, edepositHAD;

  double                          energyCut, tmaxHit, eminHit, eminHitD;
  int                             checkHits;
  bool                            useMap;

  const SimTrackManager*          m_trackManager;
  CaloG4Hit*                      currentHit;
//  TimerProxy                    theHitTimer;
  bool                            runInit;

  bool                            corrTOFBeam, suppressHeavy;
  double                          correctT;
  double                          kmaxIon, kmaxNeutron, kmaxProton;

  G4int                           emPDG, epPDG, gammaPDG;
  bool                            forceSave;

private:

  double                          timeSlice;
  bool                            ignoreTrackID;
  CaloSlaveSD*                    slave;
  int                             hcID;
  CaloG4HitCollection*            theHC; 
  std::map<CaloHitID,CaloG4Hit*>  hitMap;

  std::map<int,TrackWithHistory*> tkMap;
  CaloMeanResponse*               meanResponse;

  int                             primAncestor;
  int                             cleanIndex;
  std::vector<CaloG4Hit*>         reusehit;
  std::vector<CaloG4Hit*>         hitvec;
  std::vector<unsigned int>       selIndex;
  int                             totalHits;

};

#endif // SimG4CMS_CaloSD_h
