#ifndef SimG4CMSForward_MtdSD_h
#define SimG4CMSForward_MtdSD_h

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "SimG4CMS/Forward/interface/BscG4Hit.h"
#include "SimG4CMS/Forward/interface/BscG4HitCollection.h"
#include "Geometry/MTDCommonData/interface/MTDNumberingScheme.h"
  
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"

#include <string>

class TrackingSlaveSD;
class TrackInformation;
class SimTrackManager;
class TrackingSlaveSD;
class UpdatablePSimHit;
class G4ProcessTypeEnumerator;
class G4TrackToParticleID;
class MTDBaseNumber;

//-------------------------------------------------------------------

class MtdSD : public SensitiveTkDetector,
                    public Observer<const BeginOfJob *>,
                    public Observer<const BeginOfRun *>,
                    public Observer<const BeginOfEvent*>,
                    public Observer<const EndOfEvent*> {

public:
  
  MtdSD(const std::string&, const DDCompactView &, const SensitiveDetectorCatalog &, 
	      edm::ParameterSet const &, const SimTrackManager* );

  ~MtdSD() override;
  
  bool     ProcessHits(G4Step *,G4TouchableHistory *) override;
  uint32_t setDetUnitId(const G4Step*) override;

  void     Initialize(G4HCofThisEvent * HCE) override;
  void     EndOfEvent(G4HCofThisEvent * eventHC) override;
  void     clear() override;
  void     DrawAll() override;
  void     PrintAll() override;

  double   getEnergyDeposit(const G4Step* step);
  void     fillHits(edm::PSimHitContainer&, const std::string&) override;
  void     clearHits() override;
  
private:
  void     update(const BeginOfJob *) override;
  void     update(const BeginOfRun *) override;
  void     update(const BeginOfEvent *) override;
  void     update(const ::EndOfEvent *) override;

  G4ThreeVector    SetToLocal(const G4ThreeVector& global);
  G4ThreeVector    SetToLocalExit(const G4ThreeVector& globalPoint);
  void             GetStepInfo(G4Step* aStep);
  G4bool           HitExists();
  void             CreateNewHit();
  void             UpdateHit();
  void             StoreHit(BscG4Hit*);
  void             ResetForNewPrimary();
  void             Summarize();
  std::vector<double> getDDDArray(const std::string &, const DDsvalues_type &);
  void             setNumberingScheme(MTDNumberingScheme*);
  void             getBaseNumber(const G4Step*);
  
private:
  
  TrackingSlaveSD             *slave;
  int                          type_;

  G4ThreeVector                entrancePoint, exitPoint;
  G4ThreeVector                theEntryPoint, theExitPoint;
  
  float                        incidentEnergy;
  G4int                        primID  ; 
  
  std::string                  name;
  G4int                        hcID;
  BscG4HitCollection*          theHC; 
  const SimTrackManager*       theManager;
 
  G4int                        tsID; 
  BscG4Hit*                    currentHit;
  G4Track*                     theTrack;
  G4VPhysicalVolume*           currentPV;
  uint32_t                     unitID, previousUnitID;
  G4int                        primaryID, tSliceID;  
  G4double                     tSlice;
  
  G4StepPoint*                 preStepPoint; 
  G4StepPoint*                 postStepPoint; 
  float                        edeposit;
  
  G4ThreeVector                hitPoint;
  G4ThreeVector                hitPointExit;
  G4ThreeVector                hitPointLocal;
  G4ThreeVector                hitPointLocalExit;

  float                        Pabs, Tof, Eloss;	
  short                        ParticleType; 
  float                        ThetaAtEntry, PhiAtEntry;
  
  int                          ParentId;
  float                        Vx,Vy,Vz;
  float                        X,Y,Z;
  
  int                          eventno;

  MTDNumberingScheme *         numberingScheme;
  MTDBaseNumber                theBaseNumber;
  bool                         isBTL;
  bool                         isETL;
  
protected:
  
  float                        edepositEM, edepositHAD;
};

#endif




