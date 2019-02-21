///////////////////////////////////////////////////////////////////////////////
//Author: Seyed Mohsen Etesami
// setesami@cern.ch
// 2016 Nov
///////////////////////////////////////////////////////////////////////////////

#ifndef PPS_CTPPSDiamondSD_h
#define PPS_CTPPSDiamondSD_h


#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4CMS/PPS/interface/CTPPS_Diamond_G4Hit.h"
#include "SimG4CMS/PPS/interface/CTPPS_Diamond_G4HitCollection.h"
#include "SimG4CMS/PPS/interface/PPSVDetectorOrganization.h"
#include "SimG4CMS/PPS/interface/CTPPSDiamondNumberingScheme.h"
#include "SimG4CMS/PPS/interface/CTPPSDiamondOrganization.h"


#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
 
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
 
#include <string>


class G4Step;
class G4HCofThisEvent;
class TrackingSlaveSD;


class CTPPS_Diamond_SD : public SensitiveTkDetector,
    public Observer<const BeginOfEvent*>,
    public Observer<const EndOfEvent*>
{
  public:    
    CTPPS_Diamond_SD(const std::string &, const DDCompactView &, const SensitiveDetectorCatalog &,
    edm::ParameterSet const &,const SimTrackManager*);
    ~CTPPS_Diamond_SD() override;
    void Print_Hit_Info();
 
    void Initialize(G4HCofThisEvent * HCE) override;
    void EndOfEvent(G4HCofThisEvent * eventHC) override;
    void clear() override;
    void cleartrack();
    void clearTrack(G4Track * Track);
    void DrawAll() override;
    void PrintAll() override; 
    void fillHits(edm::PSimHitContainer&, const std::string &) override;
 
  private:
    void clearHits() override;
    bool ProcessHits(G4Step * step, G4TouchableHistory * tHistory) override;
    uint32_t setDetUnitId(const G4Step * step) override;
    void update(const BeginOfEvent *) override;
    void update (const ::EndOfEvent*) override;
    void SetNumberingScheme(PPSVDetectorOrganization* scheme);

    TrackingSlaveSD* slave;
    PPSVDetectorOrganization * numberingScheme;
 
    int verbosity_;
    int theMPDebug_;

    G4ThreeVector SetToLocal(G4ThreeVector globalPoint);
    void GetStepInfo(G4Step* aStep);
    G4bool HitExists();
    void ImportInfotoHit(); //added pps
    void UpdateHit();
    void StoreHit(CTPPS_Diamond_G4Hit*);
    void ResetForNewPrimary();
    void Summarize();
    bool IsPrimary(const G4Track * track);

    G4ThreeVector entrancePoint;
    double incidentEnergy;
    G4String name;
    G4int hcID;
    CTPPS_Diamond_G4HitCollection* theHC; 
    CTPPS_Diamond_G4Hit* currentHit;
    G4Track* theTrack;
    G4VPhysicalVolume* currentPV;
    unsigned int unitID;
    G4int primaryID, tSliceID;  
    G4double tSlice;
 
    G4StepPoint* preStepPoint; 
    G4StepPoint* postStepPoint;
    G4ThreeVector hitPoint;
    G4ThreeVector exitPoint;
    G4ThreeVector theLocalEntryPoint;
    G4ThreeVector theLocalExitPoint;
    double Pabs;
    double p_x, p_y, p_z;
    double Tof;
    double Eloss;	
    short ParticleType; 
    double ThetaAtEntry;
    double PhiAtEntry;
    int ParentId;
    double Vx,Vy,Vz;
    double Globaltimehit;
    double theglobaltimehit;
    int eventno;
};

#endif // CTPPS_CTPPSDiamondSD_h
