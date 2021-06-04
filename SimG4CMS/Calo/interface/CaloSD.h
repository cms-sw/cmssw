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

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "G4VPhysicalVolume.hh"
#include "G4Track.hh"
#include "G4VGFlashSensitiveDetector.hh"

#include <vector>
#include <map>
#include <memory>

class G4Step;
class G4HCofThisEvent;
class CaloSlaveSD;
class G4GFlashSpot;
class SimTrackManager;

class CaloSD : public SensitiveCaloDetector,
               public G4VGFlashSensitiveDetector,
               public Observer<const BeginOfRun*>,
               public Observer<const BeginOfEvent*>,
               public Observer<const BeginOfTrack*>,
               public Observer<const EndOfTrack*>,
               public Observer<const EndOfEvent*> {
public:
  CaloSD(const std::string& aSDname,
         const SensitiveDetectorCatalog& clg,
         edm::ParameterSet const& p,
         const SimTrackManager*,
         float timeSlice = 1.,
         bool ignoreTkID = false);
  ~CaloSD() override;

  G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override;
  bool ProcessHits(G4GFlashSpot* aSpot, G4TouchableHistory*) override;

  uint32_t setDetUnitId(const G4Step* step) override = 0;

  void Initialize(G4HCofThisEvent* HCE) override;
  void EndOfEvent(G4HCofThisEvent* eventHC) override;
  void clear() override;
  void DrawAll() override;
  void PrintAll() override;

  void clearHits() override;
  void fillHits(edm::PCaloHitContainer&, const std::string&) override;
  void reset() override;

  bool isItFineCalo(const G4VTouchable* touch);

protected:
  virtual double getEnergyDeposit(const G4Step* step);
  virtual double EnergyCorrected(const G4Step& step, const G4Track*);
  virtual bool getFromLibrary(const G4Step* step);

  G4ThreeVector setToLocal(const G4ThreeVector&, const G4VTouchable*) const;
  G4ThreeVector setToGlobal(const G4ThreeVector&, const G4VTouchable*) const;

  bool hitExists(const G4Step*);
  bool checkHit();
  CaloG4Hit* createNewHit(const G4Step*, const G4Track*);
  void updateHit(CaloG4Hit*);
  void resetForNewPrimary(const G4Step*);
  double getAttenuation(const G4Step* aStep, double birk1, double birk2, double birk3) const;

  static std::string printableDecayChain(const std::vector<unsigned int>& decayChain);
  void hitBookkeepingFineCalo(const G4Step* step, const G4Track* currentTrack, CaloG4Hit* hit);

  void update(const BeginOfRun*) override;
  void update(const BeginOfEvent*) override;
  void update(const BeginOfTrack* trk) override;
  void update(const EndOfTrack* trk) override;
  void update(const ::EndOfEvent*) override;
  virtual void initRun();
  virtual void initEvent(const BeginOfEvent*);
  virtual void endEvent();
  virtual bool filterHit(CaloG4Hit*, double);

  virtual int getTrackID(const G4Track*);
  virtual int setTrackID(const G4Step*);
  virtual uint16_t getDepth(const G4Step*);
  double getResponseWt(const G4Track*);
  int getNumberOfHits();
  void ignoreRejection() { ignoreReject = true; }

  inline void setParameterized(bool val) { isParameterized = val; }
  inline void setUseMap(bool val) { useMap = val; }

  inline void processHit(const G4Step* step) {
    // check if it is in the same unit and timeslice as the previous one
    if (currentID == previousID) {
      updateHit(currentHit);
    } else if (!checkHit()) {
      currentHit = createNewHit(step, step->GetTrack());
    }
  }

  inline void setNumberCheckedHits(int val) { nCheckedHits = val; }
  void printDetectorLevels(const G4VTouchable*) const;

private:
  void storeHit(CaloG4Hit*);
  bool saveHit(CaloG4Hit*);
  void cleanHitCollection();

protected:
  // Data relative to primary particle (the one which triggers a shower)
  // These data are common to all Hits of a given shower.
  // One shower is made of several hits which differ by the
  // unit ID (crystal/fibre/scintillator) and the Time slice ID.

  G4ThreeVector entrancePoint;
  G4ThreeVector entranceLocal;
  G4ThreeVector posGlobal;
  float incidentEnergy;
  float edepositEM, edepositHAD;

  CaloHitID currentID, previousID;

  double energyCut, tmaxHit, eminHit;

  CaloG4Hit* currentHit;

  bool suppressHeavy;
  double kmaxIon, kmaxNeutron, kmaxProton;

  bool forceSave;

private:
  struct Detector {
    Detector() {}
    std::string name;
    G4LogicalVolume* lv;
    int level;
  };

  const SimTrackManager* m_trackManager;

  std::unique_ptr<CaloSlaveSD> slave;
  std::unique_ptr<CaloMeanResponse> meanResponse;

  CaloG4HitCollection* theHC;

  bool ignoreTrackID;
  bool isParameterized;
  bool ignoreReject;
  bool useMap;  // use map for comparison of ID
  bool corrTOFBeam;

  int hcID;
  int primAncestor;
  int cleanIndex;
  int totalHits;
  int primIDSaved;   // ID of the last saved primary
  int nCheckedHits;  // number of last hits to compare ID

  float timeSlice;
  double eminHitD;
  double correctT;
  bool doFineCalo_;
  double eMinFine_;

  std::map<CaloHitID, CaloG4Hit*> hitMap;
  std::map<int, TrackWithHistory*> tkMap;
  std::vector<std::unique_ptr<CaloG4Hit>> reusehit;
  std::vector<Detector> fineDetectors_;
};

#endif  // SimG4CMS_CaloSD_h
