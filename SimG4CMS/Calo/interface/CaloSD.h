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
#include <unordered_map>
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

  void newCollection(const std::string& name, edm::ParameterSet const& p);
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

  void Initialize(G4HCofThisEvent* HCE, int k);
  bool isItFineCalo(const G4VTouchable* touch);

protected:
  virtual double getEnergyDeposit(const G4Step* step);
  virtual double EnergyCorrected(const G4Step& step, const G4Track*);
  virtual bool getFromLibrary(const G4Step* step);

  G4ThreeVector setToLocal(const G4ThreeVector&, const G4VTouchable*) const;
  G4ThreeVector setToGlobal(const G4ThreeVector&, const G4VTouchable*) const;

  bool hitExists(const G4Step*, int k);
  bool checkHit(int k = 0);
  CaloG4Hit* createNewHit(const G4Step*, const G4Track*, int k);
  void updateHit(CaloG4Hit*, int k);
  void resetForNewPrimary(const G4Step*);
  double getAttenuation(const G4Step* aStep, double birk1, double birk2, double birk3) const;

  static std::string printableDecayChain(const std::vector<unsigned int>& decayChain);
  std::string shortreprID(const CaloHitID& ID);
  std::string shortreprID(const CaloG4Hit* hit);
  unsigned int findBoundaryCrossingParent(const G4Track* track, bool markParentAsSaveable = true);

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
  double getResponseWt(const G4Track*, int k = 0);
  int getNumberOfHits(int k = 0);
  void ignoreRejection() { ignoreReject = true; }

  inline void setParameterized(bool val) { isParameterized = val; }
  inline void setUseMap(bool val) { useMap = val; }

  inline void processHit(const G4Step* step) {
    // check if it is in the same unit and timeslice as the previous one
    if (currentID[0] == previousID[0]) {
      updateHit(currentHit[0], 0);
    } else if (!checkHit()) {
      currentHit[0] = createNewHit(step, step->GetTrack(), 0);
    }
  }

  inline void setNumberCheckedHits(int val, int k = 0) { nCheckedHits[k] = val; }
  void printDetectorLevels(const G4VTouchable*) const;

private:
  void storeHit(CaloG4Hit*, int k = 0);
  bool saveHit(CaloG4Hit*, int k = 0);
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

  CaloHitID currentID[2], previousID[2];

  double energyCut, tmaxHit, eminHit;
  std::vector<std::string> hcn_;
  std::vector<int> useResMap_;

  CaloG4Hit* currentHit[2];

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

  std::unique_ptr<CaloSlaveSD> slave[2];
  std::unique_ptr<CaloMeanResponse> meanResponse[2];

  CaloG4HitCollection* theHC[2];

  bool ignoreTrackID;
  bool isParameterized;
  bool ignoreReject;
  bool useMap;  // use map for comparison of ID
  bool corrTOFBeam;

  int hcID[2];
  int primAncestor;
  int cleanIndex[2];
  int totalHits[2];
  int primIDSaved[2];   // ID of the last saved primary
  int nCheckedHits[2];  // number of last hits to compare ID

  float timeSlice;
  double eminHitD;
  double correctT;
  bool doFineCalo_;
  double eMinFine_;
  int nHC_;
  std::string detName_[2], collName_[2];

  std::map<CaloHitID, CaloG4Hit*> hitMap[2];
  std::map<int, TrackWithHistory*> tkMap;
  std::unordered_map<unsigned int, unsigned int> boundaryCrossingParentMap_;
  std::vector<std::unique_ptr<CaloG4Hit>> reusehit[2];
  std::vector<Detector> fineDetectors_;
  bool doFineCaloThisStep_;
};

#endif  // SimG4CMS_CaloSD_h
