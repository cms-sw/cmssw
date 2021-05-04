#ifndef SimG4CMS_Muon_MuonSensitiveDetector_h
#define SimG4CMS_Muon_MuonSensitiveDetector_h

/** \class MuonSensitiveDetector
 *
 * implementation of SensitiveDetector for the muon detector;
 * a MuonSlaveSD handles the interfacing to the database;
 * numbering scheme are booked according
 * to the detector name
 * 
 * \author Arno Straessner, CERN <arno.straessner@cern.ch>
 *
 * Modification:
 * 19/05/03. P.Arce
 * Add SimTracks selection
 */

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "CondFormats/GeometryObjects/interface/MuonOffsetMap.h"

#include <string>

class MuonSlaveSD;
class MuonSimHitNumberingScheme;
class MuonFrameRotation;
class UpdatablePSimHit;
class MuonSubDetector;
class MuonG4Numbering;
class SimHitPrinter;
class G4Step;
class G4ProcessTypeEnumerator;
class SimTrackManager;

class MuonSensitiveDetector : public SensitiveTkDetector, public Observer<const BeginOfEvent*> {
public:
  explicit MuonSensitiveDetector(const std::string&,
                                 const edm::EventSetup&,
                                 const SensitiveDetectorCatalog&,
                                 edm::ParameterSet const&,
                                 const SimTrackManager*);
  ~MuonSensitiveDetector() override;
  G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
  uint32_t setDetUnitId(const G4Step*) override;
  void EndOfEvent(G4HCofThisEvent*) override;

  void fillHits(edm::PSimHitContainer&, const std::string&) override;
  void clearHits() override;

  const MuonSlaveSD* GetSlaveMuon() const { return slaveMuon; }

protected:
  void update(const BeginOfEvent*) override;

private:
  inline Local3DPoint cmsUnits(const Local3DPoint& v) { return Local3DPoint(v.x() * 0.1, v.y() * 0.1, v.z() * 0.1); }

  MuonSlaveSD* slaveMuon;
  MuonSimHitNumberingScheme* numbering;
  MuonSubDetector* detector;
  const MuonFrameRotation* theRotation;
  MuonG4Numbering* g4numbering;

  bool newHit(const G4Step*);
  void createHit(const G4Step*);
  void updateHit(const G4Step*);
  void saveHit();

  /**
   * Transform from local coordinates of a volume to local coordinates of a parent volume
   * one or more levels up the volume hierarchy: e.g. levelsUp = 1 for immediate parent.
   * This is done by moving from local_1 -> global -> local_2.
   */
  Local3DPoint InitialStepPositionVsParent(const G4Step* currentStep, G4int levelsUp);
  Local3DPoint FinalStepPositionVsParent(const G4Step* currentStep, G4int levelsUp);

  const G4VPhysicalVolume* thePV;
  UpdatablePSimHit* theHit;
  uint32_t theDetUnitId;
  uint32_t newDetUnitId;
  int theTrackID;

  bool printHits;
  SimHitPrinter* thePrinter;

  //--- SimTracks cuts
  float ePersistentCutGeV;
  bool allMuonsPersistent;

  G4ProcessTypeEnumerator* theG4ProcessTypeEnumerator;

  const SimTrackManager* theManager;
};

#endif  // MuonSensitiveDetector_h
