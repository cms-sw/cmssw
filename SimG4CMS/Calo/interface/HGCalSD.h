#ifndef SimG4CMS_HGCalSD_h
#define SimG4CMS_HGCalSD_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCalSD.h
// Description: Stores hits of the High Granularity Calorimeter (HGC) in the
//              appropriate container (post TDR version)
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4CMS/Calo/interface/HGCalNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HGCGuardRing.h"
#include "SimG4CMS/Calo/interface/HGCMouseBite.h"
#include "SimG4CMS/Calo/interface/HGCGuardRingPartial.h"
#include <string>

class HGCalDDDConstants;
class G4LogicalVolume;
class G4Step;

class HGCalSD : public CaloSD, public Observer<const BeginOfJob *> {
public:
  HGCalSD(const std::string &,
          const HGCalDDDConstants *,
          const SensitiveDetectorCatalog &,
          edm::ParameterSet const &,
          const SimTrackManager *);
  ~HGCalSD() override = default;

  uint32_t setDetUnitId(const G4Step *step) override;

protected:
  double getEnergyDeposit(const G4Step *) override;
  using CaloSD::update;
  void update(const BeginOfJob *) override;
  void initRun() override;
  bool filterHit(CaloG4Hit *, double) override;

private:
  uint32_t setDetUnitId(int, int, int, int, G4ThreeVector &);
  bool isItinFidVolume(const G4ThreeVector &);

  const HGCalDDDConstants *hgcons_;
  std::unique_ptr<HGCalNumberingScheme> numberingScheme_;
  std::unique_ptr<HGCGuardRing> guardRing_;
  std::unique_ptr<HGCGuardRingPartial> guardRingPartial_;
  std::unique_ptr<HGCMouseBite> mouseBite_;
  DetId::Detector mydet_;
  std::string nameX_;
  HGCalGeometryMode::GeometryMode geom_mode_;
  double eminHit_, slopeMin_, distanceFromEdge_;
  double mouseBiteCut_, weight_;
  int levelT1_, levelT2_, cornerMinMask_;
  bool storeAllG4Hits_;
  bool fiducialCut_, rejectMB_, waferRot_, checkID_;
  int useSimWt_, verbose_;
  bool dd4hep_;
  const double tan30deg_, cos30deg_;
  std::vector<double> angles_;
  std::string missingFile_;
};

#endif  // HGCalSD_h
