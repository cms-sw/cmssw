#ifndef SimG4CMS_HGCSD_h
#define SimG4CMS_HGCSD_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCSD.h
// Description: Stores hits of the High Granularity Calorimeter (HGC) in the
//              appropriate container
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HGCMouseBite.h"

#include <string>
#include <TTree.h>

class G4LogicalVolume;
class G4Material;
class G4Step;

class HGCSD : public CaloSD, public Observer<const BeginOfJob *> {
public:
  HGCSD(const std::string &,
        const HGCalDDDConstants *,
        const SensitiveDetectorCatalog &,
        edm::ParameterSet const &,
        const SimTrackManager *);
  ~HGCSD() override = default;

  uint32_t setDetUnitId(const G4Step *step) override;

protected:
  double getEnergyDeposit(const G4Step *) override;
  using CaloSD::update;
  void update(const BeginOfJob *) override;
  void initRun() override;
  void initEvent(const BeginOfEvent *) override;
  void endEvent() override;
  bool filterHit(CaloG4Hit *, double) override;

private:
  uint32_t setDetUnitId(ForwardSubdetector &, int, int, int, int, G4ThreeVector &);
  bool isItinFidVolume(const G4ThreeVector &) { return true; }

  const HGCalDDDConstants *hgcons_;
  std::string nameX_;
  HGCalGeometryMode::GeometryMode geom_mode_;
  std::unique_ptr<HGCNumberingScheme> numberingScheme_;
  std::unique_ptr<HGCMouseBite> mouseBite_;
  double eminHit_;
  ForwardSubdetector myFwdSubdet_;
  double slopeMin_;
  int levelT_;
  bool storeAllG4Hits_, rejectMB_, waferRot_;
  double mouseBiteCut_;
  bool dd4hep_;
  std::vector<double> angles_;

  TTree *tree_;
  uint32_t t_EventID_;
  std::vector<int> t_Layer_, t_Parcode_;
  std::vector<double> t_dEStep1_, t_dEStep2_, t_TrackE_;
  std::vector<double> t_Angle_;
};

#endif  // HGCSD_h
