#ifndef SimG4CMS_HFNoseSD_h
#define SimG4CMS_HFNoseSD_h
///////////////////////////////////////////////////////////////////////////////
// File: HFNoseSD.h
// Description: Stores hits of the High Granularity Calorimeter (HGC) in the
//              appropriate container (post TDR version)
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4CMS/Calo/interface/HFNoseNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HGCMouseBite.h"

#include <string>

class DDCompactView;
class HGCalDDDConstants;
class G4LogicalVolume;
class G4Step;

class HFNoseSD : public CaloSD, public Observer<const BeginOfJob *> {

public:    

  HFNoseSD(const std::string& , const DDCompactView &,
	   const SensitiveDetectorCatalog &,
	   edm::ParameterSet const &, const SimTrackManager*);
  ~HFNoseSD() override  = default;

  uint32_t                setDetUnitId(const G4Step* step) override;

protected:

  double                  getEnergyDeposit(const G4Step*) override;
  using CaloSD::update;
  void                    update(const BeginOfJob *) override;
  void                    initRun() override;
  bool                    filterHit(CaloG4Hit*, double) override;

private:    

  uint32_t                setDetUnitId(int, int, int, int, G4ThreeVector &);
  bool                    isItinFidVolume (const G4ThreeVector&);

  const HGCalDDDConstants*               hgcons_;
  std::unique_ptr<HFNoseNumberingScheme> numberingScheme_;
  std::unique_ptr<HGCMouseBite>          mouseBite_;
  std::string                            nameX_;
  HGCalGeometryMode::GeometryMode        geom_mode_;
  double                                 eminHit_, slopeMin_, weight_;
  double                                 mouseBiteCut_, distanceFromEdge_;
  int                                    levelT1_, levelT2_;
  bool                                   storeAllG4Hits_;
  bool                                   fiducialCut_, rejectMB_, waferRot_;
  const double                           tan30deg_;
  std::vector<double>                    angles_;
};

#endif // HFNoseSD_h
