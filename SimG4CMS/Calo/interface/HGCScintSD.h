#ifndef SimG4CMS_HGCScintSD_h
#define SimG4CMS_HGCScintSD_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCScintSD.h
// Description: Stores hits of the High Granularity Calorimeter (HGC) in the
//              appropriate container (post TDR version)
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4CMS/Calo/interface/HGCalNumberingScheme.h"

#include <string>

class DDCompactView;
class HGCalDDDConstants;
class G4LogicalVolume;
class G4Step;

class HGCScintSD : public CaloSD, public Observer<const BeginOfJob *> {

public:    

  HGCScintSD(const std::string& , const DDCompactView &, 
	     const SensitiveDetectorCatalog &, edm::ParameterSet const &,
	     const SimTrackManager*);
  ~HGCScintSD() override  = default;

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

  const HGCalDDDConstants*              hgcons_;
  std::unique_ptr<HGCalNumberingScheme> numberingScheme_;
  DetId::Detector                       mydet_;
  std::string                           nameX_;
  HGCalGeometryMode::GeometryMode       geom_mode_;
  double                                eminHit_, slopeMin_, distanceFromEdge_;
  int                                   levelT1_, levelT2_;
  bool                                  storeAllG4Hits_, fiducialCut_;
  bool                                  useBirk_;
  double                                birk1_, birk2_, birk3_, weight_;
};

#endif // HGCScintSD_h
