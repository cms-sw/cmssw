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
#include "SimG4CMS/Calo/interface/HGCMouseBite.h"

#include <string>

class DDCompactView;
class G4LogicalVolume;
class G4Step;

class HGCalSD : public CaloSD, public Observer<const BeginOfJob *> {

public:    

  HGCalSD(const std::string& , const DDCompactView &, const SensitiveDetectorCatalog &,
	  edm::ParameterSet const &, const SimTrackManager*);
  ~HGCalSD() override;

  uint32_t                setDetUnitId(const G4Step* step) override;

protected:

  double                  getEnergyDeposit(const G4Step*) override;
  void                    update(const BeginOfJob *) override;
  void                    initRun() override;
  bool                    filterHit(CaloG4Hit*, double) override;

private:    

  uint32_t                setDetUnitId(int, int, int, int, G4ThreeVector &);
  bool                    isItinFidVolume (const G4ThreeVector&) {return true;}

  HGCalNumberingScheme*           numberingScheme_;
  HGCMouseBite*                   mouseBite_;
  DetId::Detector                 mydet_;
  std::string                     nameX_;
  HGCalGeometryMode::GeometryMode geom_mode_;
  double                          eminHit_, slopeMin_, mouseBiteCut_;
  int                             levelT1_, levelT2_;
  bool                            storeAllG4Hits_, rejectMB_, waferRot_;
  bool                            useBirk_, isScint_;
  const double                    tan30deg_;
  double                          birk1_, birk2_, birk3_, weight_;
  std::vector<double>             angles_;
};

#endif // HGCalSD_h
