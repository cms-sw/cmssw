#ifndef SimG4CMS_HGCSD_h
#define SimG4CMS_HGCSD_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCSD.h
// Description: Stores hits of the High Granularity Calorimeter (HGC) in the
//              appropriate container
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HGCMouseBite.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"

#include <string>

class DDCompactView;
class DDFilteredView;
class G4LogicalVolume;
class G4Material;
class G4Step;

class HGCSD : public CaloSD, public Observer<const BeginOfJob *> {

public:    

  HGCSD(const std::string& , const DDCompactView &, const SensitiveDetectorCatalog &,
	edm::ParameterSet const &, const SimTrackManager*);
  ~HGCSD() override;

  uint32_t                setDetUnitId(const G4Step* step) override;

protected:

  double                  getEnergyDeposit(const G4Step* ) override;
  void                    update(const BeginOfJob *) override;
  void                    initRun() override;
  bool                    filterHit(CaloG4Hit*, double) override;

private:    

  uint32_t                setDetUnitId(ForwardSubdetector&, int, int, 
				       int, int, G4ThreeVector &);
  bool                    isItinFidVolume (const G4ThreeVector&) {return true;}

  std::string                     nameX_;

  HGCalGeometryMode::GeometryMode geom_mode_;
  HGCNumberingScheme*             numberingScheme_;
  HGCMouseBite*                   mouseBite_;
  double                          eminHit_;
  ForwardSubdetector              myFwdSubdet_;
  double                          slopeMin_;
  int                             levelT_;
  bool                            storeAllG4Hits_, rejectMB_, waferRot_;
  double                          mouseBiteCut_;
  std::vector<double>             angles_;
};

#endif // HGCSD_h
