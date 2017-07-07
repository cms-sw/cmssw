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

#include "G4String.hh"
#include <map>
#include <string>
#include <TH1F.h>

class DDCompactView;
class DDFilteredView;
class G4LogicalVolume;
class G4Material;
class G4Step;

class HGCSD : public CaloSD, public Observer<const BeginOfJob *> {

public:    

  HGCSD(G4String , const DDCompactView &, const SensitiveDetectorCatalog &,
	edm::ParameterSet const &, const SimTrackManager*);
  ~HGCSD() override;
  bool                    ProcessHits(G4Step * , G4TouchableHistory * ) override;
  double                  getEnergyDeposit(G4Step* ) override;
  uint32_t                setDetUnitId(G4Step* step) override;

protected:

  void                    update(const BeginOfJob *) override;
  void                    initRun() override;
  bool                    filterHit(CaloG4Hit*, double) override;

private:    

  uint32_t                        setDetUnitId(ForwardSubdetector&, int, int, 
					       int, int, G4ThreeVector &);
  bool                            isItinFidVolume (G4ThreeVector&) {return true;}
  int                             setTrackID(G4Step * step);

  std::string                     nameX;

  HGCalGeometryMode::GeometryMode m_mode;
  HGCNumberingScheme*             numberingScheme;
  HGCMouseBite*                   mouseBite_;
  G4int                           mumPDG, mupPDG; 
  double                          eminHit;
  ForwardSubdetector              myFwdSubdet_;
  double                          slopeMin_;
  int                             levelT_;
  bool                            storeAllG4Hits_, rejectMB_, waferRot_;
  double                          mouseBiteCut_;
  std::vector<double>             angles_;
};

#endif // HGCSD_h
