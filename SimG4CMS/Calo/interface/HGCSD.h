#ifndef SimG4CMS_HGCSD_h
#define SimG4CMS_HGCSD_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCSD.h
// Description: Stores hits of the High Granularity Calorimeter (HGC) in the
//              appropriate container
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Calo/interface/HGCNumberingScheme.h"
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

class HGCSD : public CaloSD {

public:    

  HGCSD(G4String , const DDCompactView &, const SensitiveDetectorCatalog &,
	edm::ParameterSet const &, const SimTrackManager*);
  virtual ~HGCSD();
  virtual bool                  ProcessHits(G4Step * , G4TouchableHistory * );
  virtual double                getEnergyDeposit(G4Step* );
  virtual uint32_t              setDetUnitId(G4Step* step);

protected:

  virtual void                  initRun();
  virtual bool                  filterHit(CaloG4Hit*, double);

private:    

  uint32_t                      setDetUnitId(ForwardSubdetector &, int &, int &, int &, G4ThreeVector &);
  bool                          isItinFidVolume (G4ThreeVector&) {return true;}
  int                           setTrackID(G4Step * step);

  HGCNumberingScheme*           numberingScheme;
  G4int                         verbosity, mumPDG, mupPDG; 
  double                        eminHit;
  ForwardSubdetector            myFwdSubdet_;
};

#endif // HGCSD_h
