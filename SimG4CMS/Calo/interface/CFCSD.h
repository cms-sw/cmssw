#ifndef SimG4CMS_CFCSD_h
#define SimG4CMS_CFCSD_h
///////////////////////////////////////////////////////////////////////////////
// File: CFCSD.h
// Description: Stores hits of the Combined Forward Calorimeter (CFC) in the
//              appropriate container
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Calo/interface/CFCShowerLibrary.h"
#include "SimG4CMS/Calo/interface/CFCNumberingScheme.h"
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

class CFCSD : public CaloSD {

public:    

  CFCSD(G4String , const DDCompactView &, SensitiveDetectorCatalog &,
	edm::ParameterSet const &, const SimTrackManager*);
  virtual ~CFCSD();
  virtual bool                  ProcessHits(G4Step * , G4TouchableHistory * );
  virtual double                getEnergyDeposit(G4Step* );
  virtual uint32_t              setDetUnitId(G4Step* step);

protected:

  virtual void                  initRun();
  virtual bool                  filterHit(CaloG4Hit*, double);

private:    

  uint32_t                      setDetUnitId(int, G4ThreeVector, int, int);
  std::vector<double>           getDDDArray(const std::string&, 
                                            const DDsvalues_type&);
  bool                          isItinFidVolume (G4ThreeVector&);
  void                          getFromLibrary(G4Step * step);
  int                           setTrackID(G4Step * step);
  double                        attLength(double lambda);
  double                        tShift(G4ThreeVector point);
  double                        fiberL(G4ThreeVector point);

  CFCNumberingScheme*           numberingScheme;
  CFCShowerLibrary *            showerLibrary;
  G4int                         mumPDG, mupPDG, nBinAtt; 
  double                        eminHit, cFibre;
  bool                          applyFidCut;
  std::vector<double>           attL, lambLim, gpar;

};

#endif // CFCSD_h
