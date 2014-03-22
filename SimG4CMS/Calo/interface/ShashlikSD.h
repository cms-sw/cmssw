#ifndef SimG4CMS_ShashlikSD_h
#define SimG4CMS_ShashlikSD_h
///////////////////////////////////////////////////////////////////////////////
// File: ShashlikSD.h
// Description: Stores hits of Shashlik detector in appropriate  container
// Use in your sensitive detector builder:
//    ShashlikSD* ecalSD = new ShashlikSD(SDname, cpv, sdc, ps, tkm);
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/ShashlikDDDConstants.h"

#include "G4String.hh"
 
#include "TROOT.h"
#include "TFile.h"
#include "TH2D.h"

#include <string>
#include <map>

class G4LogicalVolume;

class ShashlikSD : public CaloSD {

public:    

  ShashlikSD(G4String, const DDCompactView &, SensitiveDetectorCatalog &, 
	     edm::ParameterSet const &, const SimTrackManager*);
  virtual ~ShashlikSD();
  virtual bool                      ProcessHits(G4Step*, G4TouchableHistory*);
  virtual double                    getEnergyDeposit(G4Step*);
  virtual uint16_t                  getDepth(G4Step*);
  virtual uint32_t                  setDetUnitId(G4Step*);

private:    

  G4double                          fiberWt(G4int, G4ThreeVector);
  G4double                          fiberLoss(G4int, G4int); 
  std::vector<double>               getDDDArray(const std::string&,
						const DDsvalues_type&);

  ShashlikDDDConstants*             sdc;
  bool                              useWeight, useAtt,  useBirk;
  double                            birk1, birk2, birk3, attL, moduleL;
  int                               roType;
  TH2D                             *hFibre[5];
  std::vector<double>               fiberL;
};

#endif // ShashlikSD_h
