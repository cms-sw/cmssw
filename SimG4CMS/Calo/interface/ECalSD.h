#ifndef SimG4CMS_ECalSD_h
#define SimG4CMS_ECalSD_h
///////////////////////////////////////////////////////////////////////////////
// File: ECalSD.h
// Description: Stores hits of Electromagnetic calorimeters in appropriate
//              container
// Use in your sensitive detector builder:
//    ECalSD* ecalSD = new ECalSD(SDname, new CaloNumberingScheme());
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4String.hh"
#include <map>

class EcalBaseNumber;
class G4LogicalVolume;

class ECalSD : public CaloSD {

public:    

  ECalSD(G4String, const DDCompactView &, SensitiveDetectorCatalog &, 
	 edm::ParameterSet const &, const SimTrackManager*);
  virtual ~ECalSD();
  virtual double                    getEnergyDeposit(G4Step*);
  virtual int                       getRadiationLenght(G4Step *);
  virtual uint32_t                  setDetUnitId(G4Step*);
  void                              setNumberingScheme(EcalNumberingScheme*);

private:    

  void                              initMap(G4String, const DDCompactView &);
  double                            curve_LY(G4Step*); 
  double                            crystalLength(G4LogicalVolume*);
  void                              getBaseNumber(const G4Step*); 
  double                            getBirkL3(G4Step*);

  EcalNumberingScheme *             numberingScheme;
  bool                              useWeight;
  bool                              useBirk, useBirkL3;
  double                            birk1, birk2, birk3, birkSlope, birkCut;
  double                            slopeLY;
  std::map<G4LogicalVolume*,double> xtalLMap;
  EcalBaseNumber                    theBaseNumber;
};

#endif // ECalSD_h
