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
#include "SimG4CMS/Calo/interface/EnergyResolutionVsLumi.h"
#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4String.hh"
#include <string>
#include <map>

class EcalBaseNumber;
class G4LogicalVolume;
class EnergyResolutionVsLumi;

class ECalSD : public CaloSD {

public:    

  ECalSD(G4String, const DDCompactView &, SensitiveDetectorCatalog &, 
	 edm::ParameterSet const & p, const SimTrackManager*);
  virtual ~ECalSD();
  virtual double                    getEnergyDeposit(G4Step*);
  virtual uint16_t                  getRadiationLength(G4Step *);
  virtual uint32_t                  setDetUnitId(G4Step*);
  void                              setNumberingScheme(EcalNumberingScheme*);
  virtual int                       getTrackID(G4Track*);
  virtual uint16_t                  getDepth(G4Step*);

private:    

  void                              initMap(G4String, const DDCompactView &);
  double                            curve_LY(G4Step*); 
  double                            crystalLength(G4LogicalVolume*);
  void                              getBaseNumber(const G4Step*); 
  double                            getBirkL3(G4Step*);
  std::vector<double>               getDDDArray(const std::string&,
						const DDsvalues_type&);
  std::vector<std::string>          getStringArray(const std::string&,
						   const DDsvalues_type&);

  EcalNumberingScheme *             numberingScheme;
  bool                              useWeight, storeTrack, storeRL;
  bool                              useBirk, useBirkL3;
  double                            birk1, birk2, birk3, birkSlope, birkCut;
  double                            slopeLY;
  std::string                       crystalMat, depth1Name, depth2Name;
  std::map<G4LogicalVolume*,double> xtalLMap;
  std::vector<G4LogicalVolume*>     useDepth1, useDepth2, noWeight;
  EcalBaseNumber                    theBaseNumber;
  EnergyResolutionVsLumi            ageing;
  bool                              ageingWithSlopeLY;
};

#endif // ECalSD_h
