#ifndef SimG4CMS_ECalSD_h
#define SimG4CMS_ECalSD_h
///////////////////////////////////////////////////////////////////////////////
// File: ECalSD.h
// Description: Stores hits of Electromagnetic calorimeters in appropriate
//              container
// Use in your sensitive detector builder:
//    ECalSD* ecalSD = new ECalSD(SDname, new CaloNumberingScheme());
///////////////////////////////////////////////////////////////////////////////
//#define plotDebug

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Calo/interface/EnergyResolutionVsLumi.h"
#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4String.hh"
#ifdef plotDebug
#include <TH2F.h>
#endif
#include <string>
#include <map>

class EcalBaseNumber;
class G4LogicalVolume;
class EnergyResolutionVsLumi;

class ECalSD : public CaloSD {

public:    

  ECalSD(const std::string&, const DDCompactView &, const SensitiveDetectorCatalog &,
	 edm::ParameterSet const & p, const SimTrackManager*);
  ~ECalSD() override;
  double                    getEnergyDeposit(G4Step*) override;
  virtual uint16_t          getRadiationLength(const G4Step *);
  virtual uint16_t          getLayerIDForTimeSim(const G4Step *);
  uint32_t                  setDetUnitId(const G4Step*) override;
  void                      setNumberingScheme(EcalNumberingScheme*);

protected:

  int                       getTrackID(const G4Track*) override;
  uint16_t                  getDepth(const G4Step*) override;

private:    

  void                              initMap(const G4String&, const DDCompactView &);
  double                            curve_LY(const G4Step*); 
  double                            crystalLength(const G4LogicalVolume*);
  double                            crystalDepth(const G4LogicalVolume*, const G4ThreeVector&);
  void                              getBaseNumber(const G4Step*); 
  double                            getBirkL3(const G4Step*);
  std::vector<double>               getDDDArray(const std::string&,
						const DDsvalues_type&);
  std::vector<std::string>          getStringArray(const std::string&,
						   const DDsvalues_type&);

  bool                              isEB;
  bool                              isEE;
  EcalNumberingScheme *             numberingScheme;
  bool                              useWeight, storeTrack, storeRL, storeLayerTimeSim;
  bool                              useBirk, useBirkL3;
  double                            birk1, birk2, birk3, birkSlope, birkCut;
  double                            slopeLY, scaleRL;
  std::string                       crystalMat, depth1Name, depth2Name;
  std::map<const G4LogicalVolume*,double> xtalLMap;
  std::vector<const G4LogicalVolume*>     useDepth1, useDepth2, noWeight;
  EcalBaseNumber                    theBaseNumber;
  EnergyResolutionVsLumi            ageing;
  bool                              ageingWithSlopeLY;
#ifdef plotDebug
  TH2F                             *g2L_[4];
#endif
};

#endif // ECalSD_h
