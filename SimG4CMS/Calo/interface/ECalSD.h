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
#include "CondFormats/GeometryObjects/interface/EcalSimulationParameters.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
  ECalSD(const std::string &,
         const edm::EventSetup &,
         const SensitiveDetectorCatalog &,
         edm::ParameterSet const &p,
         const SimTrackManager *);
  ~ECalSD() override;
  uint32_t setDetUnitId(const G4Step *) override;
  void setNumberingScheme(EcalNumberingScheme *);

protected:
  double getEnergyDeposit(const G4Step *) override;
  int getTrackID(const G4Track *) override;
  uint16_t getDepth(const G4Step *) override;
  double EnergyCorrected(const G4Step &, const G4Track *) override;

private:
  void initMap();
  uint16_t getRadiationLength(const G4StepPoint *hitPoint, const G4LogicalVolume *lv);
  uint16_t getLayerIDForTimeSim();
  double curve_LY(const G4LogicalVolume *);

  void getBaseNumber(const G4Step *);
  double getBirkL3(const G4Step *);
  std::string getNameNoNS(const std::string &name);

  // initialised before run
  const EcalSimulationParameters *ecalSimParameters_;
  EcalNumberingScheme *numberingScheme_;
  bool useWeight, storeTrack, storeRL, storeLayerTimeSim;
  bool useBirk, useBirkL3;
  double birk1, birk2, birk3, birkSlope, birkCut;
  double slopeLY, scaleRL;
  std::string crystalMat, depth1Name, depth2Name;
  std::map<const G4LogicalVolume *, double> xtalLMap;
  std::vector<const G4LogicalVolume *> useDepth1, useDepth2, noWeight;
  EcalBaseNumber theBaseNumber;
  EnergyResolutionVsLumi ageing;
  bool ageingWithSlopeLY;

  // run time cache
  G4ThreeVector currentLocalPoint;
  double crystalLength;
  double crystalDepth;
  uint16_t depth;
#ifdef plotDebug
  TH2F *g2L_[4];
#endif
};

#endif  // ECalSD_h
