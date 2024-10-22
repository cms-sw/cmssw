#ifndef SimG4CMS_HCalSD_h
#define SimG4CMS_HCalSD_h
///////////////////////////////////////////////////////////////////////////////
// File: HCalSD.h
// Description: Stores hits of Hadron calorimeter in appropriate container
// Use in your sensitive detector builder:
//    HCalSD* hcalSD = new HCalSD(SDname, new CaloNumberingScheme());
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Calo/interface/HFShower.h"
#include "SimG4CMS/Calo/interface/HFShowerLibrary.h"
#include "SimG4CMS/Calo/interface/HFShowerParam.h"
#include "SimG4CMS/Calo/interface/HFShowerPMT.h"
#include "SimG4CMS/Calo/interface/HFShowerFibreBundle.h"
#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalTestNS.h"
#include "CondFormats/HcalObjects/interface/HBHEDarkening.h"
#include "SimG4CMS/Calo/interface/HFDarkening.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalCommonData/interface/HcalSimulationConstants.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"

#include "G4String.hh"
#include <map>
#include <string>

class DDFilteredView;
class G4LogicalVolume;
class G4Material;
class G4Step;
class HcalTestNS;
class TH1F;

class HCalSD : public CaloSD, public Observer<const BeginOfJob*> {
public:
  HCalSD(const std::string&,
         const HcalDDDSimConstants*,
         const HcalDDDRecConstants*,
         const HcalSimulationConstants*,
         const HBHEDarkening*,
         const HBHEDarkening*,
         const SensitiveDetectorCatalog&,
         edm::ParameterSet const&,
         const SimTrackManager*);
  ~HCalSD() override = default;
  uint32_t setDetUnitId(const G4Step* step) override;
  void setNumberingScheme(HcalNumberingScheme*);

protected:
  double getEnergyDeposit(const G4Step*) override;
  bool getFromLibrary(const G4Step*) override;
  using CaloSD::update;
  void update(const BeginOfJob*) override;
  void initRun() override;
  bool filterHit(CaloG4Hit*, double) override;
  void initEvent(const BeginOfEvent*) override;
  void endEvent() override;

private:
  void fillLogVolumeVector(const std::string&, const std::vector<std::string>&, std::vector<const G4LogicalVolume*>&);
  uint32_t setDetUnitId(int, const G4ThreeVector&, int, int);
  uint32_t setDetUnitId(HcalNumberingFromDDD::HcalID& tmp);
  bool isItHF(const G4Step*);
  bool isItHF(const G4String&);
  bool isItFibre(const G4LogicalVolume*);
  bool isItFibre(const G4String&);
  bool isItPMT(const G4LogicalVolume*);
  bool isItStraightBundle(const G4LogicalVolume*);
  bool isItConicalBundle(const G4LogicalVolume*);
  bool isItScintillator(const G4Material*);
  bool isItinFidVolume(const G4ThreeVector&);
  void getFromHFLibrary(const G4Step* step, bool& isKilled);
  void hitForFibre(const G4Step* step);
  void getFromParam(const G4Step* step, bool& isKilled);
  void getHitPMT(const G4Step* step);
  void getHitFibreBundle(const G4Step* step, bool type);
  void readWeightFromFile(const std::string&);
  double layerWeight(int, const G4ThreeVector&, int, int);
  void plotProfile(const G4Step* step, const G4ThreeVector& pos, double edep, double time, int id);
  void plotHF(const G4ThreeVector& pos, bool emType);
  void modifyDepth(HcalNumberingFromDDD::HcalID& id);
  void printVolume(const G4VTouchable* touch) const;

  std::unique_ptr<HcalNumberingFromDDD> numberingFromDDD;
  std::unique_ptr<HcalNumberingScheme> numberingScheme;
  std::unique_ptr<HFShowerLibrary> showerLibrary;
  std::unique_ptr<HFShower> hfshower;
  std::unique_ptr<HFShowerParam> showerParam;
  std::unique_ptr<HFShowerPMT> showerPMT;
  std::unique_ptr<HFShowerFibreBundle> showerBundle;

  const HcalDDDSimConstants* hcalConstants_;
  const HcalSimulationConstants* hcalSimConstants_;
  const HBHEDarkening* m_HBDarkening;
  const HBHEDarkening* m_HEDarkening;
  std::unique_ptr<HFDarkening> m_HFDarkening;
  std::unique_ptr<HcalTestNS> m_HcalTestNS;

  static constexpr double maxZ_ = 10000.0;
  static constexpr double minRoff_ = -1500.0;
  static constexpr double maxRoff_ = 450.0;
  static constexpr double slopeHE_ = 0.4;
  bool isHF;
  bool agingFlagHB, agingFlagHE;
  bool useBirk, useLayerWt, useFibreBundle, usePMTHit;
  bool testNumber, neutralDensity, testNS_;
  double birk1, birk2, birk3, betaThr;
  bool useHF, useShowerLibrary, useParam, applyFidCut;
  double eminHitHB, eminHitHE, eminHitHO, eminHitHF;
  double deliveredLumi;
  double weight_;
  int depth_;
  bool dd4hep_;
  std::vector<double> gpar;
  std::vector<int> hfLevels;
  std::vector<std::string> hfNames;
  std::vector<std::string> fibreNames;
  std::vector<std::string> matNames;
  std::vector<const G4Material*> materials;
  std::vector<const G4LogicalVolume*> hfLV, fibreLV, pmtLV, fibre1LV, fibre2LV;
  std::map<uint32_t, double> layerWeights;
  TH1F *hit_[9], *time_[9], *dist_[9], *hzvem, *hzvhad;
  std::vector<int> detNull_;
};

#endif  // HCalSD_h
