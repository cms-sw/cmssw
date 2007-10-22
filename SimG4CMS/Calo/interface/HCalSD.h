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
#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"

#include "G4String.hh"

#include <string>

class DDCompactView;
class DDFilteredView;
class G4Step;

class HCalSD : public CaloSD {

public:    

  HCalSD(G4String , const DDCompactView &, SensitiveDetectorCatalog &,
	 edm::ParameterSet const &, const SimTrackManager*);
  virtual ~HCalSD();
  virtual bool ProcessHits(G4Step * step,G4TouchableHistory * tHistory);
  virtual double getEnergyDeposit(G4Step* );
  virtual uint32_t setDetUnitId(G4Step* step);
  void setNumberingScheme(HcalNumberingScheme* scheme);

private:    

  uint32_t              setDetUnitId(int, G4ThreeVector, int, int);
  std::vector<double>   getDDDArray(const std::string&, const DDsvalues_type&);
  std::vector<G4String> getNames(DDFilteredView&);
  bool                  isItHF(G4String);
  bool                  isItFibre(G4String);
  bool                  isItPMT(G4String);
  bool                  isItScintillator(G4String);
  void                  getFromLibrary(G4Step * step);
  void                  hitForFibre(G4Step * step);
  void                  getFromParam(G4Step * step);
  void                  getHitPMT(G4Step * step);
  int                   setTrackID(G4Step * step);

  HcalNumberingFromDDD* numberingFromDDD;
  HcalNumberingScheme*  numberingScheme;
  HFShowerLibrary *     showerLibrary;
  HFShower *            hfshower;
  HFShowerParam *       showerParam;
  HFShowerPMT *         showerPMT;
  bool                  useBirk;
  double                birk1, birk2;
  bool                  useHF, useShowerLibrary, useParam, usePMTHit;
  std::vector<double>   layer0wt;
  std::vector<G4String> hfNames;
  std::vector<G4String> fibreNames;
  std::vector<G4String> matNames;
  std::vector<G4String> pmtNames;

};

#endif // HCalSD_h
