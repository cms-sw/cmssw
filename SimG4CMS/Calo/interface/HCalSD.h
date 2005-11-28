///////////////////////////////////////////////////////////////////////////////
// File: HCalSD.h
// Description: Stores hits of Hadron calorimeter in appropriate container
// Use in your sensitive detector builder:
//    HCalSD* hcalSD = new HCalSD(SDname, new CaloNumberingScheme());
///////////////////////////////////////////////////////////////////////////////
#ifndef HCalSD_h
#define HCalSD_h

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Calo/interface/HFShower.h"
#include "SimG4CMS/Calo/interface/HFShowerLibrary.h"
#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"

#include "G4String.hh"

#include <string>

class DDCompactView;
class DDFilteredView;
class G4Step;
class SimTrackManager;

class HCalSD : public CaloSD {

public:    

  HCalSD(G4String , const DDCompactView &, edm::ParameterSet const &,
	 const SimTrackManager*);
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
  bool                  isItScintillator(G4String);
  void                  getFromLibrary(G4Step * step);
  void                  hitForFibre(G4Step * step);

  HcalNumberingFromDDD* numberingFromDDD;
  HcalNumberingScheme*  numberingScheme;
  HFShowerLibrary *     showerLibrary;
  HFShower *            hfshower;
  int                   verbosity;
  bool                  useBirk;
  double                birk1, birk2;
  bool                  useShowerLibrary;
  std::vector<double>   layer0wt;
  std::vector<G4String> hfNames;
  std::vector<G4String> fibreNames;
  std::vector<G4String> matNames;

};

#endif // HCalSD_h
