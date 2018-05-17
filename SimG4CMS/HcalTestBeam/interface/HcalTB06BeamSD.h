#ifndef SimG4CMS_HcalTB06BeamSD_h
#define SimG4CMS_HcalTB06BeamSD_h
///////////////////////////////////////////////////////////////////////////////
// File: HcalTB06BeamSD.h
// Description: Stores hits of Beam counters for H2 TB06 in appropriate 
//              containers
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"

#include "G4String.hh"

#include <string>

class DDCompactView;
class DDFilteredView;
class G4Step;
class G4Material;

class HcalTB06BeamSD : public CaloSD {

public:    

  HcalTB06BeamSD(const std::string&, const DDCompactView &, 
                 const SensitiveDetectorCatalog &,
		 edm::ParameterSet const &, const SimTrackManager*);
  ~HcalTB06BeamSD() override;
  uint32_t setDetUnitId(const G4Step* step) override;

protected:

  double getEnergyDeposit(const G4Step*) override;

private:    

  std::vector<G4String> getNames(DDFilteredView&);
  bool                  isItWireChamber(const G4String&);

  bool                  useBirk;
  double                birk1, birk2, birk3;
  std::vector<G4String> wcNames;
  G4String              matName;
};

#endif // HcalTB06BeamSD_h
