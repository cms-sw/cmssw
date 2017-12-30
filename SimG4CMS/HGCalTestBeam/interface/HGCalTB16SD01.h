#ifndef SimG4CMS_HGCalTB16SD01_h
#define SimG4CMS_HGCalTB16SD01_h
///////////////////////////////////////////////////////////////////////////////
// File: HGCalTB16SD01.h
// Description: Stores hits of Beam counters for Fermilab TB16 in appropriate 
//              containers
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"

#include "G4String.hh"

#include <string>

class DDCompactView;
class G4Step;
class G4Material;

class HGCalTB16SD01 : public CaloSD {

public:    

  HGCalTB16SD01(const std::string& , const DDCompactView &, 
		const SensitiveDetectorCatalog &, edm::ParameterSet const &, 
		const SimTrackManager*);
  ~HGCalTB16SD01() override;
  uint32_t setDetUnitId(const G4Step* step) override;
  static uint32_t  packIndex(int det, int lay, int x, int y);
  static void      unpackIndex(const uint32_t & idx, int& det, int& lay,
			       int& x, int& y);

protected:

  double           getEnergyDeposit(const G4Step*, bool& ) override;

private:    
  void             initialize(const G4StepPoint* point);

  std::string      matName_;
  bool             useBirk_;
  double           birk1_, birk2_, birk3_;
  bool             initialize_;
  G4Material*      matScin_;
};

#endif // HGCalTB16SD01_h
