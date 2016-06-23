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
class DDFilteredView;
class G4Step;
class G4Material;

class HGCalTB16SD01 : public CaloSD {

public:    

  HGCalTB16SD01(G4String , const DDCompactView &, 
		const SensitiveDetectorCatalog &, edm::ParameterSet const &, 
		const SimTrackManager*);
  virtual ~HGCalTB16SD01();
  virtual void   initRun();
  virtual double getEnergyDeposit(G4Step* );
  virtual uint32_t setDetUnitId(G4Step* step);
  static uint32_t  packIndex(int det, int lay, int x, int y);
  static void      unpackIndex(const uint32_t & idx, int& det, int& lay,
			       int& x, int& y);

private:    

  std::vector<G4String> getNames(DDFilteredView&);

  bool                  useBirk;
  double                birk1, birk2, birk3;
  G4String              matName;
  const G4Material*     matScin;
};

#endif // HGCalTB16SD01_h
