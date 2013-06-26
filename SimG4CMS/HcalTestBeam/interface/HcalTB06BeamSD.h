#ifndef SimG4CMS_HcalTB06BeamSD_h
#define SimG4CMS_HcalTB06BeamSD_h
///////////////////////////////////////////////////////////////////////////////
// File: HcalTB06BeamSD.h
// Description: Stores hits of Beam counters for H2 TB06 in appropriate 
//              containers
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"

#include "G4String.hh"

#include <boost/cstdint.hpp>
#include <string>

class DDCompactView;
class DDFilteredView;
class G4Step;

class HcalTB06BeamSD : public CaloSD {

public:    

  HcalTB06BeamSD(G4String , const DDCompactView &, SensitiveDetectorCatalog &,
		 edm::ParameterSet const &, const SimTrackManager*);
  virtual ~HcalTB06BeamSD();
  virtual double getEnergyDeposit(G4Step* );
  virtual uint32_t setDetUnitId(G4Step* step);
  static uint32_t  packIndex(int det, int lay, int x, int y);
  static void      unpackIndex(const uint32_t & idx, int& det, int& lay,
			       int& x, int& y);

private:    

  std::vector<G4String> getNames(DDFilteredView&);
  bool                  isItWireChamber(G4String);

  bool                  useBirk;
  double                birk1, birk2, birk3;
  std::vector<G4String> wcNames;
  G4String              matName;

};

#endif // HcalTB06BeamSD_h
