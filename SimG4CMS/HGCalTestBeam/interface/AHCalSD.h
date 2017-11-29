#ifndef SimG4CMS_AHCalSD_h
#define SimG4CMS_AHCalSD_h

#include "SimG4CMS/Calo/interface/CaloSD.h"

#include "G4String.hh"
#include <map>
#include <string>

class G4Step;

class AHCalSD : public CaloSD {

public:    

  AHCalSD(const std::string& , const DDCompactView &, const SensitiveDetectorCatalog &,
	  edm::ParameterSet const &, const SimTrackManager*);
  ~AHCalSD() override;
  double                getEnergyDeposit(G4Step* ) override;
  uint32_t              setDetUnitId(const G4Step* step) override;
  bool                  unpackIndex(const uint32_t & idx, int & row, 
				    int& col, int& depth);
protected:

  bool                  filterHit(CaloG4Hit*, double) override;

private:    

  bool                          useBirk;
  double                        birk1, birk2, birk3, betaThr;
  double                        eminHit;
};

#endif // AHCalSD_h
