#ifndef SimG4CMS_AHCalSD_h
#define SimG4CMS_AHCalSD_h

#include "SimG4CMS/Calo/interface/CaloSD.h"

#include "G4String.hh"
#include <map>
#include <string>

class G4Step;

class AHCalSD : public CaloSD {

public:    

  AHCalSD(G4String , const DDCompactView &, const SensitiveDetectorCatalog &,
	  edm::ParameterSet const &, const SimTrackManager*);
  virtual ~AHCalSD();
  virtual double                getEnergyDeposit(G4Step* );
  virtual uint32_t              setDetUnitId(G4Step* step);
  static bool                   unpackIndex(const uint32_t & idx, int & row, 
					    int& col, int& depth);
protected:

  virtual bool                  filterHit(CaloG4Hit*, double);

private:    

  bool                          useBirk;
  double                        birk1, birk2, birk3, betaThr;
  double                        eminHit;
};

#endif // AHCalSD_h
