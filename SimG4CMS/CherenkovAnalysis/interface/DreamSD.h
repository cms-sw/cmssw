#ifndef SimG4CMS_DreamSD_h
#define SimG4CMS_DreamSD_h

#include "SimG4CMS/Calo/interface/CaloSD.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4String.hh"
#include <map>

class G4LogicalVolume;

class DreamSD : public CaloSD {

public:    

  DreamSD(G4String, const DDCompactView &, SensitiveDetectorCatalog &, 
	  edm::ParameterSet const &, const SimTrackManager*);
  virtual ~DreamSD();
  virtual double                    getEnergyDeposit(G4Step*);
  virtual uint32_t                  setDetUnitId(G4Step*);

private:    

  void                              initMap(G4String, const DDCompactView &);
  double                            curve_LY(G4Step*); 
  double                            crystalLength(G4LogicalVolume*);
  bool                              useBirk;
  double                            birk1, birk2;
  double                            slopeLY;
  std::map<G4LogicalVolume*,double> xtalLMap;
};

#endif // DreamSD_h
