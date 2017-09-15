///////////////////////////////////////////////////////////////////////////////
// File: ZdcSD.h
// Date: 02.04
// Description: Stores hits of Zdc in appropriate  container
//
///////////////////////////////////////////////////////////////////////////////
#ifndef ZdcSD_h
#define ZdcSD_h
#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Forward/interface/ZdcShowerLibrary.h"
#include "SimG4CMS/Forward/interface/ZdcNumberingScheme.h"
#undef debug

class ZdcSD : public CaloSD {

public:    
  ZdcSD(G4String, const DDCompactView &, const SensitiveDetectorCatalog &,
	edm::ParameterSet const &,const SimTrackManager*);
 
  ~ZdcSD() override;
  bool ProcessHits(G4Step * step,G4TouchableHistory * tHistory) override;
  uint32_t setDetUnitId(G4Step* step) override;
  virtual double getEnergyDeposit(G4Step*, edm::ParameterSet const &);
 
  void setNumberingScheme(ZdcNumberingScheme* scheme);
  void getFromLibrary(G4Step * step);
 

protected:
  void initRun() override;
private:    

  int verbosity;
  bool  useShowerLibrary,useShowerHits; 
  int   setTrackID(G4Step * step);
  double thFibDir;
  double zdcHitEnergyCut;
  ZdcShowerLibrary *    showerLibrary;
  ZdcNumberingScheme * numberingScheme;

  std::vector<ZdcShowerLibrary::Hit> hits;

};

#endif // ZdcSD_h
