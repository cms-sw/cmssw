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

class ZdcSD : public CaloSD {

public:    
  ZdcSD(const std::string&, const DDCompactView &, const SensitiveDetectorCatalog &,
	edm::ParameterSet const &,const SimTrackManager*);
 
  ~ZdcSD() = default;
  G4bool ProcessHits(G4Step * step,G4TouchableHistory * tHistory) override;
  uint32_t setDetUnitId(const G4Step* step) override;
  double getEnergyDeposit(const G4Step*, bool&);
 
  void setNumberingScheme(ZdcNumberingScheme* scheme);
  void getFromLibrary(G4Step * step);
 
protected:
  void initRun() override;
private:    

  int verbosity;
  bool  useShowerLibrary,useShowerHits; 
  int   setTrackID(const G4Step * step);
  double thFibDir;
  double zdcHitEnergyCut;

  std::unique_ptr<ZdcShowerLibrary>   showerLibrary;
  std::unique_ptr<ZdcNumberingScheme> numberingScheme;
  std::vector<ZdcShowerLibrary::Hit> hits;

};

#endif // ZdcSD_h
