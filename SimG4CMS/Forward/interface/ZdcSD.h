///////////////////////////////////////////////////////////////////////////////
// File: ZdcSD.h
// Date: 02.04
// Description: Stores hits of Zdc in appropriate  container
//
///////////////////////////////////////////////////////////////////////////////
#ifndef ZdcSD_h
#define ZdcSD_h
#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Forward/interface/ZdcNumberingScheme.h"
#undef debug

class ZdcSD : public CaloSD {

public:    
  ZdcSD(G4String, const DDCompactView &, SensitiveDetectorCatalog &, 
	edm::ParameterSet const &,const SimTrackManager*);
  virtual ~ZdcSD();
  void setNumberingScheme(ZdcNumberingScheme* scheme);
  virtual uint32_t setDetUnitId(G4Step* step);
  virtual double getEnergyDeposit(G4Step*, edm::ParameterSet const &);
private:    

  int verbosity;
  double thFibDir;
  ZdcNumberingScheme * numberingScheme;


};

#endif // ZdcSD_h
