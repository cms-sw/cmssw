///////////////////////////////////////////////////////////////////////////////
// File: CastorSD.h
// Description: Stores hits of Castor in appropriate  container
// Use in your sensitive detector builder:
//    CastorSD* castorSD = new CastorSD(SDname, new CaloNumberingScheme());
///////////////////////////////////////////////////////////////////////////////
#ifndef CastorSD_h
#define CastorSD_h

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Forward/interface/CastorNumberingScheme.h"

class CastorSD : public CaloSD {

public:    

  CastorSD(G4String, const DDCompactView &, edm::ParameterSet const &,
	   const SimTrackManager*);
  virtual ~CastorSD();
  virtual double   getEnergyDeposit(G4Step* );
  virtual uint32_t setDetUnitId(G4Step* step);
  void             setNumberingScheme(CastorNumberingScheme* scheme);

private:    
  double curve_Castor(G4String& , G4StepPoint*); 
  int                     verbosity;
  CastorNumberingScheme * numberingScheme;
};

#endif // CastorSD_h
