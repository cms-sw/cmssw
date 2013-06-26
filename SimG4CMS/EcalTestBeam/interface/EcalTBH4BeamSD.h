#ifndef SimG4CMS_EcalTBH4BeamSD_h
#define SimG4CMS_EcalTBH4BeamSD_h
///////////////////////////////////////////////////////////////////////////////
// File: EcalTBH4BeamSD.h
// Description: Stores hits of TBH4 hodoscope fibers in appropriate
//              container
// Use in your sensitive detector builder:
// $Id: EcalTBH4BeamSD.h,v 1.4 2008/05/12 09:28:41 sunanda Exp $
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "Geometry/EcalCommonData/interface/EcalNumberingScheme.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4String.hh"
#include <map>

class EcalBaseNumber;

class EcalTBH4BeamSD : public CaloSD {

public:    

  EcalTBH4BeamSD(G4String, const DDCompactView &, SensitiveDetectorCatalog &,
		 edm::ParameterSet const &, const SimTrackManager*);
  virtual ~EcalTBH4BeamSD();
  virtual double getEnergyDeposit(G4Step*);
  virtual uint32_t setDetUnitId(G4Step* step);
  void setNumberingScheme(EcalNumberingScheme* scheme);

private:    

  void                 getBaseNumber(const G4Step* aStep); 
  EcalNumberingScheme *numberingScheme;
  bool                 useWeight;
  bool                 useBirk;
  double               birk1, birk2, birk3;
  EcalBaseNumber       theBaseNumber;

};

#endif // EcalTBH4BeamSD_h
