#ifndef Forward_CastorSD_h
#define Forward_CastorSD_h
// -*- C++ -*-
//
// Package:     Forward
// Class  :     CastorSD
//
/**\class CastorSD CastorSD.h SimG4CMS/Forward/interface/CastorSD.h
 
 Description: Stores hits of Castor in appropriate  container
 
 Usage:
    Used in sensitive detector builder 
 
*/
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: CastorSD.h,v 1.3 2006/05/17 16:18:57 sunanda Exp $
//
 
// system include files

// user include files

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Forward/interface/CastorNumberingScheme.h"

class CastorSD : public CaloSD {

public:    

  CastorSD(G4String, const DDCompactView &, SensitiveDetectorCatalog & clg, 
	   edm::ParameterSet const &, const SimTrackManager*);
  virtual ~CastorSD();
  virtual double   getEnergyDeposit(G4Step* );
  virtual uint32_t setDetUnitId(G4Step* step);
  void             setNumberingScheme(CastorNumberingScheme* scheme);

private:    
  double curve_Castor(G4String& , G4StepPoint*); 
  CastorNumberingScheme * numberingScheme;
};

#endif // CastorSD_h
