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
// $Id: CastorSD.h,v 1.5 2007/06/19 22:16:43 sunanda Exp $
//
 
// system include files

// user include files

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Forward/interface/CastorShowerLibrary.h"
#include "SimG4CMS/Forward/interface/CastorNumberingScheme.h"
#include "SimDataFormats/CaloHit/interface/CastorShowerEvent.h"

class CastorSD : public CaloSD {

public:    

  CastorSD(G4String, const DDCompactView &, SensitiveDetectorCatalog & clg, 
	   edm::ParameterSet const &, const SimTrackManager*);
  virtual ~CastorSD();
  virtual double   getEnergyDeposit(G4Step* );
  virtual uint32_t setDetUnitId(G4Step* step);
  void             setNumberingScheme(CastorNumberingScheme* scheme);

private:

  void                    getFromLibrary(G4Step*);
  int                     setTrackID(G4Step*);
  uint32_t                rotateUnitID(uint32_t, G4Track*, CastorShowerEvent);
  CastorNumberingScheme * numberingScheme;
  CastorShowerLibrary *   showerLibrary;
  
  bool                    useShowerLibrary;   // Add useShowerLibrary to PSet !!! (WC)
  double                  energyThresholdSL;  // Add useShowerLibrary to PSet !!! (WC)

protected:

  virtual void            initRun();

};

#endif // CastorSD_h
