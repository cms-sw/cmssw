#ifndef Forward_CastorSensitiveDetector_h
#define Forward_CastorSensitiveDetector_h
// -*- C++ -*-
//
// Package:     Forward
// Class  :     CastorSensitiveDetector
//
/**\class CastorSensitiveDetector CastorSensitiveDetector.h SimG4CMS/Forward/interface/CastorSensitiveDetector.h
 
 Description: Stores hits of Castor in appropriate  container
 
 Usage:
    Used in sensitive detector builder 
 
*/
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006

// system include files

// user include files

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Forward/interface/CastorShowerLibrary.h"
#include "SimG4CMS/Forward/interface/CastorNumberingScheme.h"
#include "SimDataFormats/CaloHit/interface/CastorShowerEvent.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "G4LogicalVolume.hh"

class CastorSensitiveDetector : public CaloSD {
public:
  CastorSensitiveDetector(const std::string &,
			  const SensitiveDetectorCatalog &clg,
			  edm::ParameterSet const &,
			  const SimTrackManager *);
  ~CastorSensitiveDetector() override;
  uint32_t setDetUnitId(const G4Step *step) override;
  void setNumberingScheme(CastorNumberingScheme *scheme);

protected:
  double getEnergyDeposit(const G4Step *) override;
  bool getFromLibrary(const G4Step *) override;

private:
  uint32_t rotateUnitID(uint32_t, const G4Track *, const CastorShowerEvent &);
  CastorNumberingScheme *numberingScheme;
  CastorShowerLibrary *showerLibrary;
  G4LogicalVolume *lvC3EF, *lvC3HF, *lvC4EF, *lvC4HF;
  G4LogicalVolume *lvCAST;  // Pointer for CAST sensitive volume  (SL trigger)

  bool useShowerLibrary;
  double energyThresholdSL;
  double non_compensation_factor;
};

#endif  // CastorSensitiveDetector_h
