//
//---------------------------------------------------------------
//
//  GFlashEMShowerModel
//
//  Class description:
//
//  GFlash parameterisation shower model.

// Authors: E.Barberio & Joanna Weng - 9.11.04
// other authors : Soon Yung Jun & Dongwook Jang - 2007/12/07
//                 V.Ivanchenko rename the class and move
//                 to SimG4Core/Application - 2012/08/14
//---------------------------------------------------------------

#ifndef GFlashEMShowerModel_h
#define GFlashEMShowerModel_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4VFastSimulationModel.hh"
#include "G4TouchableHandle.hh"
#include "G4Navigator.hh"
#include "G4Step.hh"

class GflashEMShowerProfile;
class G4Region;

class GFlashEMShowerModel : public G4VFastSimulationModel {

public:
  
  GFlashEMShowerModel (const G4String& name, G4Envelope* env, 
		       const edm::ParameterSet& parSet);
  virtual ~GFlashEMShowerModel ();  

  G4bool ModelTrigger(const G4FastTrack &); 
  G4bool IsApplicable(const G4ParticleDefinition&);
  void DoIt(const G4FastTrack&, G4FastStep&);

private:

  G4bool excludeDetectorRegion(const G4FastTrack& fastTrack);
  void makeHits(const G4FastTrack& fastTrack);
  void updateGflashStep(const G4ThreeVector& position, G4double time);

  //
  edm::ParameterSet theParSet;
  bool theWatcherOn;

  GflashEMShowerProfile *theProfile;

  const G4Region* theRegion;

  G4Step *theGflashStep;
  G4Navigator *theGflashNavigator;
  G4TouchableHandle  theGflashTouchableHandle;

};
#endif
