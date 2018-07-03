#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/CMSFieldManager.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/FieldStepper.h"
#include "SimG4Core/MagneticField/interface/MonopoleEquation.h"

#include "G4Mag_UsualEqRhs.hh"
#include "G4ClassicalRK4.hh"
#include "G4PropagatorInField.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4ChordFinder.hh"
#include "G4UniformMagField.hh"
#include "G4LogicalVolumeStore.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

using namespace sim;

FieldBuilder::FieldBuilder(const MagneticField * f, const edm::ParameterSet & p) 
  : theTopVolume(nullptr),thePSet(p) 
{
  delta = p.getParameter<double>("delta")*CLHEP::mm;
  theField = new Field(f, delta);
  theFieldEquation = new G4Mag_UsualEqRhs(theField);
}

FieldBuilder::~FieldBuilder()
{} 

void FieldBuilder::build( CMSFieldManager* fM, G4PropagatorInField* fP) 
{    
  edm::ParameterSet thePSetForGMFM =
    thePSet.getParameter<edm::ParameterSet>("ConfGlobalMFM");

  std::string volName = thePSetForGMFM.getParameter< std::string >("Volume");
  
  edm::ParameterSet volPSet =
    thePSetForGMFM.getParameter< edm::ParameterSet >( volName );
    
  configureForVolume( volName, volPSet, fM, fP);

  edm::LogInfo("SimG4CoreMagneticField") 
    << " FieldBuilder::build: Global magnetic field is used";
}

void FieldBuilder::configureForVolume( const std::string& volName,
                                       edm::ParameterSet& volPSet,
				       CMSFieldManager * fM,
				       G4PropagatorInField * fP) 
{
  G4LogicalVolumeStore* theStore = G4LogicalVolumeStore::GetInstance();
  for (auto vol : *theStore) {
    if ( (std::string)vol->GetName() == volName ) {
      theTopVolume = vol;
      break;
    }
  }

  std::string fieldType = volPSet.getParameter<std::string>("Type");
  std::string stepper   = volPSet.getParameter<std::string>("Stepper");

  edm::ParameterSet stpPSet = volPSet.getParameter<edm::ParameterSet>("StepperParam");
  double minStep = stpPSet.getParameter<double>("MinStep") ;
  int maxLoopCount = 
    (int)stpPSet.getUntrackedParameter<double>("MaximumLoopCounts",1000);
  double minEpsilonStep = 
    stpPSet.getUntrackedParameter<double>("MinimumEpsilonStep",0.00001);
  double maxEpsilonStep = 
    stpPSet.getUntrackedParameter<double>("MaximumEpsilonStep",0.01);

  FieldStepper * theStepper = new FieldStepper(theFieldEquation, delta);
  theStepper->select(stepper);
  G4ChordFinder * cf = new G4ChordFinder(theField,minStep,theStepper);

  MonopoleEquation* monopoleEquation = new MonopoleEquation(theField);
  G4MagIntegratorStepper* theStepperMon = new G4ClassicalRK4(monopoleEquation,8);
  G4ChordFinder * cfmon = new G4ChordFinder(theField,minStep,theStepperMon);

  fM->InitialiseForVolume(stpPSet, theField, cf, cfmon, volName, 
			  fieldType, stepper, delta, minStep); 

  if(fP) {
    fP->SetMaxLoopCount(maxLoopCount);
    fP->SetMinimumEpsilonStep(minEpsilonStep);
    fP->SetMaximumEpsilonStep(maxEpsilonStep);
    //fP->SetVerboseLevel(0);
  }
}
