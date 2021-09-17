#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/MagneticField/interface/CMSFieldManager.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/FieldStepper.h"
#include "SimG4Core/MagneticField/interface/MonopoleEquation.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "G4ChordFinder.hh"
#include "G4ClassicalRK4.hh"
#include "G4FieldManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4TMagFieldEquation.hh"
#include "G4PropagatorInField.hh"

using namespace sim;

FieldBuilder::FieldBuilder(const MagneticField *f, const edm::ParameterSet &p) : theTopVolume(nullptr), thePSet(p) {
  theDelta = p.getParameter<double>("delta") * CLHEP::mm;
  theField = new Field(f, theDelta);
  theFieldEquation = nullptr;
}

FieldBuilder::~FieldBuilder() {}

void FieldBuilder::build(CMSFieldManager *fM, G4PropagatorInField *fP) {
  edm::ParameterSet thePSetForGMFM = thePSet.getParameter<edm::ParameterSet>("ConfGlobalMFM");
  std::string volName = thePSetForGMFM.getParameter<std::string>("Volume");
  edm::ParameterSet volPSet = thePSetForGMFM.getParameter<edm::ParameterSet>(volName);

  configureForVolume(volName, volPSet, fM, fP);

  edm::LogVerbatim("SimG4CoreMagneticField") << " FieldBuilder::build: Global magnetic field is used";
}

void FieldBuilder::configureForVolume(const std::string &volName,
                                      edm::ParameterSet &volPSet,
                                      CMSFieldManager *fM,
                                      G4PropagatorInField *fP) {
  G4LogicalVolumeStore *theStore = G4LogicalVolumeStore::GetInstance();
  for (auto vol : *theStore) {
    if ((std::string)vol->GetName() == volName) {
      theTopVolume = vol;
      break;
    }
  }

  std::string fieldType = volPSet.getParameter<std::string>("Type");
  std::string stepper = volPSet.getParameter<std::string>("Stepper");

  edm::ParameterSet stpPSet = volPSet.getParameter<edm::ParameterSet>("StepperParam");
  double minStep = stpPSet.getParameter<double>("MinStep") * CLHEP::mm;

  if (stepper == "G4TDormandPrince45") {
    theFieldEquation = new G4TMagFieldEquation<Field>(theField);
  } else {
    theFieldEquation = new G4Mag_UsualEqRhs(theField);
  }

  FieldStepper *dStepper = new FieldStepper(theFieldEquation, theDelta, stepper);
  G4ChordFinder *cf = new G4ChordFinder(theField, minStep, dStepper);

  MonopoleEquation *monopoleEquation = new MonopoleEquation(theField);
  G4MagIntegratorStepper *mStepper = new G4ClassicalRK4(monopoleEquation, 8);
  G4ChordFinder *cfmon = new G4ChordFinder(theField, minStep, mStepper);

  fM->InitialiseForVolume(stpPSet, theField, cf, cfmon, volName, fieldType, stepper, theDelta, fP);
}
