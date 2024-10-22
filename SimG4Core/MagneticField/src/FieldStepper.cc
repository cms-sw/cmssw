#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/MagneticField/interface/FieldStepper.h"
#include "SimG4Core/MagneticField/interface/Field.h"

#include "G4BogackiShampine45.hh"
#include "G4CashKarpRKF45.hh"
#include "G4TCashKarpRKF45.hh"
#include "G4ClassicalRK4.hh"
#include "G4TClassicalRK4.hh"
#include "G4DormandPrince745.hh"
#include "G4TDormandPrince45.hh"
#include "CMSTDormandPrince45.h"
#include "G4HelixExplicitEuler.hh"
#include "G4HelixHeum.hh"
#include "G4HelixImplicitEuler.hh"
#include "G4HelixSimpleRunge.hh"
#include "G4ImplicitEuler.hh"
#include "G4Mag_UsualEqRhs.hh"
#include "G4TMagFieldEquation.hh"
#include "CMSTMagFieldEquation.h"
#include "G4NystromRK4.hh"
#include "G4SimpleHeum.hh"
#include "G4SimpleRunge.hh"
#include "G4TsitourasRK45.hh"

FieldStepper::FieldStepper(G4Mag_UsualEqRhs *eq, double del, const std::string &nam)
    : G4MagIntegratorStepper(eq, 6), theEquation(eq), theDelta(del) {
  selectStepper(nam);
}

FieldStepper::~FieldStepper() {}

void FieldStepper::Stepper(const G4double y[], const G4double dydx[], G4double h, G4double yout[], G4double yerr[]) {
  theStepper->Stepper(y, dydx, h, yout, yerr);
}

G4double FieldStepper::DistChord() const { return theStepper->DistChord(); }

G4int FieldStepper::IntegratorOrder() const { return theStepper->IntegratorOrder(); }

void FieldStepper::selectStepper(const std::string &ss) {
  if (ss == "G4ClassicalRK4")
    theStepper = new G4ClassicalRK4(theEquation);
  else if (ss == "G4TClassicalRK4")
    theStepper = new G4TClassicalRK4<G4Mag_UsualEqRhs, 8>(theEquation);
  else if (ss == "G4NystromRK4")
    theStepper = new G4NystromRK4(theEquation, theDelta);
  else if (ss == "G4SimpleRunge")
    theStepper = new G4SimpleRunge(theEquation);
  else if (ss == "G4SimpleHeum")
    theStepper = new G4SimpleHeum(theEquation);
  else if (ss == "G4CashKarpRKF45")
    theStepper = new G4CashKarpRKF45(theEquation);
  else if (ss == "G4TCashKarpRKF45")
    theStepper = new G4TCashKarpRKF45<G4Mag_UsualEqRhs>(theEquation);
  else if (ss == "G4DormandPrince745")
    theStepper = new G4DormandPrince745(theEquation);
  else if (ss == "G4TDormandPrince45")
    theStepper = new G4TDormandPrince45<G4TMagFieldEquation<sim::Field>>(
        dynamic_cast<G4TMagFieldEquation<sim::Field> *>(theEquation));
  else if (ss == "CMSTDormandPrince45")
    theStepper = new CMSTDormandPrince45<CMSTMagFieldEquation<sim::Field>>(
        dynamic_cast<CMSTMagFieldEquation<sim::Field> *>(theEquation));
  else if (ss == "G4BogackiShampine45")
    theStepper = new G4BogackiShampine45(theEquation);
  else if (ss == "G4TsitourasRK45")
    theStepper = new G4TsitourasRK45(theEquation);
  else if (ss == "G4ImplicitEuler")
    theStepper = new G4ImplicitEuler(theEquation);
  else if (ss == "G4HelixExplicitEuler")
    theStepper = new G4HelixExplicitEuler(theEquation);
  else if (ss == "G4HelixImplicitEuler")
    theStepper = new G4HelixImplicitEuler(theEquation);
  else if (ss == "G4HelixSimpleRunge")
    theStepper = new G4HelixSimpleRunge(theEquation);
  else if (ss == "G4HelixHeum")
    theStepper = new G4HelixHeum(theEquation);
  else {
    edm::LogWarning("SimG4CoreMagneticField")
        << " FieldStepper <" << ss << "> is not known, defaulting to G4ClassicalRK4 ";
    theStepper = new G4ClassicalRK4(theEquation);
  }
  edm::LogVerbatim("SimG4CoreMagneticField") << "### FieldStepper: <" << ss << ">";
}
