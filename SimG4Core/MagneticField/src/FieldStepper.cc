#include "SimG4Core/MagneticField/interface/FieldStepper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Mag_UsualEqRhs.hh"
#include "G4ClassicalRK4.hh"
#include "G4SimpleRunge.hh"
#include "G4SimpleHeum.hh"
#include "G4CashKarpRKF45.hh"
#include "G4RKG3_Stepper.hh"
#include "G4ExplicitEuler.hh"
#include "G4ImplicitEuler.hh"
#include "G4HelixExplicitEuler.hh"
#include "G4HelixImplicitEuler.hh"
#include "G4HelixSimpleRunge.hh"
#include "G4HelixHeum.hh"
#include "G4NystromRK4.hh"

using namespace sim;

FieldStepper::FieldStepper(G4Mag_UsualEqRhs * eq, double del) :
  G4MagIntegratorStepper(eq, 6), theEquation(eq), delta(del) {}

FieldStepper::~FieldStepper() {}

void FieldStepper::Stepper(const double y[],const double dydx[],double h,
			      double yout[],double yerr[])
{ theStepper->Stepper(y,dydx,h,yout,yerr); }

double FieldStepper::DistChord() const { return theStepper->DistChord(); }

int FieldStepper::IntegratorOrder() const
{ return theStepper->IntegratorOrder(); }

G4MagIntegratorStepper * FieldStepper::select(const std::string & ss)
{
    if      (ss == "G4ClassicalRK4")       theStepper = new G4ClassicalRK4(theEquation);
    else if (ss == "G4NystromRK4")         theStepper = new G4NystromRK4(theEquation, delta);
    else if (ss == "G4SimpleRunge")        theStepper = new G4SimpleRunge(theEquation);
    else if (ss == "G4SimpleHeum")         theStepper = new G4SimpleHeum(theEquation);
    else if (ss == "G4CashKarpRKF45")      theStepper = new G4CashKarpRKF45(theEquation);
    else if (ss == "G4RKG3_Stepper")       theStepper = new G4RKG3_Stepper(theEquation);
    else if (ss == "G4ExplicitEuler")      theStepper = new G4ExplicitEuler(theEquation);
    else if (ss == "G4ImplicitEuler")      theStepper = new G4ImplicitEuler(theEquation);
    else if (ss == "G4HelixExplicitEuler") theStepper = new G4HelixExplicitEuler(theEquation);
    else if (ss == "G4HelixImplicitEuler") theStepper = new G4HelixImplicitEuler(theEquation);
    else if (ss == "G4HelixSimpleRunge")   theStepper = new G4HelixSimpleRunge(theEquation);
    else if (ss == "G4HelixHeum")          theStepper = new G4HelixHeum(theEquation);
    else
    {
      edm::LogWarning("SimG4CoreMagneticField") 
        << " FieldStepper invalid choice, defaulting to G4ClassicalRK4 ";
      theStepper = new G4ClassicalRK4(theEquation);
    }
    return theStepper;
}
