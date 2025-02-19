#include "SimG4Core/MagneticField/interface/FieldStepper.h"

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

using namespace sim;

FieldStepper::FieldStepper(G4Mag_UsualEqRhs * eq) :
    G4MagIntegratorStepper(eq, 6), theEquation(eq) {}

FieldStepper::~FieldStepper() {}

void FieldStepper::Stepper(const double y[],const double dydx[],double h,
			      double yout[],double yerr[])
{ theStepper->Stepper(y,dydx,h,yout,yerr); }

double FieldStepper::DistChord() const { return theStepper->DistChord(); }

int FieldStepper::IntegratorOrder() const
{ return theStepper->IntegratorOrder(); }

G4MagIntegratorStepper * FieldStepper::select(const std::string & s)
{
    if      (s == "G4ClassicalRK4")       theStepper = new G4ClassicalRK4(theEquation);
    else if (s == "G4SimpleRunge")        theStepper = new G4SimpleRunge(theEquation);
    else if (s == "G4SimpleHeum")         theStepper = new G4SimpleHeum(theEquation);
    else if (s == "G4CashKarpRKF45")      theStepper = new G4CashKarpRKF45(theEquation);
    else if (s == "G4RKG3_Stepper")       theStepper = new G4RKG3_Stepper(theEquation);
    else if (s == "G4ExplicitEuler")      theStepper = new G4ExplicitEuler(theEquation);
    else if (s == "G4ImplicitEuler")      theStepper = new G4ImplicitEuler(theEquation);
    else if (s == "G4HelixExplicitEuler") theStepper = new G4HelixExplicitEuler(theEquation);
    else if (s == "G4HelixImplicitEuler") theStepper = new G4HelixImplicitEuler(theEquation);
    else if (s == "G4HelixSimpleRunge")   theStepper = new G4HelixSimpleRunge(theEquation);
    else if (s == "G4HelixHeum")          theStepper = new G4HelixHeum(theEquation);
    else
    {
        std::cout << " FieldStepper invalid choice, defaulting to G4ClassicalRK4 " << std::endl;
        theStepper = new G4ClassicalRK4(theEquation);
    }
    return theStepper;
}
