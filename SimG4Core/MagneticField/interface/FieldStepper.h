#ifndef SimG4Core_FieldStepper_H
#define SimG4Core_FieldStepper_H

#include "G4MagIntegratorStepper.hh"

class G4Mag_UsualEqRhs;

namespace sim {
   class FieldStepper : public G4MagIntegratorStepper
   {
      public:
     FieldStepper(G4Mag_UsualEqRhs * eq, double del = 0.0);
	 ~FieldStepper() override;
	 void Stepper(const double y[],const double dydx[],double h,
			      double yout[],double yerr[]) override;
	 double DistChord() const override;
	 int IntegratorOrder() const override;
	 G4MagIntegratorStepper * select(const std::string & s);
      private:
	 G4MagIntegratorStepper * theStepper;
	 G4Mag_UsualEqRhs * theEquation;  
         double delta;
   };
}

#endif
