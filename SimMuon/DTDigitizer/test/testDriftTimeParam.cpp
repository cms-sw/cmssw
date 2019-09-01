/**
   \file
   Test suit for DTDigitizer

   \author Riccardo Bellan
   \version
   \date 15 Feb 2006

   \note
    Compute the DT drift time using the DT prarametrization,
    prompting the user for the parameters.

*/

#include "SimMuon/DTDigitizer/src/DTDriftTimeParametrization.cc"

#include <iostream>

void average() {
  short interpolate = 1;
  DTDriftTimeParametrization::drift_time DT;
  static const DTDriftTimeParametrization p;
  double alpha = 0;
  double Bwire = 0;
  double Bnorm = 0;

  float mean = 0;
  int count = 0;
  for (int i = 1; i < 21; i++) {
    p.MB_DT_drift_time(i, alpha, Bwire, Bnorm, 0, &DT, interpolate);
    mean += DT.v_drift;
    count++;
  }

  mean /= count;

  std::cout << " mean vd = " << mean;
}

void printDt(double x, double alpha, double Bwire, double Bnorm, int ifl) {
  short interpolate = 1;
  //  DRIFT_TIME * DT;
  DTDriftTimeParametrization::drift_time DT;
  static const DTDriftTimeParametrization p;
  unsigned short status = p.MB_DT_drift_time(x, alpha, Bwire, Bnorm, ifl, &DT, interpolate);

  std::cout << "(x = " << x << ", alpha = " << alpha << ", Bwire = " << Bwire << ", Bnorm = " << Bnorm
            << ", ifl = " << ifl << "):" << std::endl
            << "\tt_drift (ns) = " << DT.t_drift << " t_width_m = " << DT.t_width_m << " t_width_p = " << DT.t_width_p
            << " v_drift (ns) = " << DT.v_drift << " status " << status << std::endl;

}  // end printDt()

int main() {
  average();

  std::cout << "Enter x(mm), alpha(deg), Bwire(T), Bnorm(T), ifl(0=x from "
               "wire; 1=x fom cell edge)"
            << std::endl;

  while (true) {
    double x = 0, alpha = 0, Bwire = 0, Bnorm = 0;
    int ifl = 1;
    std::cin >> x >> alpha >> Bwire >> Bnorm >> ifl;
    if (!std::cin)
      break;
    printDt(x, alpha, Bwire, Bnorm, ifl);
  }
}
