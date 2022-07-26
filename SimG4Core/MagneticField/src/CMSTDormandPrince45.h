//
// copied from G4TDormandPrince45
//
// Class desription:
//
//  An implementation of the 5th order embedded RK method from the paper:
//  J. R. Dormand and P. J. Prince, "A family of embedded Runge-Kutta formulae"
//  Journal of computational and applied Math., vol.6, no.1, pp.19-26, 1980.
//
//  DormandPrince7 - 5(4) embedded RK method
//
//
// Created: Somnath Banerjee, Google Summer of Code 2015, 25 May 2015
// Supervision: John Apostolakis, CERN
//
//  Modified by Vincenzo Innocente on 7/2022 to account for constant momentum magnitude
//
// --------------------------------------------------------------------
#ifndef CMSTDORMAND_PRINCE_45_HH
#define CMSTDORMAND_PRINCE_45_HH

#include <cassert>
#include "G4MagIntegratorStepper.hh"
#include "G4FieldUtils.hh"

template <class T_Equation, unsigned int N = 6>
class CMSTDormandPrince45 : public G4MagIntegratorStepper {
public:
  CMSTDormandPrince45(T_Equation* equation);
  CMSTDormandPrince45(T_Equation* equation, G4int numVar);  // must have numVar == N

  inline void StepWithError(
      const G4double yInput[], const G4double dydx[], G4double hstep, G4double yOutput[], G4double yError[]);

  void Stepper(
      const G4double yInput[], const G4double dydx[], G4double hstep, G4double yOutput[], G4double yError[]) final;

  inline void StepWithFinalDerivate(const G4double yInput[],
                                    const G4double dydx[],
                                    G4double hstep,
                                    G4double yOutput[],
                                    G4double yError[],
                                    G4double dydxOutput[]);

  inline void SetupInterpolation() {}

  void Interpolate(G4double tau, G4double yOut[]) const { Interpolate4thOrder(yOut, tau); }
  // For calculating the output at the tau fraction of Step

  G4double DistChord() const final;

  G4int IntegratorOrder() const override { return 4; }

  const field_utils::ShortState<N>& GetYOut() const { return fyOut; }

  void Interpolate4thOrder(G4double yOut[], G4double tau) const;

  void SetupInterpolation5thOrder();
  void Interpolate5thOrder(G4double yOut[], G4double tau) const;

  // __attribute__((always_inline))
  void RightHandSideInl(const G4double y[], G4double inv_momentum_magnitude, G4double dydx[]) {
    fEquation_Rhs->T_Equation::TRightHandSide(y, inv_momentum_magnitude, dydx);
  }

  inline void Stepper(const G4double yInput[],
                      const G4double dydx[],
                      G4double hstep,
                      G4double yOutput[],
                      G4double yError[],
                      G4double dydxOutput[]) {
    StepWithFinalDerivate(yInput, dydx, hstep, yOutput, yError, dydxOutput);
  }

  T_Equation* GetSpecificEquation() { return fEquation_Rhs; }

  static constexpr int N8 = N > 8 ? N : 8;  //  y[

private:
  field_utils::ShortState<N> ak2, ak3, ak4, ak5, ak6, ak7, ak8, ak9;
  field_utils::ShortState<N8> fyIn;
  field_utils::ShortState<N> fyOut, fdydxIn;

  // - Simpler :
  // field_utils::State ak2, ak3, ak4, ak5, ak6, ak7, ak8, ak9;
  // field_utils::State fyIn, fyOut, fdydxIn;

  G4double fLastStepLength = -1.0;
  T_Equation* fEquation_Rhs;
};

// G4TDormandPrince745 implementation -- borrowed from G4DormandPrince745
//
// DormandPrince7 - 5(4) non-FSAL
// definition of the stepper() method that evaluates one step in
// field propagation.
// The coefficients and the algorithm have been adapted from
//
// J. R. Dormand and P. J. Prince, "A family of embedded Runge-Kutta formulae"
// Journal of computational and applied Math., vol.6, no.1, pp.19-26, 1980.
//
// The Butcher table of the Dormand-Prince-7-4-5 method is as follows :
//
//    0   |
//    1/5 | 1/5
//    3/10| 3/40       9/40
//    4/5 | 44/45      56/15      32/9
//    8/9 | 19372/6561 25360/2187 64448/6561  212/729
//    1   | 9017/3168  355/33     46732/5247  49/176  5103/18656
//    1   | 35/384     0          500/1113    125/192 2187/6784    11/84
//    ------------------------------------------------------------------------
//          35/384     0          500/1113    125/192 2187/6784    11/84    0
//          5179/57600 0          7571/16695  393/640 92097/339200 187/2100 1/40
//
// Created: Somnath Banerjee, Google Summer of Code 2015, 25 May 2015
// Supervision: John Apostolakis, CERN
// --------------------------------------------------------------------

#include "G4LineSection.hh"

#include <cstring>

// using namespace field_utils;

/////////////////////////////////////////////////////////////////////
// Constructor
//
template <class T_Equation, unsigned int N>
CMSTDormandPrince45<T_Equation, N>::CMSTDormandPrince45(T_Equation* equation)
    : G4MagIntegratorStepper(dynamic_cast<G4EquationOfMotion*>(equation), N), fEquation_Rhs(equation) {
  // assert( dynamic_cast<G4EquationOfMotion*>(equation) != nullptr );
  if (dynamic_cast<G4EquationOfMotion*>(equation) == nullptr) {
    G4Exception("G4TDormandPrince745CMS: constructor",
                "GeomField0001",
                FatalException,
                "T_Equation is not an G4EquationOfMotion.");
  }

  /***
  assert( equation->GetNumberOfVariables == N );
  if( equation->GetNumberOfVariables != N ){
    G4ExceptionDescription msg;
    msg << "Equation has an incompatible number of variables." ;
    msg << "   template N = " << N << " equation-Nvar= "
        << equation->GetNumberOfVariables;
    G4Exception("G4TCashKarpRKF45: constructor", "GeomField0001",
                FatalException, msg );
   } ****/
}

template <class T_Equation, unsigned int N>
CMSTDormandPrince45<T_Equation, N>::CMSTDormandPrince45(T_Equation* equation, G4int numVar)
    : CMSTDormandPrince45<T_Equation, N>(equation) {
  if (numVar != G4int(N)) {
    G4ExceptionDescription msg;
    msg << "Equation has an incompatible number of variables.";
    msg << "   template N = " << N << "   argument numVar = " << numVar;
    //    << " equation-Nvar= " << equation->GetNumberOfVariables(); // --> Expected later
    G4Exception("G4TCashKarpRKF45CMS: constructor", "GeomField0001", FatalErrorInArgument, msg);
  }
  assert(numVar == N);
}

template <class T_Equation, unsigned int N>
inline void CMSTDormandPrince45<T_Equation, N>::StepWithFinalDerivate(const G4double yInput[],
                                                                      const G4double dydx[],
                                                                      G4double hstep,
                                                                      G4double yOutput[],
                                                                      G4double yError[],
                                                                      G4double dydxOutput[]) {
  StepWithError(yInput, dydx, hstep, yOutput, yError);
  field_utils::copy(dydxOutput, ak7, N);
}

// Stepper
//
// Passing in the value of yInput[],the first time dydx[] and Step length
// Giving back yOut and yErr arrays for output and error respectively
//

template <class T_Equation, unsigned int N>
inline void CMSTDormandPrince45<T_Equation, N>::StepWithError(
    const G4double yInput[], const G4double dydx[], G4double hstep, G4double yOut[], G4double yErr[]) {
  // The parameters of the Butcher tableu
  //
  constexpr G4double b21 = 0.2, b31 = 3.0 / 40.0, b32 = 9.0 / 40.0, b41 = 44.0 / 45.0, b42 = -56.0 / 15.0,
                     b43 = 32.0 / 9.0,

                     b51 = 19372.0 / 6561.0, b52 = -25360.0 / 2187.0, b53 = 64448.0 / 6561.0, b54 = -212.0 / 729.0,

                     b61 = 9017.0 / 3168.0, b62 = -355.0 / 33.0, b63 = 46732.0 / 5247.0, b64 = 49.0 / 176.0,
                     b65 = -5103.0 / 18656.0,

                     b71 = 35.0 / 384.0, b72 = 0., b73 = 500.0 / 1113.0, b74 = 125.0 / 192.0, b75 = -2187.0 / 6784.0,
                     b76 = 11.0 / 84.0,

                     //Sum of columns, sum(bij) = ei
      //    e1 = 0. ,
      //    e2 = 1.0/5.0 ,
      //    e3 = 3.0/10.0 ,
      //    e4 = 4.0/5.0 ,
      //    e5 = 8.0/9.0 ,
      //    e6 = 1.0 ,
      //    e7 = 1.0 ,

      // Difference between the higher and the lower order method coeff. :
      // b7j are the coefficients of higher order

      dc1 = -(b71 - 5179.0 / 57600.0), dc2 = -(b72 - .0), dc3 = -(b73 - 7571.0 / 16695.0), dc4 = -(b74 - 393.0 / 640.0),
                     dc5 = -(b75 + 92097.0 / 339200.0), dc6 = -(b76 - 187.0 / 2100.0), dc7 = -(-1.0 / 40.0);

  // const G4int numberOfVariables = GetNumberOfVariables();
  //   The number of variables to be integrated over
  field_utils::ShortState<N8> yTemp;

  yOut[7] = yTemp[7] = fyIn[7] = yInput[7];  // Pass along the time - used in RightHandSide

  //  Saving yInput because yInput and yOut can be aliases for same array
  //
  for (unsigned int i = 0; i < N; ++i) {
    fyIn[i] = yInput[i];
    yTemp[i] = yInput[i] + b21 * hstep * dydx[i];
  }
  G4double momentum_mag_square = yTemp[3] * yTemp[3] + yTemp[4] * yTemp[4] + yTemp[5] * yTemp[5];
  G4double inv_momentum_magnitude = 1.0 / std::sqrt(momentum_mag_square);
  RightHandSideInl(yTemp, inv_momentum_magnitude, ak2);  // 2nd stage

  for (unsigned int i = 0; i < N; ++i) {
    yTemp[i] = fyIn[i] + hstep * (b31 * dydx[i] + b32 * ak2[i]);
  }
  RightHandSideInl(yTemp, inv_momentum_magnitude, ak3);  // 3rd stage

  for (unsigned int i = 0; i < N; ++i) {
    yTemp[i] = fyIn[i] + hstep * (b41 * dydx[i] + b42 * ak2[i] + b43 * ak3[i]);
  }
  RightHandSideInl(yTemp, inv_momentum_magnitude, ak4);  // 4th stage

  for (unsigned int i = 0; i < N; ++i) {
    yTemp[i] = fyIn[i] + hstep * (b51 * dydx[i] + b52 * ak2[i] + b53 * ak3[i] + b54 * ak4[i]);
  }
  RightHandSideInl(yTemp, inv_momentum_magnitude, ak5);  // 5th stage

  for (unsigned int i = 0; i < N; ++i) {
    yTemp[i] = fyIn[i] + hstep * (b61 * dydx[i] + b62 * ak2[i] + b63 * ak3[i] + b64 * ak4[i] + b65 * ak5[i]);
  }
  RightHandSideInl(yTemp, inv_momentum_magnitude, ak6);  // 6th stage

  for (unsigned int i = 0; i < N; ++i) {
    yOut[i] =
        fyIn[i] + hstep * (b71 * dydx[i] + b72 * ak2[i] + b73 * ak3[i] + b74 * ak4[i] + b75 * ak5[i] + b76 * ak6[i]);
  }
  RightHandSideInl(yOut, inv_momentum_magnitude, ak7);  // 7th and Final stage

  for (unsigned int i = 0; i < N; ++i) {
    yErr[i] = hstep * (dc1 * dydx[i] + dc2 * ak2[i] + dc3 * ak3[i] + dc4 * ak4[i] + dc5 * ak5[i] + dc6 * ak6[i] +
                       dc7 * ak7[i]) +
              1.5e-18;

    // Store Input and Final values, for possible use in calculating chord
    //
    fyOut[i] = yOut[i];
    fdydxIn[i] = dydx[i];
  }

  fLastStepLength = hstep;
}

template <class T_Equation, unsigned int N>
inline void CMSTDormandPrince45<T_Equation, N>::Stepper(
    const G4double yInput[], const G4double dydx[], G4double Step, G4double yOutput[], G4double yError[]) {
  assert(yOutput != yInput);
  assert(yError != yInput);

  StepWithError(yInput, dydx, Step, yOutput, yError);
}

template <class T_Equation, unsigned int N>
G4double CMSTDormandPrince45<T_Equation, N>::DistChord() const {
  // Coefficients were taken from Some Practical Runge-Kutta Formulas
  // by Lawrence F. Shampine, page 149, c*
  //
  const G4double hf1 = 6025192743.0 / 30085553152.0, hf3 = 51252292925.0 / 65400821598.0,
                 hf4 = -2691868925.0 / 45128329728.0, hf5 = 187940372067.0 / 1594534317056.0,
                 hf6 = -1776094331.0 / 19743644256.0, hf7 = 11237099.0 / 235043384.0;

  G4ThreeVector mid;

  for (unsigned int i = 0; i < 3; ++i) {
    mid[i] =
        fyIn[i] + 0.5 * fLastStepLength *
                      (hf1 * fdydxIn[i] + hf3 * ak3[i] + hf4 * ak4[i] + hf5 * ak5[i] + hf6 * ak6[i] + hf7 * ak7[i]);
  }

  const G4ThreeVector begin = makeVector(fyIn, field_utils::Value3D::Position);
  const G4ThreeVector end = makeVector(fyOut, field_utils::Value3D::Position);

  return G4LineSection::Distline(mid, begin, end);
}

// The lower (4th) order interpolant given by Dormand and Prince:
//        J. R. Dormand and P. J. Prince, "Runge-Kutta triples"
//        Computers & Mathematics with Applications, vol. 12, no. 9,
//        pp. 1007-1017, 1986.
//
template <class T_Equation, unsigned int N>
void CMSTDormandPrince45<T_Equation, N>::Interpolate4thOrder(G4double yOut[], G4double tau) const {
  // const G4int numberOfVariables = GetNumberOfVariables();

  const G4double tau2 = tau * tau, tau3 = tau * tau2, tau4 = tau2 * tau2;

  const G4double bf1 =
      1.0 / 11282082432.0 *
      (157015080.0 * tau4 - 13107642775.0 * tau3 + 34969693132.0 * tau2 - 32272833064.0 * tau + 11282082432.0);

  const G4double bf3 =
      -100.0 / 32700410799.0 * tau * (15701508.0 * tau3 - 914128567.0 * tau2 + 2074956840.0 * tau - 1323431896.0);

  const G4double bf4 =
      25.0 / 5641041216.0 * tau * (94209048.0 * tau3 - 1518414297.0 * tau2 + 2460397220.0 * tau - 889289856.0);

  const G4double bf5 =
      -2187.0 / 199316789632.0 * tau * (52338360.0 * tau3 - 451824525.0 * tau2 + 687873124.0 * tau - 259006536.0);

  const G4double bf6 =
      11.0 / 2467955532.0 * tau * (106151040.0 * tau3 - 661884105.0 * tau2 + 946554244.0 * tau - 361440756.0);

  const G4double bf7 = 1.0 / 29380423.0 * tau * (1.0 - tau) * (8293050.0 * tau2 - 82437520.0 * tau + 44764047.0);

  for (unsigned int i = 0; i < N; ++i) {
    yOut[i] =
        fyIn[i] + fLastStepLength * tau *
                      (bf1 * fdydxIn[i] + bf3 * ak3[i] + bf4 * ak4[i] + bf5 * ak5[i] + bf6 * ak6[i] + bf7 * ak7[i]);
  }
}

// Following interpolant of order 5 was given by Baker,Dormand,Gilmore, Prince :
//        T. S. Baker, J. R. Dormand, J. P. Gilmore, and P. J. Prince,
//        "Continuous approximation with embedded Runge-Kutta methods"
//        Applied Numerical Mathematics, vol. 22, no. 1, pp. 51-62, 1996.
//
// Calculating the extra stages for the interpolant
//
template <class T_Equation, unsigned int N>
void CMSTDormandPrince45<T_Equation, N>::SetupInterpolation5thOrder() {
  // Coefficients for the additional stages
  //
  const G4double b81 = 6245.0 / 62208.0, b82 = 0.0, b83 = 8875.0 / 103032.0, b84 = -125.0 / 1728.0,
                 b85 = 801.0 / 13568.0, b86 = -13519.0 / 368064.0, b87 = 11105.0 / 368064.0,

                 b91 = 632855.0 / 4478976.0, b92 = 0.0, b93 = 4146875.0 / 6491016.0, b94 = 5490625.0 / 14183424.0,
                 b95 = -15975.0 / 108544.0, b96 = 8295925.0 / 220286304.0, b97 = -1779595.0 / 62938944.0,
                 b98 = -805.0 / 4104.0;

  // const G4int numberOfVariables = GetNumberOfVariables();
  field_utils::ShortState<N> yTemp;

  // Evaluate the extra stages
  //
  for (unsigned int i = 0; i < N; ++i) {
    yTemp[i] = fyIn[i] + fLastStepLength * (b81 * fdydxIn[i] + b82 * ak2[i] + b83 * ak3[i] + b84 * ak4[i] +
                                            b85 * ak5[i] + b86 * ak6[i] + b87 * ak7[i]);
  }
  G4double momentum_mag_square = yTemp[3] * yTemp[3] + yTemp[4] * yTemp[4] + yTemp[5] * yTemp[5];
  G4double inv_momentum_magnitude = 1.0 / std::sqrt(momentum_mag_square);
  RightHandSideInl(yTemp, inv_momentum_magnitude, ak8);  // 8th Stage

  for (unsigned int i = 0; i < N; ++i) {
    yTemp[i] = fyIn[i] + fLastStepLength * (b91 * fdydxIn[i] + b92 * ak2[i] + b93 * ak3[i] + b94 * ak4[i] +
                                            b95 * ak5[i] + b96 * ak6[i] + b97 * ak7[i] + b98 * ak8[i]);
  }
  RightHandSideInl(yTemp, inv_momentum_magnitude, ak9);  // 9th Stage
}

// Calculating the interpolated result yOut with the coefficients
//
template <class T_Equation, unsigned int N>
void CMSTDormandPrince45<T_Equation, N>::Interpolate5thOrder(G4double yOut[], G4double tau) const {
  // Define the coefficients for the polynomials
  //
  G4double bi[10][5];

  //  COEFFICIENTS OF   bi[1]
  bi[1][0] = 1.0, bi[1][1] = -38039.0 / 7040.0, bi[1][2] = 125923.0 / 10560.0, bi[1][3] = -19683.0 / 1760.0,
  bi[1][4] = 3303.0 / 880.0,
  //  --------------------------------------------------------
      //
      //  COEFFICIENTS OF  bi[2]
      bi[2][0] = 0.0, bi[2][1] = 0.0, bi[2][2] = 0.0, bi[2][3] = 0.0, bi[2][4] = 0.0,
  //  --------------------------------------------------------
      //
      //  COEFFICIENTS OF  bi[3]
      bi[3][0] = 0.0, bi[3][1] = -12500.0 / 4081.0, bi[3][2] = 205000.0 / 12243.0, bi[3][3] = -90000.0 / 4081.0,
  bi[3][4] = 36000.0 / 4081.0,
  //  --------------------------------------------------------
      //
      //  COEFFICIENTS OF  bi[4]
      bi[4][0] = 0.0, bi[4][1] = -3125.0 / 704.0, bi[4][2] = 25625.0 / 1056.0, bi[4][3] = -5625.0 / 176.0,
  bi[4][4] = 1125.0 / 88.0,
  //  --------------------------------------------------------
      //
      //  COEFFICIENTS OF  bi[5]
      bi[5][0] = 0.0, bi[5][1] = 164025.0 / 74624.0, bi[5][2] = -448335.0 / 37312.0, bi[5][3] = 295245.0 / 18656.0,
  bi[5][4] = -59049.0 / 9328.0,
  //  --------------------------------------------------------
      //
      //  COEFFICIENTS OF  bi[6]
      bi[6][0] = 0.0, bi[6][1] = -25.0 / 28.0, bi[6][2] = 205.0 / 42.0, bi[6][3] = -45.0 / 7.0, bi[6][4] = 18.0 / 7.0,
  //  --------------------------------------------------------
      //
      //  COEFFICIENTS OF  bi[7]
      bi[7][0] = 0.0, bi[7][1] = -2.0 / 11.0, bi[7][2] = 73.0 / 55.0, bi[7][3] = -171.0 / 55.0, bi[7][4] = 108.0 / 55.0,
  //  --------------------------------------------------------
      //
      //  COEFFICIENTS OF  bi[8]
      bi[8][0] = 0.0, bi[8][1] = 189.0 / 22.0, bi[8][2] = -1593.0 / 55.0, bi[8][3] = 3537.0 / 110.0,
  bi[8][4] = -648.0 / 55.0,
  //  --------------------------------------------------------
      //
      //  COEFFICIENTS OF  bi[9]
      bi[9][0] = 0.0, bi[9][1] = 351.0 / 110.0, bi[9][2] = -999.0 / 55.0, bi[9][3] = 2943.0 / 110.0,
  bi[9][4] = -648.0 / 55.0;
  //  --------------------------------------------------------

  // Calculating the polynomials

  G4double b[10];
  std::memset(b, 0.0, sizeof(b));

  G4double tauPower = 1.0;
  for (G4int j = 0; j <= 4; ++j) {
    for (G4int iStage = 1; iStage <= 9; ++iStage) {
      b[iStage] += bi[iStage][j] * tauPower;
    }
    tauPower *= tau;
  }

  // const G4int numberOfVariables = GetNumberOfVariables();
  const G4double stepLen = fLastStepLength * tau;
  for (G4int i = 0; i < N; ++i) {
    yOut[i] = fyIn[i] + stepLen * (b[1] * fdydxIn[i] + b[2] * ak2[i] + b[3] * ak3[i] + b[4] * ak4[i] + b[5] * ak5[i] +
                                   b[6] * ak6[i] + b[7] * ak7[i] + b[8] * ak8[i] + b[9] * ak9[i]);
  }
}

#endif
