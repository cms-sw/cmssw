//
// $Id: Resolution.h,v 1.2 2012/03/19 18:15:04 vadler Exp $
//
// File: hitfit/Resolution.h
// Purpose: Calculate resolutions for a quantity.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// This object will calculate resolutions for some quantity.
// We have three parameters:
//
//   C - constant term
//   R - resolution term
//   N - noise term
//
// Given a `momentum' p, we calculate the uncertainty in a quantity x as
//
//   sigma(x) = sqrt (C^2 p^2 + R^2 p + N^2)
//
// In addition, we have an `inverse' flag.  If that is set,
// we take the inverse of p before doing the above calculation
// (and for p, `sigma(p)' is regarded as actually sigma(1/p)).
//
// We encode the resolution parameters into a string, from which these
// objects get initialized.  The format is
//
//    [-]C[,R[,N]]
//
// If a leading minus sign is present, that turns on the invert flag.
// Omitted parameters are set to 0.
//
// CMSSW File      : interface/Resolution.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Resolution.h

    @brief Calculate and represent resolution for a physical quantity.

    @author Scott Stuart Snyder <snyder@bnl.gov>

    @par Creation date:
    Jul 2000.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Nov 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent for the original author (Scott Snyder).

 */

#ifndef HITFIT_RESOLUTION_H
#define HITFIT_RESOLUTION_H


#include <string>
#include <iosfwd>
#include "CLHEP/Random/Random.h"


namespace hitfit {


/**
    @class Resolution

    @brief Calculate and represent resolution for a physical quantity.

    This class calculate resolutions for some quantity.  In general we have
    three parameters:
    - <b>C</b> constant term.
    - <b>R</b> resolution term.
    - <b>N</b> noise term.

    Given a physical quantitiy \f$p\f$, we calculate the uncertainty
    in quantity \f$x\f$ as:

    \f[
    \sigma(x) = \sqrt{C^{2}p^{2} + R^{2}p + N^{2}}
    \f]

    In addition, we also have an inverse flag.  If the flag is set, we
    take the inverse of \f$p\f$ before doing the calculations.
    Therefore \f$\sigma(x)\f$ is regarded as actually \f$\sigma(1/x)\f$.

    We encode he resolution parameters into a string, from which these
    objects get initialized.  The format is

    \verbatim
[-]C[,R[,N]]
    \endverbatim

    where parameters within the brackets are optional.  If the leading minus
    is present, the inverse flag is turned on.  Omitted parameters are set
    to 0.
*/
class Resolution
//
// Purpose: Calculate resolutions for a quantity.
//
{
public:
  // Initialize from a string S.  The format is as described above.
  /**
     @brief Constructor, initialize from a string.

     @param s A string encoding the resolution parameters, as described in
     the class description.
   */
  Resolution (std::string s = "");

  /**
     @brief Constructor to initialize with four values for C, R, m, N,
     and the boolean for inverse

     @param C The constant term

     @param R The resolution term

     @param m The exponent factor term

     @param N The noise term

     @param inverse The inverse flag.
   */
  Resolution (double C,
              double R,
              double m,
              double N,
              bool inverse = false);

  // Initialize to a constant resolution RES.  I.e., sigma() will
  // always return RES.  If INVERSE is true, set the inverse flag.
  /**
     @brief Constructor to initialize a constant resolution.

     @param res The resolution value.

     @param inverse The inverse flag.
   */
  Resolution (double res, bool inverse = false);

  // Return the setting of the inverse flag.
  /**
     @brief Return the setting of the inverse flag.
   */
  bool inverse () const;

  /**
     @brief Return the C term (constant term)
   */
  double C() const;

  /**
     @brief Return the R term (resolution term)
   */
  double R() const;

  /**
     @brief Return the exponent factor in the resolution term.
   */
  double m() const;

  /**
     @brief Return the N term (noise term)
   */
  double N() const;

  // Return the uncertainty for a momentum P.
  /**
     @brief Return the uncertainty for a variable with magnitude
     <i>p</i>.

     @param p The momentum.
   */
  double sigma (double p) const;

  // Given a value X, measured for an object with momentum P,
  // pick a new value from a Gaussian distribution
  // described by this resolution --- with mean X and width sigma(P).
  /**
     @brief Generate random value from a Gaussian distribution
     described by this resolution.  Given a value \f$x\f$, measured
     for an object with momentum \f$p\f$, pick a new value
     from a Gaussian distribution described by this resolution:
     with mean \f$x\f$ and width \f$\sigma(p)\f$.

     @param x The quantity value (distributed mean).

     @param p The momentum, for calculating the width.

     @param engine The underlying random number generator.
   */
  double pick (double x, double p, CLHEP::HepRandomEngine& engine) const;

  // Dump, for debugging.
  friend std::ostream& operator<< (std::ostream& s, const Resolution& r);


private:
  // The resolution parameters.

  /**
     The constant term.
   */
  double _constant_sigma;

  /**
     The resolution term.
   */
  double _resolution_sigma;

  /**
     The m exponential factor in the resolution term.
   */
  double _resolution_exponent;

  /**
     The noise term.
   */
  double _noise_sigma;

  /**
     The inverse flag.
   */
  bool _inverse;
};


} // namespace hitfit


#endif // not HITFIT_RESOLUTION_H
