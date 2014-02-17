//
// $Id: Resolution.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/Resolution.cc
// Purpose: Calculate resolutions for a quantity.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Resolution.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Resolution.cc

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

#include "TopQuarkAnalysis/TopHitFit/interface/Resolution.h"
#include "CLHEP/Random/RandGauss.h"
#include <cmath>
#include <iostream>
#include <cctype>
#include <cstdlib>


using std::sqrt;
using std::ostream;
using std::string;
using std::isspace;
using std::isdigit;
#ifndef __GNUC__
using std::atof;
#endif



namespace {


bool get_field (string s, string::size_type i, double& x)
//
// Purpose: Scan string S starting at position I.
//          Find the value of the first floating-point number we
//          find there.
//
// Inputs:
//   s -           The string to scan.
//   i -           Starting character position in the string.
//
// Outputs:
//   x -           The value of the number we found.
//
// Returns:
//   True if we found something that looks like a number, false otherwise.
//
{
  string::size_type j = i;
  while (j < s.size() && s[j] != ',' && !isdigit (s[j]) && s[j] != '.')
    ++j;
  if (j < s.size() && (isdigit (s[j]) || s[j] == '.')) {
    x = atof (s.c_str() + j);
    return true;
  }
  return false;
}


} // unnamed namespace


namespace hitfit {


Resolution::Resolution (std::string s /*= ""*/)
//
// Purpose: Constructor.
//
// Inputs:
//   s -           String encoding the resolution parameters, as described
//                 in the comments in the header.
//
	:_resolution_exponent(0)
{
  _inverse = false;
  _constant_sigma = 0;
  _resolution_sigma = 0;
  _noise_sigma = 0;

  // Skip spaces.
  double x;
  string::size_type i = 0;
  while (i < s.size() && isspace (s[i]))
    ++i;

  // Check for the inverse flag.
  if (s[i] == '-') {
    _inverse = true;
    ++i;
  }
  else if (s[i] == '+') {
    ++i;
  }

  // Get the constant term.
  if (get_field (s, i, x)) _constant_sigma = x;
  i = s.find (',', i);

  // Look for a resolution term.
  if (i != string::npos) {
    ++i;
    if (get_field (s, i, x)) _resolution_sigma = x;

    // Look for a noise term.
    i = s.find (',', i);
    if (i != string::npos) {
      if (get_field (s, i+1, x)) _noise_sigma = x;
    }
  }
}


Resolution::Resolution (double C,
                        double R,
                        double m,
                        double N,
                        bool inverse /*= false*/)
  : _constant_sigma (C),
    _resolution_sigma (R),
    _resolution_exponent(m),
    _noise_sigma (N),
    _inverse (inverse)
{
}


Resolution::Resolution (double res,
                        bool inverse /*= false*/)
//
// Purpose: Constructor, to give a constant resolution.
//          I.e., sigma() will always return RES.
//
// Inputs:
//   res -         The resolution value.
//   inverse -     The inverse flag.
//
  : _constant_sigma (0),
    _resolution_sigma (0),
    _resolution_exponent(0),
    _noise_sigma (res),
    _inverse (inverse)
{
}


bool Resolution::inverse () const
//
// Purpose: Return the inverse flag.
//
// Returns:
//   The inverse flag.
//
{
  return _inverse;
}


double Resolution::C() const
{
    return _constant_sigma;
}


double Resolution::R() const
{
    return _resolution_sigma;
}


double Resolution::m() const
{
    return _resolution_exponent;
}


double Resolution::N() const
{
    return _noise_sigma;
}


double Resolution::sigma (double p) const
//
// Purpose: Return the uncertainty for a momentum P.
//
// Inputs:
//    p -          The momentum
//
// Returns:
//   The uncertainty for a momentum P.
//
{
  if (_inverse)
    p = 1 / p;

  return sqrt ((_constant_sigma*_constant_sigma*p +
        _resolution_sigma*_resolution_sigma)*p +
           _noise_sigma*_noise_sigma);
}


double Resolution::pick (double x, double p, CLHEP::HepRandomEngine& engine) const
//
// Purpose: Given a value X, measured for an object with momentum P, 
//          pick a new value from a Gaussian distribution
//          described by this resolution --- with mean X and width sigma(P).
//
// Inputs:
//    x -          The quantity value (distribution mean).
//    p -          The momentum, for calculating the width.
//    engine -     The underlying RNG.
//
// Returns:
//   A smeared value of X.
//
{
  CLHEP::RandGauss gen (engine);
  if (_inverse)
    return 1 / gen.fire (1 / x, sigma (p));
  else
    return gen.fire (x, sigma (p));
}


/**
    @brief Output stream operator, print the content of this Resolution
    to an output stream.

    @param s The stream to which to write.

    @param r The instance of Resolution to be printed.
 */
std::ostream& operator<< (std::ostream& s, const Resolution& r)
//
// Purpose: Dump this object to S.
//
// Inputs:
//    s -          The stream to which to write.
//    r -          The object to dump.
//
// Returns:
//   The stream S.
//   
{
  if (r._inverse) s << "-";
  s << r._constant_sigma << ","
    << r._resolution_sigma << ","
    << r._noise_sigma;
  return s;
}


} // namespace hitfit
