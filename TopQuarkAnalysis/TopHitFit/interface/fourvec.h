//
// $Id: fourvec.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/fourvec.h
// Purpose: Define 3- and 4-vector types for the hitfit package, and
//          supply a few additional operations.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// This file defines the types `Threevec' and `Fourvec' to be used
// in hitfit code.  These are based on the corresponding CLHEP classes.
// We also provide a handful of operations in addition to those that
// CLHEP has.
//
// CMSSW File      : interface/fourvec.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**
    @file fourvec.h

    @brief Define three-vector and four-vector classes for the HitFit
    package, and supply a few additional operations.

    This file defines the type Threevec and Fourvec to be used in HitFit.
    These classes are based on the corresponding CLHEP classes.
    This file also provides some other operations in addition to those
    that CLHEP has.

    @author Scott Stuart Snyder <snyder@bnl.gov>

    @par Creation date:
    Jul 2000.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Nov 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added Doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent from the original author (Scott Snyder).
 */

#ifndef HITFIT_FOURVEC_H
#define HITFIT_FOURVEC_H

#include "CLHEP/Vector/LorentzVector.h"


namespace hitfit {


// Define the types that we want to use.
/**
    @brief Typedef for a HepLorentzVector.
 */
typedef CLHEP::HepLorentzVector Fourvec;

/**
    @brief Typedef for a Hep3Vector.
 */
typedef CLHEP::Hep3Vector Threevec;

// Adjust the 3-vector part of V (leaving the energy unchanged) so that
// it has mass MASS.
/**
    @brief Adjust the three-vector part of v, leaving the energy unchanged,

    so that the four-vector has mass as specified in the argument.

    @param v The four-vector to scale.

    @param mass The desired mass of the four-vector.

    @par Output:
    <i>v</i> The scaled four-vector.
 */
void adjust_p_for_mass (Fourvec& v, double mass);

// Adjust the energy component of V (leaving the 3-vector part unchanged)
// so that it has mass MASS.
/**
    @brief Adjust the energy component of four-vector v (leaving the three-vector
    part unchanged) so that the four-vector has mass as specified in the
    argument.

    @param v The four-vector to scale.

    @param mass The desired-mass of the four-vector.

    @par Output:
    <i>v</i> The scaled four-vector.
 */
void adjust_e_for_mass (Fourvec& v, double mass);

// Rotate V through polar angle THETA.
/**
    @brief Rotate four-vector v through a polar angle.

    @param v The four-vector to rotate.

    @param theta The rotation angle.

    @par Output:
    <i>v</i> The rotated vector.
 */
void rottheta (Fourvec& v, double theta);

// Rotate V through a polar angle such that its pseudorapidity changes by ETA.
/**
    @brief Rotate four-vector v through a polar angle such that the four-vector
    pseudorapidity changes by a desired value.

    @param v The four-vector to rotate.

    @param eta The desired change in the pseudorapidity.

    @par Output:
    <i>v</i> The rotated four-vector.
 */
void roteta (Fourvec& v,   double eta);

// Conversions between pseudorapidity and polar angle.
/**
    @brief Convert pseudorapidity to polar angle.

    @param eta The value of pseudorapidity to convert.
 */
double eta_to_theta (double eta);
/**
    @brief Convert polar angle to pseudorapidity.

    @param theta The polar angle to convert.
 */
double theta_to_eta (double theta);

// Get the detector eta (D0-specific).  Needs a Z-vertex.
/**
    @brief NOT USED ANYMORE: Get the detector \f$\eta\f$ (D0-specific),
	requires z-vertex.

    @param v The four-vector on which to operate.

    @param zvert z-vertex of the event.
*/
double deteta (const Fourvec& v, double zvert);  // XXX

//  Handle wraparound for a difference in azimuthal angles.
/**
    @brief Normalized difference in azimuthal angles to a range
    between \f$[-\pi \dot \pi]\f$.

    @param phi The azimuthal to be normalized.
 */
double phidiff (double phi);

// Find the distance in R between two four-vectors.
/**
    @brief Find the distance between two four-vectors in the two-dimensional
    space \f$\eta-\phi\f$.
 */
double delta_r (const Fourvec& a, const Fourvec& b);


} // namespace hitfit


#endif // not HITFIT_FOURVEC_H

