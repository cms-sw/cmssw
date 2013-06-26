//
// $Id: Vector_Resolution.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Vector_Resolution.h
// Purpose: Calculate resolutions in p, phi, eta.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// These objects hold three Resolution objects, one each for p, phi, eta.
// In addition, we have a use_et flag; if set, then the p resolution
// is really in pt.
//
// We can initialize these objects from a string; the format is
//
//   <p-res>/<eta-res>/<phi-res>[/et]
//
// where the resolution formats are as given in Resolution.h.
//
// CMSSW File      : interface/Vector_Resolution.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Vector_Resolution.h

    @brief Calculate and represent resolution for a vector
    of momentum \f$p\f$, pseudorapidity \f$\eta\f$, and azimuthal
    angle \f$\phi\f$.

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

#ifndef HITFIT_VECTOR_RESOLUTION_H
#define HITFIT_VECTOR_RESOLUTION_H


#include <string>
#include <iosfwd>
#include "TopQuarkAnalysis/TopHitFit/interface/Resolution.h"
#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"


namespace hitfit {


/**
    @class Vector_Resolution

    @brief Calculate and represent resolution for a vector of
    \f$p\f$, pseudorapidity \f$\eta\f$, and azimuthal angle \f$\phi\f$.
    An instance of this class holds three Resolution objects, one each for
    \f$p\f$, \f$\eta\f$, and \f$\phi\f$.  In addition, we have a flag
    to indicate if the momentum resolution is in \f$p_{T}\f$ or \f$p\f$.
    This flag is set by appending [/et] at the end of the string.

    This class is initialized from a string with format

    \verbatim
<p-resolution>/<eta-resolution>/<phi-resolution>[/et]
    \endverbatim

    where the resolution formats are given in the documentation for
    the Resolution class.

    Addendum by Haryo Sumowidagdo:
    When using the inverse flag, only the inverse flag in p-resolution
    will be used by the fitter to set that flag which says the error in 'p'
    is really the error on '1/p'.
    The inverse flag for \f$\eta\f$ in eta-resolution and \f$\phi\f$
    in phi-resolution are not used by the fitter.  As usually the
    angular resolution is of the form

    \f$ \sigma_{\eta,\phi} = \sqrt{\frac{C^{2}}{p^{2}} + \frac{R^{2}}{p} + N^2}\f$

    where \f$ p \f$ can be the magnitude of the momentum (\f$ p \f$) or
    or transverse momentum (\f$ p_{T} \f$), one then can (and should!)
    use the inverse flag for \f$ \eta \f$ and  \f$ \phi \f$ resolution.

 */
class Vector_Resolution
//
// Purpose: Calculate resolutions in p, phi, eta.
//
{
public:
  // Constructor.  Create a vector resolution object with infinite precision
  /**
     @brief Constructor, instantiate an instance of Vector_Resolution with
     infinite precision.
   */
  Vector_Resolution ();

  // Constructor.  Parse a string as described above.
  /**
     @brief Constructor, instantiate an instance of Vector_Resolution from
     a string using format as described in the class description.

     @param s String enconding the resolution parameters.
   */
  Vector_Resolution (std::string s);

  // Constructor from individual resolution objects.
  /**
     @brief Constructor, instantiate an instance of Vector_Resolution from
     three instances of Resolution objects.

     @param p_res The momentum resolution.

     @param eta_res The pseudorapidity resolution.

     @param phi_res The azimuthal angle resolution.

     @param use_et If <b>TRUE</b> then use \f$p_{T}\f$ instead of \f$p\f$
     for momentum resolution.
   */
  Vector_Resolution (const Resolution& p_res,
                     const Resolution& eta_res,
                     const Resolution& phi_res,
                     bool use_et = false);

  // Get back the individual resolution objects.
  /**
     @brief Return a constant reference to the momentum resolution.
   */
  const Resolution& p_res () const;

  /**
     @brief Return a constant reference to the pseudorapidity resolution.
   */
  const Resolution& eta_res () const;

  /**
     @brief Return a constant reference to the azimuthal angle resolution.
   */
  const Resolution& phi_res () const;

  // Return the use_et flag.
  /**
     @brief Return the <i>use_et</i> flag.
   */
  bool use_et () const;

  // Calculate resolutions from a 4-momentum.
  /**
     @brief Calculate the momentum resolution of a four-momentum.

     @param v The four-momentum.
   */
  double p_sigma   (const Fourvec& v) const;

  /**
     @brief Calculate the pseudorapidity resolution of a four-momentum.

     @param v The four-momentum.
   */
  double eta_sigma (const Fourvec& v) const;

  /**
     @brief Calculate the azimuthal angle resolution of a four-momentum.

     @param v The four-momentum.
   */
  double phi_sigma (const Fourvec& v) const;

  // Smear a 4-vector V according to the resolutions.
  // If DO_SMEAR_DIR is false, only smear the total energy.
  /**
     @brief Smear a four-momentum according to the resolutions.

     @param v The four-momentum to smear.

     @param engine The underlying random number generator.

     @param do_smear_dir If <b>FALSE</b>, only smear the energy.
     If <b>TRUE</b>, also smear the direction.
   */
  void smear (Fourvec& v,
              CLHEP::HepRandomEngine& engine,
              bool do_smear_dir = false) const;

  // Dump this object, for debugging.
  friend std::ostream& operator<< (std::ostream& s,
                                   const Vector_Resolution& r);

private:
  // State for this object.
  /**
     The momentum resolution.
   */
  Resolution _p_res;

  /**
     The pseudorapidity resolution.
   */
  Resolution _eta_res;

  /**
     The phi resolution.
   */
  Resolution _phi_res;

  /**
     The momentum resolution.
   */
  bool _use_et;

  // Helper.
  /**
     @brief Helper function to smear direction.

     @param v The four-momentum to smear.

     @param engine The underlying random number generator.
   */
  void smear_dir (Fourvec& v, CLHEP::HepRandomEngine& engine) const;
};


} // namespace hitfit


#endif // not HITFIT_VECTOR_RESOLUTION_H
