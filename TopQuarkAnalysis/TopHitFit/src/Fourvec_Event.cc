//
// $Id: Fourvec_Event.cc,v 1.2 2013/05/28 17:55:59 gartung Exp $
//
// File: src/Fourvec_Event.cc
// Purpose: Represent an event for kinematic fitting as a collection
//          of 4-vectors.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Fourvec_Event.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


#include "TopQuarkAnalysis/TopHitFit/interface/Fourvec_Event.h"
#include <cassert>
#include <ostream>


/**
    @file Fourvec_Event.cc

    @brief Represent an event for kinematic fitting as a collection of
    four-momenta.  See the documentation the for header file
    Fourvec_Event.h for details.

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

using std::ostream;


namespace hitfit {


FE_Obj::FE_Obj (const Fourvec& the_p,
                double the_mass,
                int the_label,
                double the_p_error,
                double the_phi_error,
                double the_eta_error,
                bool the_muon_p)
//
// Purpose: Contructor.
//
// Inputs:
//   the_p -       4-momentum.
//   the_mass -    The mass of the object.
//                 The constrained fit will fix the mass to this value.
//   the_p_error - Uncertainty in p (or, if the_muon_p is set, in 1/p).
//   the_phi_error-Uncertainty in phi.
//   the_eta_error-Uncertainty in eta.
//   the_muon_p -  If true, the `p' uncertainty is in 1/p, and 1/p
//                 should be used as the fit variable instead of p.
//
  : p (the_p),
    mass (the_mass),
    label (the_label),
    p_error (the_p_error),
    phi_error (the_phi_error),
    eta_error (the_eta_error),
    muon_p (the_muon_p)
{
}


/**
    @brief Output stream operator, print the content of this FE_Obj object
    to an output stream.

    @param s The output stream to which to write.

    @param o The instance of FE_Obj to be printed.
 */
std::ostream& operator<< (std::ostream& s, const FE_Obj& o)
//
// Purpose: Print the object to S.
//
// Inputs:
//   s -           The stream to which to write.
//   o -           The object to write.
//
// Returns:
//   The stream S.
//
{
  s << o.p << " - " << o.mass << " - " << o.label << "\n";
  s << "    errors: " << o.p_error << " " << o.phi_error << " " << o.eta_error;
  if (o.muon_p)
    s << " (mu)";
  s << "\n";
  return s;
}


//************************************************************************


Fourvec_Event::Fourvec_Event ()
//
// Purpose: Constructor.
//
  : _kt_x_error (0),
    _kt_y_error (0),
    _kt_xy_covar (0),
    _has_neutrino (false)
{
}


bool Fourvec_Event::has_neutrino () const
//
// Purpose: Return true if this event has a neutrino.
//
// Returns:
//   True if this event has a neutrino.
//
{
  return _has_neutrino;
}


int Fourvec_Event::nobjs () const
//
// Purpose: Return the number of objects in the event, not including
//          any neutrino.
//
// Returns:
//   The number of objects in the event, not including any neutrino.
//
{
  return _objs.size() - (_has_neutrino ? 1 : 0);
}


int Fourvec_Event::nobjs_all () const
//
// Purpose: Return the number of objects in the event, including any neutrino.
//
// Returns:
//   The number of objects in the event, including any neutrino.
//
{
  return _objs.size();
}


const FE_Obj& Fourvec_Event::obj (std::vector<FE_Obj>::size_type i) const
//
// Purpose: Access object I.
//
// Inputs:
//   i -           The index of the desired object (0-based indexing).
//
{
  assert (i < _objs.size ());
  return _objs[i];
}


const Fourvec& Fourvec_Event::nu () const
//
// Purpose: Access the neutrino 4-vector.
//
{
  assert (_has_neutrino);
  return _objs.back().p;
}


const Fourvec& Fourvec_Event::kt () const
//
// Purpose: Access the kt 4-vector.
//
{
  return _kt;
}


const Fourvec& Fourvec_Event::x () const
//
// Purpose: Access the X 4-vector.
//
{
  return _x;
}


double Fourvec_Event::kt_x_error () const
//
// Purpose: Return the X uncertainty in kt.
//
// Returns:
//   The X uncertainty in kt.
//
{
  return _kt_x_error;
}


double Fourvec_Event::kt_y_error () const
//
// Purpose: Return the Y uncertainty in kt.
//
// Returns:
//   The Y uncertainty in kt.
//
{
  return _kt_y_error;
}


double Fourvec_Event::kt_xy_covar () const
//
// Purpose: Return the kt XY covariance.
//
// Returns:
//   The kt XY covariance.
//
{
  return _kt_xy_covar;
}


/**
    @brief Output stream operator, print the content of this Fourvec_Event
    object to an output stream.

    @param s The stream to which to write.

    @param fe The instance of Fourvec_Event to be printed.
*/
std::ostream& operator<< (std::ostream& s, const Fourvec_Event& fe)
//
// Purpose: Print out the contents of the class.
//
// Inputs:
//   s -           The stream to which to write.
//   fe -          The object to write.
//
// Returns:
//   The stream S.
//
{
  s << "kt: (" << fe._kt.x() << ", " << fe._kt.y() << "); "
    << " error: " << fe._kt_x_error << " " << fe._kt_y_error << " "
    << fe._kt_xy_covar << "\n";
  s << "x: " << fe._x << "\n";
  for (unsigned i = 0; i < fe._objs.size(); i++)
    s << i+1 << ": " << fe._objs[i];
  return s;
}


void Fourvec_Event::add (const FE_Obj& obj)
//
// Purpose: Add an object to the event.
//
// Inputs:
//   obj -         The object to add.
//                 It should not be a neutrino.
//
{
  assert (obj.label != nu_label);

  // Add to the end of the list, but before any neutrino.
  if (_has_neutrino) {
    assert (_objs.size() > 0);
    _objs.insert (_objs.begin() + _objs.size() - 1, obj);
  }
  else
    _objs.push_back (obj);

  // Maintain kt.
  _kt += obj.p;
}


void Fourvec_Event::set_nu_p (const Fourvec& p)
//
// Purpose: Set the neutrino 4-momentum to P.
//          This adds a neutrino if there wasn't one there already.
//
// Inputs:
//   p -           The new 4-momentum of the neutrino.
//
{
  if (_has_neutrino) {
    _kt -= _objs.back().p;
    _objs.back().p = p;
  }
  else {
    _has_neutrino = true;
    _objs.push_back (FE_Obj (p, 0, nu_label, 0, 0, 0, false));
  }

  _kt += p;
}


void Fourvec_Event::set_obj_p (std::vector<FE_Obj>::size_type i, const Fourvec& p)
//
// Purpose: Set object I's 4-momentum to P.
//
// Inputs:
//   i -           The index of the object to change (0-based indexing).
//   p -           The new 4-momentum.
//
{
  assert (i < _objs.size ());
  _kt -= _objs[i].p;
  _objs[i].p = p;
  _kt += p;
}


void Fourvec_Event::set_x_p (const Fourvec& p)
//
// Purpose: Set the momentum of the X object to P.
//
// Inputs:
//   p -           The new 4-momentum.
//
{
  _kt -= _x;
  _x = p;
  _kt += p;
}


void Fourvec_Event::set_kt_error (double kt_x_error,
                                  double kt_y_error,
                                  double kt_xy_covar)
//
// Purpose: Set the kt uncertainties.
//
// Inputs:
//   kt_x_error -  The uncertainty in the X component of kt.
//   kt_y_error -  The uncertainty in the Y component of kt.
//   kt_xy_covar - The covariance between the X and Y components.
//
{
  _kt_x_error = kt_x_error;
  _kt_y_error = kt_y_error;
  _kt_xy_covar = kt_xy_covar;
}


} // namespace hitfit

