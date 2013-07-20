//
// $Id: Constraint.cc,v 1.1 2011/05/26 09:46:59 mseidel Exp $
//
// File: src/Constraint.cc
// Purpose: Represent a mass constraint equation.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Constraint.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Constraint.cc

    @brief Represent a mass constraint equation.
    See the documentation for the header file Constraint.h for details.

    @par Creation date:
    July 2000.

    @author
    Scott Stuart Snyder <snyder@bnl.gov>.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Oct 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added Doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent from the original author (Scott Snyder).
 */

#include "TopQuarkAnalysis/TopHitFit/interface/Constraint.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Constraint_Intermed.h"
#include <iostream>
#include <cassert>


using std::auto_ptr;
using std::ostream;
using std::string;

namespace hitfit {


Constraint::Constraint (const Constraint& c)
//
// Purpose: Copy constructor.
//
// Inputs:
//   c -           The instance to copy.
//
  : _lhs (c._lhs->clone ()),
    _rhs (c._rhs->clone ())
{
}


Constraint& Constraint::operator= (const Constraint& c)
//
// Purpose: Assignment.
//
// Inputs:
//   c -           The instance to copy.
//
// Returns:
//   This instance.
//
{
  {
    auto_ptr<Constraint_Intermed> ci = c._lhs->clone ();
    _lhs = ci;
  }
  {
    auto_ptr<Constraint_Intermed> ci = c._rhs->clone ();
    _rhs = ci;
  }

  return *this;
}


Constraint::Constraint (std::string s)
//
// Purpose: Constructor.
//          Build a constraint from the string describing it.
//
// Inputs:
//   s -           The string describing the constraint.
//
{
  // Split it at the equals sign.
  string::size_type i = s.find ('=');
  assert (i != string::npos);

  // And then build the two halves.
  {
    auto_ptr<Constraint_Intermed> ci =
      make_constraint_intermed (s.substr (0, i));
    _lhs = ci;
  }
  {
    auto_ptr<Constraint_Intermed> ci =
      make_constraint_intermed (s.substr (i+1));
    _rhs = ci;
  }
}


int Constraint::has_labels (int ilabel, int jlabel) const
//
// Purpose: See if this guy references both labels ILABEL and JLABEL
//          on a single side of the constraint equation.
//
// Inputs:
//   ilabel -      The first label to test.
//   jlabel -      The second label to test.
//
// Returns:
//   +1 if the LHS references both.
//   -1 if the RHS references both.
//    0 if neither reference both.
//
{
  if (_lhs->has_labels (ilabel, jlabel))
    return 1;
  else if (_rhs->has_labels (ilabel, jlabel))
    return -1;
  else
    return 0;
}


double Constraint::sum_mass_terms (const Fourvec_Event& ev) const
//
// Purpose: Evaluate the mass constraint, using the data in EV.
//          Return m(lhs)^2/2 - m(rhs)^2/2.
//
// Inputs:
//   ev -          The event for which the constraint should be evaluated.
//
// Returns:
//   m(lhs)^2/2 - m(rhs)^2/2.
//
{
  return _lhs->sum_mass_terms (ev) - _rhs->sum_mass_terms (ev);
}


/**

    @brief Output stream operator, print the content of this Constraint to
    an output stream.

    @param s The stream to which to write.

    @param c The instance of Constraint to be printed.

*/
std::ostream& operator<< (std::ostream& s, const Constraint& c)
//
// Purpose: Print the object to S.
//
// Inputs:
//   s -           The stream to which to write.
//   c -           The object to write.
//
// Returns:
//   The stream S.
//
{
  s << *c._lhs.get() << " = " << *c._rhs.get();
  return s;
}


} // namespace hitfit
