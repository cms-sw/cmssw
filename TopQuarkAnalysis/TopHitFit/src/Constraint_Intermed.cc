//
// $Id: Constraint_Intermed.cc,v 1.1 2011/05/26 09:46:59 mseidel Exp $
//
// File: src/Constraint_Intermed.cc
// Purpose: Represent one side of a mass constraint equation.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Constraint_Intermed.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Constraint_Intermed.cc

    @brief Represent one side of a mass constraint
    equation.  See the documentation for the header file
    Constraint_Intermed.h for details.

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


#include "TopQuarkAnalysis/TopHitFit/interface/Constraint_Intermed.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Fourvec_Event.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdlib>


using std::auto_ptr;
using std::ostream;
using std::sqrt;
using std::stable_sort;
using std::find;
using std::swap;
using std::string;
using std::vector;
#ifndef __GNUC__
using std::atoi;
using std::atof;
#endif


namespace hitfit {


//************************************************************************


Constraint_Intermed_Constant::Constraint_Intermed_Constant (double constant)
//
// Purpose: Constructor.
//
// Inputs:
//   constant -    The constant in the mass constraint half.
//
  : _c2 (constant * constant / 2)
{
}


Constraint_Intermed_Constant::Constraint_Intermed_Constant
 (const Constraint_Intermed_Constant& c)
//
// Purpose: Copy constructor.
//
// Inputs:
//   c -           The instance to copy.
//
   : _c2 (c._c2)
{
}


bool Constraint_Intermed_Constant::has_labels (int ilabel, int jlabel) const
//
// Purpose: Return true if this guy references both labels ILABEL and JLABEL.
//
//          This version always returns false.
//
// Inputs:
//   ilabel -      The first label to test.
//   jlabel -      The second label to test.
//
// Returns:
//   True if this guy references both labels ILABEL and JLABEL.
//
{
  return false;
}


double
Constraint_Intermed_Constant::sum_mass_terms (const Fourvec_Event&/*ev*/) const
//
// Purpose: Evaluate this half of the mass constraint, using the data in EV.
//          Return m^2/2.
//
// Inputs:
//   ev -          The event for which the constraint should be evaluated.
//
// Returns:
//   m^2/2.
//
{
  return _c2;
}


void Constraint_Intermed_Constant::print (std::ostream& s) const
//
// Purpose: Print out this object.
//
// Inputs:
//   s -           The stream to which we should write.
//
{
  s << sqrt (2 * _c2);
}


auto_ptr<Constraint_Intermed> Constraint_Intermed_Constant::clone () const
//
// Purpose: Copy this object.
//
// Returns:
//   A new copy of this object.
//
{
  return auto_ptr<Constraint_Intermed>
    (new Constraint_Intermed_Constant (*this));
}


//************************************************************************


Constraint_Intermed_Labels::Constraint_Intermed_Labels
 (const std::vector<int>& labels)
//
// Purpose: Constructor.
//
// Inputs:
//   labels -      The labels used by this half-constraint.
//
  : _labels (labels)
{
  // Sort them.
  stable_sort (_labels.begin(), _labels.end());
}


Constraint_Intermed_Labels::Constraint_Intermed_Labels
 (const Constraint_Intermed_Labels& c)
//
// Purpose: Copy constructor.
//
// Inputs:
//   c -           The instance to copy.
//
   : _labels (c._labels)
{
}


bool Constraint_Intermed_Labels::has_labels (int ilabel, int jlabel) const
//
// Purpose: Return true if this guy references both labels ILABEL and JLABEL.
//
// Inputs:
//   ilabel -      The first label to test.
//   jlabel -      The second label to test.
//
// Returns:
//   True if this guy references both labels ILABEL and JLABEL.
//
{
  if (ilabel > jlabel)
    swap (ilabel, jlabel);

  unsigned sz = _labels.size();
  unsigned i;
  for (i=0; i < sz; i++) {
    if (_labels[i] == ilabel)
      break;
  }

  if (i == sz)
    return false;

  for (; i < sz; i++) {
    if (_labels[i] == jlabel)
      break;
  }

  if (i == sz)
    return false;

  return true;
}


double
Constraint_Intermed_Labels::sum_mass_terms (const Fourvec_Event& ev) const
//
// Purpose: Evaluate this half of the mass constraint, using the data in EV.
//          Return m^2/2.
//
// Inputs:
//   ev -          The event for which the constraint should be evaluated.
//
// Returns:
//   m^2/2.
//
{
  int nobjs = ev.nobjs();
  double sum = 0;
  for (int i = 0; i < nobjs; i++) {
    const FE_Obj& o = ev.obj (i);
    if (has_label (o.label))
      sum += o.mass * o.mass / 2;
  }

  return sum;
}


void Constraint_Intermed_Labels::print (std::ostream& s) const
//
// Purpose: Print out this object.
//
// Inputs:
//   s -           The stream to which we should write.
//
{
  s << "(";
  for (unsigned i = 0; i < _labels.size(); i++) {
    if (i > 0)
      s << "+";
    s << _labels[i];
  }
  s << ")";
}


bool Constraint_Intermed_Labels::has_label (int label) const
//
// Purpose: Helper function: Test to see if we use label LABEL.
//
// Inputs:
//   label -       THe label for which to search.
//
// Returns:
//   True if we use label LABEL.
//
{
  return find (_labels.begin(), _labels.end(), label) != _labels.end();
}


auto_ptr<Constraint_Intermed> Constraint_Intermed_Labels::clone () const
//
// Purpose: Copy this object.
//
// Returns:
//   A new copy of this object.
//
{
  return auto_ptr<Constraint_Intermed> 
    (new Constraint_Intermed_Labels (*this));
}


//************************************************************************


/**
   @brief Output stream operator, print the content of this Constraint_Intermed
   to an output stream.

   @param s The output stream to write.

   @param ci The instance of <i>Constraint_Intermed</i> to be printed.

   @par Return:
   The stream <i>s</i>.
 */
std::ostream& operator<< (std::ostream& s, const hitfit::Constraint_Intermed& ci)
//
// Purpose: Print the object to S.
//
// Inputs:
//   s -           The stream to which to write.
//   ci -          The object to write.
//
// Returns:
//   The stream S.
//
{
  ci.print (s);
  return s;
}


auto_ptr<Constraint_Intermed> make_constraint_intermed (string s)
//
// Purpose: Parse the string S and return an appropriate
//          Constraint_Intermed instance.
//          Returns null if we can't interpret the string.
//
//          The string should either be a numeric constant like
//
//            80.2
//
//          or a list of integers in parens, like
//
//            (1 4 2)
//
//          Leading spaces are ignored, as is text in a leading < >
//          construction.
//
// Inputs:
//   s -           The string to parse.
//
// Returns:
//   A new Constraint_Intermed instance, or null if we couldn't
//   interpret the string.
//
{
  // Skip leading spaces, `=', '< ... >'.
  string::size_type i = 0;
  while (i < s.size() && s[i] == ' ')
    ++i;
  if (s[i] == '=')
    ++i;
  while (i < s.size() && s[i] == ' ')
    ++i;
  if (i < s.size() && s[i] == '<') {
    i = s.find ('>', i);
    if (i == string::npos)
      return auto_ptr<Constraint_Intermed> ();
    ++i;
  }
  while (i < s.size() && s[i] == ' ')
    ++i;

  // Fail if there's nothing left.
  if (i == s.size())
    return auto_ptr<Constraint_Intermed> ();

  if (s[i] == '(') {
    // List of labels.
    // Make a Constraint_Intermed_Labels instance.
    vector<int> labels;
    ++i;
    while (i < s.size()) {
      while (i < s.size() && s[i] == ' ')
        ++i;
      if (i < s.size() && s[i] == ')')
        break;
      if (i < s.size())
        labels.push_back (atoi (s.c_str() + i));
      while (i < s.size() && s[i] != ' ' && s[i] != ')')
        ++i;
    }
    return auto_ptr<Constraint_Intermed>
      (new Constraint_Intermed_Labels (labels));
  }
  else {
    // Make a Constraint_Intermed_Constant instance.
    return auto_ptr<Constraint_Intermed>
      (new Constraint_Intermed_Constant (atof (s.c_str() + i)));
  }
}


} // namespace hitfit

