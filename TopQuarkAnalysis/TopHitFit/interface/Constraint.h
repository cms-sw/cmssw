//
// $Id: Constraint.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/private/Constraint.h
// Purpose: Represent a mass constraint equation.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// This class represents a mass constraint equation.
// Mass constraints come in two varieties, either saying that the sum
// of a set of labels should equal a constant:
//
//     (1 + 2) = 80
//
// or that two such sums should equal each other:
//
//     (1 + 2) = (3 + 4)
//
// We represent such a constraint equation by two Constraint_Intermed
// instances, each of which represents one side of the equation.
//
// CMSSW File      : interface/Constraint.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>


/**
    @file Constraint.h

    @brief Represent a mass constraint equation.

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


#ifndef HITFIT_CONSTRAINT_H
#define HITFIT_CONSTRAINT_H


#include <memory>
#include <string>
#include "TopQuarkAnalysis/TopHitFit/interface/Constraint_Intermed.h"


namespace hitfit {


class Fourvec_Event;


/**
    @class Constraint.

    @brief Represent a mass constraint equation.  Mass constraints
    come in two varieties, either saying that the sum of a set of labels
    should equal a constant:<br>

    \f$(1 + 2) = C\f$.<br>

    or that two such sums should be equal to each other:<br>

    \f$(1 + 2) =  (3 + 4)\f$.<br>

    We represent such a constraint equation by two Constraint_Intermed
    instances, each of which represents one side of the equation.

     */
class Constraint
//
// Purpose: Represent a mass constraint equation.
//
{
public:
  // Constructor, destructor.  S is the string to parse describing
  // the constraint.

  /**
     Constructor.

     @param s The string to parse describing the constraint.
   */
  Constraint (std::string s);

  /**
     Copy constructor.

     @param c The original object to be copied.
   */
  Constraint (const Constraint& c);


  /**
     Destructor.
   */
  ~Constraint () {}

  // Assignment.
  /**
     Assignment operator.

     @param c The original object to be copied.
   */
  Constraint& operator= (const Constraint& c);

  // See if this guy references both labels ILABEL and JLABEL
  // on a single side of the constraint equation.

  /**
     See if this guy references both labels <i>ilabel</i> and <i>jlabel</i>
     on a single single side of the constraint equation.

     @param ilabel The first label to test.
     @param jlabel The second label to test.

     @par Return:
     - +1 if the LHS references both.
     - -1 if the RHS references both.
     -  0 if neither reference both.
   */
  int has_labels (int ilabel, int jlabel) const;

  // Evaluate the mass constraint, using the data in EV.
  // Return m(lhs)^2/2 - m(rhs)^2/2.
  /**
     Evaluate the mass constraint, using the data in <i>ev</i>.

     @param ev The event for which the constraint should be evaluated.

     @par Return:
     \f$ \frac{m(\rm{lhs})^{2}}{2} - \frac{m(\rm{rhs})^{2}}{2}\f$
   */
  double sum_mass_terms (const Fourvec_Event& ev) const;

  // Print this object.
  friend std::ostream& operator<< (std::ostream& s, const Constraint& c);


private:
  // The two sides of the constraint.

  /**
     Left hand side of the constraint.
   */
  std::auto_ptr<Constraint_Intermed> _lhs;

  /**
     Right hand side of the constraint.
   */
  std::auto_ptr<Constraint_Intermed> _rhs;
};


} // namespace hitfit


#endif // not HITFIT_CONSTRAINT_H

