//
// $Id: Objpair.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/private/Objpair.h
// Purpose: Helper class for Pair_Table.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// An `Objpair' consists of two object indices in a Fourvec_Event,
// plus a value for each constraint.  This value is +1 if these two objects
// are used on the lhs of that constraint, -1 if they are used on the rhs
// of that constraint, and 0 if they are not used by that constraint.
//
// CMSSW File      : interface/Objpair.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Objpair.h

    @brief Represent a pair of objects in Pair_Table.

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


#ifndef HITFIT_OBJPAIR_H
#define HITFIT_OBJPAIR_H


#include <vector>
#include <iosfwd>


namespace hitfit {

/**
    @brief Represent a pair of objects in Pair_Table.

    An Objpair consists of two object indices in a Fourvec_Event,
    plua a value for each constraint.  This value is defined as:
    - +1 if these two objects are used on the left-hand side of the
    constraint.
    - -1 if these two objects are used on the right-hand side of the
    constraint.
    0 if these two objects are not used by that constraint.

 */
class Objpair
//
// Purpose: Helper class for Pair_Table.
//
{
public:
  // Constructor.  I and J are the two object indices, and NCONSTRAINTS
  // is the number of constraints in the problem.
  /**
     @brief Constructor

     @param i Index of the first object.

     @param j Index of the second object.

     @param nconstraints The number of constraints in the problem.
   */
  Objpair (int i, int j, int nconstraints);

  // Set the value for constraint K (0-based) to VAL.
  /**
     @brief Set the value for a constraint to a value.

     @param k The index of the constraint, index starts from 0.

     @param val The value to set for this constraint.
   */
  void has_constraint (std::vector<signed char>::size_type k, int val);

  // Get back the indices in this pair.
  /**
     @brief Return the index of the first object.
   */
  int i () const;

  /**
     @brief Return the index of the second object.
   */
  int j () const;

  // Retrieve the value set for constraint K.
  /**
     @brief Retrieve the value set for a constraint.

     @param k The index of the constraint, index starts from 0.
   */
  int for_constraint (std::vector<signed char>::size_type k) const;

  // Print this object.

  friend std::ostream& operator<< (std::ostream& s, const Objpair& o);


private:
  // The object indices for this pair.

  /**
     Index of the first object.
   */
  int _i;

  /**
     Index of the second object.
   */
  int _j;

  // The list of values for each constraint.
  /**
     The list of values for each constraint.
   */
  std::vector<signed char> _for_constraint;
};


} // namespace hitfit


#include "TopQuarkAnalysis/TopHitFit/interface/Objpair.i"


#endif // not HITFIT_OBJPAIR_H
