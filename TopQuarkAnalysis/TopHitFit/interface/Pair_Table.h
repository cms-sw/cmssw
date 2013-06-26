//
// $Id: Pair_Table.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/private/Pair_Table.h
// Purpose: Helper for Fourvec_Constrainer.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// Build a lookup table to speed up constraint evaluation.
//
// We have a set of constraints, which reference labels, like
//
//    (1 2) = 80
//    (1 2) = (3 4)
//
// We also have a Fourvec_Event, which has a set of objects, each of which
// has a label.  A label may correspond to multiple objects.
//
// We'll be evaluating the mass constraints by considering each
// pair of objects o_1 o_2 and finding its contribution to each
// constraint.  (We get pairs because the constraints are quadratic
// in the objects.)
//
// We build a Pair_Table by calling the constructor, giving it the event
// and the set of constraints.  We can then get back from it a list
// of Objpair's, each representing a pair of objects that are
// used in some constraint.  The Objpair will be able to tell us
// in which constraints the pair is used (and on which side of the
// equation).
//
// CMSSW File      : interface/Pair_Table.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Pair_Table.h

    @brief A lookup table to speed up constraint evaluation.

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

#ifndef HITFIT_PAIR_TABLE_H
#define HITFIT_PAIR_TABLE_H

#include <vector>
#include <iosfwd>
#include "TopQuarkAnalysis/TopHitFit/interface/Constraint.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Objpair.h"


namespace hitfit {


class Fourvec_Event;


/**
    @brief A lookup table to speed up constraint evaluation using
    Fourvec_Constrainer.

    We have a set of constraints, which reference labels, like

    \f[
    (1~~2) = 80.4
    \f]
    \f[
    (1~~2) = (3~~4)
    \f]

    We also have a Fourvec_Event, which has a set of objects, each of
    which has a label.  A label may correspond to multiple objects.

    We'll be evaluating the mass constraints by considering each pair of
    objects \f$o_{1}\f$ and \f$o_{2}\f$ and finding its contribution to each
    constraint.  We get pairs because the constraints are quadratic in the
    objects.

    We build a Pair_Table by calling the constructor, giving it the event,
    and the set of constraints.  We can then get back from it a list
    of Objpair's each representing a pair of objects that are used
    in some constraint.  The Objpair will be able to tell us in which
    constraints the pair is used and on which side of the equation.

 */
class Pair_Table
//
// Purpose: Helper for Fourvec_Constrainer.
//
{
public:
  // Constructor.  Give it the event and the list of constraints.
  /**
     @brief Constructor, give it the event and the list of constraints.

     @param cv The list of constraints for the problem.

     @param ev The event.
   */
  Pair_Table (const std::vector<Constraint>& cv,
              const Fourvec_Event& ev);

  // The number of pairs in the table.

  /**
     @brief Return the number of pairs in the table.
   */
  int npairs () const;

  // Get one of the pairs from the table.
  /**
     @brief Get one of the pairs from the table, index starts from 0.

     @param pairno The index of the pair, index starts from 0.
   */
  const Objpair& get_pair (std::vector<Objpair>::size_type pairno) const;


private:
  //The table of pairs.
  /**
     The list of pairs.
   */
  std::vector<Objpair> _pairs;
};


// Dump out the table.
std::ostream& operator<< (std::ostream& s, const Pair_Table& p);


} // namespace hitfit


#endif // not PAIR_TABLE_H
