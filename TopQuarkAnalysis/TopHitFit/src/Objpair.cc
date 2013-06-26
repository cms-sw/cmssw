//
// $Id: Objpair.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/Objpair.cc
// Purpose: Helper class for Pair_Table.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Objpair.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Objpair.cc

    @brief Represent a pair of objects in Pair_Table.  See the documentation
    for the header file Objpair.h for details.

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


#include "TopQuarkAnalysis/TopHitFit/interface/Objpair.h"
#include <ostream>
#include <cassert>


using std::ostream;


namespace hitfit {


Objpair::Objpair (int i, int j, int nconstraints)
//
// Purpose: Constructor.
//
// Inputs:
//   i -           The first object index.
//   j -           The second object index.
//   nconstraints- The number of constraints in the problem.
//
  : _i (i),
    _j (j),
    _for_constraint (nconstraints)
{
}


void Objpair::has_constraint (std::vector<signed char>::size_type k, int val)
//
// Purpose: Set the value for constraint K (0-based) to VAL.
//
// Inputs:
//   k -           The constraint number (0-based).
//   val -         The value to set for this constraint.
//
{
  assert (k < _for_constraint.size());
  _for_constraint[k] = static_cast<signed char> (val);
}


/**
    @brief Output stream operator, print the content of this Objpair to
    an output stream.

    @param s The stream to which to write.

    @param o The instance of Objpair to be printed.
 */
std::ostream& operator<< (std::ostream& s, const Objpair& o)
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
  s << o._i << " " << o._j;
  for (unsigned k = 0; k < o._for_constraint.size(); ++k)
    s << " " << static_cast<int> (o._for_constraint[k]);
  return s;
}


} // namespace hitfit
