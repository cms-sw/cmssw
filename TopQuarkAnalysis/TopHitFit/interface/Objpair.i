//
// $Id: Objpair.i,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/private/Objpair.i
// Purpose: Helper class for Pair_Table.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : interface/Objpair.i
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//



/**
    @file Objpair.i

    @brief Inline source file for Objpair class.

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


#include <cassert>


namespace hitfit {


inline
int Objpair::i () const
//
// Purpose: Return the first object index for this pair.
//
// Returns:
//   The first object index for this pair.
//
{
  return _i;
}


inline
int Objpair::j () const
//
// Purpose: Return the second object index for this pair.
//
// Returns:
//   The second object index for this pair.
//
{
  return _j;
}


inline
int Objpair::for_constraint (std::vector<signed char>::size_type k) const
//
// Purpose: Retrieve the value set for constraint K.
//
// Inputs:
//   k -           The constraint number (0-based).
//
// Returns:
//   The value for constraint K.
//
{
  assert (k < _for_constraint.size());
  return _for_constraint[k];
}


} // namespace hitfit
