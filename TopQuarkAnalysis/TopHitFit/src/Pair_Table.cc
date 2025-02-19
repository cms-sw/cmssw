//
// $Id: Pair_Table.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/Pair_Table.cc
// Purpose: Helper for Fourvec_Constrainer.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Pair_Table.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Pair_Table.cc

    @brief A lookup table to speed up constraint evaluation.
    See the documentation for the header file Pair_Table.h for details.

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


#include "TopQuarkAnalysis/TopHitFit/interface/Pair_Table.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Fourvec_Event.h"
#include <ostream>


using std::vector;
using std::ostream;


namespace hitfit {


Pair_Table::Pair_Table (const std::vector<Constraint>& cv,
                        const Fourvec_Event& ev)
//
// Purpose: Constructor.
//
// Inputs:
//   cv -          The list of constraints for the problem.
//   ev -          The event.
//
{
  // The number of objects in the event, including any neutrino.
  int nobjs = ev.nobjs_all();

  // Number of constraints.
  int nc = cv.size();

  // Loop over pairs of objects.
  for (int i=0; i < nobjs-1; i++)
    for (int j=i+1; j < nobjs; j++) {
      // Make an Objpair instance out of it.
      Objpair p (i, j, nc);

      // Loop over constraints.
      bool wanted = false;
      for (int k=0; k < nc; k++) {
        int val = cv[k].has_labels (ev.obj (i).label, ev.obj (j).label);
        if (val) {
          // This pair is used by this constraint.  Record it.
          p.has_constraint (k, val);
          wanted = true;
        }
      }

      // Was this pair used by any constraint?
      if (wanted)
        _pairs.push_back (p);
    }
}


int Pair_Table::npairs () const
//
// Purpose: Return the number of pairs in the table.
//
// Returns:
//   The number of pairs in the table.
//
{
  return _pairs.size();
}


const Objpair& Pair_Table::get_pair (std::vector<Objpair>::size_type pairno) const
//
// Purpose: Return one pair from the table.
//
// Inputs:
//   pairno -      The number of the pair (0-based).
//
// Returns:
//   Pair PAIRNO.
//
{
  assert (pairno < _pairs.size());
  return _pairs[pairno];
}

/**
    @brief Output stream operator, print the content of this Pair_Table
    to an output stream.

    @param s The stream to which to write.

    @param p The instance of Pair_Table to be printed.
 */
std::ostream& operator<< (std::ostream& s, const Pair_Table& p)
//
// Purpose: Print the object to S.
//
// Inputs:
//   s -           The stream to which to write.
//   p -           The object to write.
//
// Returns:
//   The stream S.
//
{
  for (int i=0; i < p.npairs(); i++)
    s << " " << p.get_pair (i) << "\n";
  return s;
}


} // namespace hitfit
