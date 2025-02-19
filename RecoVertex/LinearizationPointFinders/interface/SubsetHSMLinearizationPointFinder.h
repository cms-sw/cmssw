#ifndef SubsetHSMLinearizationPointFinder_H
#define SubsetHSMLinearizationPointFinder_H

#include "RecoVertex/LinearizationPointFinders/interface/CrossingPtBasedLinearizationPointFinder.h"

  /** A linearization point finder. It works the following way:
   *  1. Calculate in an optimal way 'n_pairs' different crossing points.
   *     Optimal in this context means the following:
   *     a. Try to use as many different tracks as possible;
   *        avoid using the same track all the time.
   *     b. Use the most energetic tracks.
   *     c. Try not to group the most energetic tracks together.
   *        Try to group more energetic tracks with less energetic tracks.
   *        We assume collimated bundles here, so this is why.
   *     d. Perform optimally. Do not sort more tracks (by total energy, see b)
   *        than necessary.
   *     e. If n_pairs >= (number of all possible combinations),
   *        do not leave any combinations out.
   *     ( a. and e. are almost but not entirely fulfilled in the current impl )
   *  2. Do a SubsetHSM on the n points.
   */

class SubsetHSMLinearizationPointFinder : public CrossingPtBasedLinearizationPointFinder
{
public:
  SubsetHSMLinearizationPointFinder( const signed int n_pairs = 10 );
  SubsetHSMLinearizationPointFinder( const RecTracksDistanceMatrix * m,
      const signed int n_pairs = -1 );

  virtual SubsetHSMLinearizationPointFinder * clone() const;
};

#endif
