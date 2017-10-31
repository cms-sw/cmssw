#ifndef FsmwLinearizationPointFinder_H
#define FsmwLinearizationPointFinder_H

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
   *  2. Do a Fsmw on the n points.
   */

class FsmwLinearizationPointFinder : public CrossingPtBasedLinearizationPointFinder
{
public:
  /**
   *  \param n_pairs how many track pairs are considered
   *  The weight is defined as w = ( d + cut )^weight_exp
   *  Where d and cut are given in microns.
   *  \param weight_exp exponent of the weight function
   *  \param cut cut parameter of the weight function
   *  \param fraction Fraction that is considered
   */
  FsmwLinearizationPointFinder( signed int n_pairs = 250,
                                float weight_exp = -2., float fraction = .5,
                                float cut=10, int no_weight_above = 10 );

  FsmwLinearizationPointFinder( const RecTracksDistanceMatrix * m,
      signed int n_pairs = 250, float weight_exp = -2., float fraction = .5,
      float cut=10, int no_weight_above = 10 );

  FsmwLinearizationPointFinder * clone() const override;
};

#endif
