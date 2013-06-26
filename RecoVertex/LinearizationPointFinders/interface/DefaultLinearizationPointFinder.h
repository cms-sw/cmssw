#ifndef DefaultLinearizationPointFinder_H
#define DefaultLinearizationPointFinder_H

#include "RecoVertex/LinearizationPointFinders/interface/FsmwLinearizationPointFinder.h"

  /** 
   *  The default linearization point finder.
   *  This class is supposed to cover a very wide range of use uses;
   *  this is the class to use, unless you really know that you want 
   *  something else.
   */

class DefaultLinearizationPointFinder : public FsmwLinearizationPointFinder
{
public:
  DefaultLinearizationPointFinder() : 
    FsmwLinearizationPointFinder ( 400, -.5, .4, 10, 5 ) {};

  DefaultLinearizationPointFinder( const RecTracksDistanceMatrix * m ) :
    FsmwLinearizationPointFinder ( m, 400, -.5, .4, 10, 5 ) {};
};

#endif
