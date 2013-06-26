#ifndef DefaultMVFAnnealing_H
#define DefaultMVFAnnealing_H

#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"

class DefaultMVFAnnealing : public GeometricAnnealing {

public:
  /**
   *  Default annealing schedule from mvf
   *  Additionally to change the defaults at construction time,
   *  the values can also be overridden via .orcarcs:
   *
   *  DefaultMVFAnnealing:UseOrcarc=false
   *  If true, orcarcs parameters will be used _instead_
   *  of the parameters given at construction time.
   *
   *  DefaultMVFAnnealing:Cutoff=20.0
   *  DefaultMVFAnnealing:Tini=1024
   *  DefaultMVFAnnealing:Ratio=0.2
   */
  DefaultMVFAnnealing( const double cutoff=9., const double T=1024.,
     const double annealing_ratio=0.2 );
};

#endif
