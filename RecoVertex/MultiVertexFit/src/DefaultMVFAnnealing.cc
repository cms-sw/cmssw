#include "RecoVertex/MultiVertexFit/interface/DefaultMVFAnnealing.h"

namespace {
  bool useOrcarcs()
  {
    static bool ret = false; /* SimpleConfigurable<bool>
      (false,"DefaultMVFAnnealing:UseOrcarc").value();
      */
    return ret;
  }

  double defaultCutoff ( double cutoff )
  {
    if ( useOrcarcs() )
    {
      return 3.0;
      /*
      return SimpleConfigurable<double>
        (3.0,"DefaultMVFAnnealing:Cutoff").value();
        */
    };
    return cutoff;
  }

  double defaultT ( double T )
  {
    if ( useOrcarcs() )
    {
      return 1.0;
      /*
      return SimpleConfigurable<double>
        (1.0,"DefaultMVFAnnealing:Tini").value();
        */
    };
    return T;
  }

  double defaultRatio ( double ratio )
  {
    if ( useOrcarcs() )
    {
      return 0.5;
      /*
      return SimpleConfigurable<double>
        (0.5,"DefaultMVFAnnealing:Ratio").value();
        */
    };
    return ratio;
  }
}

DefaultMVFAnnealing::DefaultMVFAnnealing (
     const double cutoff, const double T, const double ratio ) :
  GeometricAnnealing ( defaultCutoff ( cutoff ), 
                       defaultT ( T ) , 
                       defaultRatio ( ratio ) )
{}
