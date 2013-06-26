#include "RecoVertex/MultiVertexFit/interface/DefaultMVFAnnealing.h"

/* Old (ORCA) defaults:

      (false,"DefaultMVFAnnealing:UseOrcarc").value();
        (3.0,"DefaultMVFAnnealing:Cutoff").value();
        (1.0,"DefaultMVFAnnealing:Tini").value();
        (0.5,"DefaultMVFAnnealing:Ratio").value();
*/

DefaultMVFAnnealing::DefaultMVFAnnealing (
     const double cutoff, const double T, const double ratio ) :
  GeometricAnnealing (cutoff, 
                      T, 
                      ratio)
{}
