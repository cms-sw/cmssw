#ifndef GeometricAnnealing_H
#define GeometricAnnealing_H

#include "RecoVertex/VertexTools/interface/AnnealingSchedule.h"
#include <vector>

class GeometricAnnealing : public AnnealingSchedule {

public:

  /**
   *  \class GeometricAnnealing.
   *  A very simple class that returns the association probabilty of a (any)
   *  chi2 value, given a cutoff (as a "sigma"), a temperature, and (optionally used) an
   *  annealing ratio ( geometric annealing ).
   */

  GeometricAnnealing( const double cutoff=3.0, const double T=256.0,
     const double annealing_ratio=0.25 );

  void anneal() override; //< One annealing step. theT *= theRatio.
  void resetAnnealing() override; //< theT = theT0.

  /**
   *  phi ( chi2 ) = e^( -.5 * chi2 / T )
   */
  double phi ( double chi2 ) const override;

  /**
   *  Returns phi(chi2) / ( phi(cutoff^2) + phi(chi2) ),
   */
  double weight ( double chi2 ) const override;

  double cutoff() const override;
  double currentTemp() const override;
  double initialTemp() const override;

  /**
   * is it annealed yet?
   */
  bool isAnnealed() const override;

  void debug() const override;

  GeometricAnnealing * clone() const override
  {
    return new GeometricAnnealing ( * this );
  };

private:
  double theT0;
  double theT;
  double theChi2cut;
  double theRatio;

};

#endif
