#ifndef DeterministicAnnealing_H
#define DeterministicAnnealing_H

#include "RecoVertex/VertexTools/interface/AnnealingSchedule.h"
#include <vector>

class DeterministicAnnealing : public AnnealingSchedule {

public:

  /**
   *  \class DeterministicAnnealing.
   *  A very simple class that returns the association probabilty of a (any)
   *  chi2 value, given a cutoff. Default schedule is 256 64 16 4 2 1
   *  Note that cutoff is given "sigma-like", i.e. as a sqrt ( chi2 )!!
   */

  DeterministicAnnealing( float cutoff = 3.0 );
  DeterministicAnnealing( const std::vector < float > & sched, float cutoff = 3.0 );

  void anneal() override; //< One annealing step. theT *= theRatio.
  void resetAnnealing() override; //< theT = theT0.

  /**
   *  phi ( chi2 ) = e^( -.5*chi2 / T )
   */
  double phi ( double chi2 ) const override;

  /**
   *  Returns phi(chi2) / ( phi(cutoff^2) + phi(chi2) ),
   */
  double weight ( double chi2 ) const override;
  
  /**
   * is it annealed yet?
   */
  bool isAnnealed() const override;

  void debug() const override;

  /**
   *  Returns phi(chi2) / ( phi(cutoff^2) + sum_i { phi(chi2s[i]) } )
   */
  // double weight ( double chi2, const vector < double > & chi2s ) const;


  double cutoff() const override;
  double currentTemp() const override;
  double initialTemp() const override;

  DeterministicAnnealing * clone() const override
  {
    return new DeterministicAnnealing ( * this );
  };

private:
  std::vector<float> theTemperatures;
  unsigned int theIndex;
  double theChi2cut;
  bool theIsAnnealed;
};

#endif
