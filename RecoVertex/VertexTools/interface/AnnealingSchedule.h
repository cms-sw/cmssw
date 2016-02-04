#ifndef AnnealingSchedule_H
#define AnnealingSchedule_H

#include <vector>

class AnnealingSchedule {
  /** Abstract base class that is implemented by the different
   *  annealing schedules.
   */
public:

  virtual ~AnnealingSchedule() {};
  virtual void anneal() = 0; //< One annealing step.
  virtual void resetAnnealing() = 0;

  /**
   *  phi ( chi2 ) = e^( -.5*chi2 / T )
   */
  virtual double phi ( double chi2 ) const = 0;

  /**
   *  Returns phi(chi2) / ( phi(cutoff^2) + phi(chi2) ),
   */
  virtual double weight ( double chi2 ) const = 0;

  /**
   *  Returns phi(chi2) / ( phi(cutoff^2) + sum_i { phi(chi2s[i]) } )
   */
  // double weight ( double chi2, const vector < double > & chi2s ) const;

  /**
   * is it annealed yet?
   */
  virtual bool isAnnealed() const = 0;

  virtual double cutoff() const = 0;
  virtual double currentTemp() const = 0;
  virtual double initialTemp() const = 0;

  virtual void debug() const = 0;

  virtual AnnealingSchedule * clone() const = 0;
};

#endif
