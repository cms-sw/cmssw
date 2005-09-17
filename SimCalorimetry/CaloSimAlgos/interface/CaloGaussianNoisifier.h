#ifndef CaloGaussianNoisifier_h
#define CaloGaussianNoisifier_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoisifier.h"
#include "CLHEP/Random/RandGaussQ.h"


/**

   \class CaloGaussianNoisifier

   \brief Adds noise to a signal 
   
*/

namespace cms {
  class CaloGaussianNoisifier : public CaloVNoisifier
  {
  public:
    explicit CaloGaussianNoisifier(double sigma);

    virtual void noisify(CaloSamples & frame) const;

  private:
    RandGaussQ theRandomGaussian;
    double theSigma;
  };
}

#endif
