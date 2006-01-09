#ifndef MU_END_SCA_NOISE_GENERATOR_H
#define MU_END_SCA_NOISE_GENERATOR_H

/** \class CSCScaNoiseGenerator
 * Generate noise for the SCA samples
 *
 * \author Rick Wilkinson
 *
 **/
#include<vector>

class CSCScaNoiseGenerator
{
public:
  explicit CSCScaNoiseGenerator(int nScaBins);
  virtual ~CSCScaNoiseGenerator() {};

  /** returns a list of SCA readings  */
  virtual std::vector<int> getNoise() const;

protected:
  int nScaBins_;
};

#endif
