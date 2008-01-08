#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "VistaTools/FewKDE/interface/Kernel.hh"

#ifndef __KernelDensityEstimate__
#define __KernelDensityEstimate__


class KernelDensityEstimate
{
public:

  // Constructor
  KernelDensityEstimate() {};

  virtual ~KernelDensityEstimate(){}

  // Derive the Kernel Density Estimate from a set of data points "data"
  // with corresponding weights "_wt"
  virtual void derive(const std::vector<std::vector<double> >& data, 
		      const std::vector<double>& _wt = std::vector<double>(0)) = 0;

  // Evaluate the Kernel Density Estimate at a point x
  virtual double evaluate(const std::vector<double>& x) const
  {
    double ans = 0;
    for(size_t i=0; i<weights.size(); i++)
      ans += weights[i]*kernels[i].evaluate(x);
    return(ans);
  }

  // Accessor for individual Kernels
  void getKernels(std::vector<Kernel>& _kernels, std::vector<double>& _weights)
  {
    _kernels = kernels;
    _weights = weights;
    return;
  }


protected:
  std::vector<Kernel> kernels; // vector of Kernels
  std::vector<double> weights; // corresponding weights
  // kernels.size() == weights.size()
};

#endif
