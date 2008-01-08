#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "VistaTools/FewKDE/interface/KernelDensityEstimate.hh"

#ifndef __SimpleKDE__
#define __SimpleKDE__


class SimpleKDE : public KernelDensityEstimate
{
friend std::istream &operator>>(std::istream&, SimpleKDE &);
friend std::ostream &operator<<(std::ostream&, const SimpleKDE &);

public:
  SimpleKDE();
  SimpleKDE(const std::vector<double>& _weights, 
	    const std::vector<Kernel>& _kernels);

  void derive(const std::vector<std::vector<double> >& data, 
	      const std::vector<double>& _wt = std::vector<double>(0));
//  double evaluate(const std::vector<double>& x) const;

};

#endif
