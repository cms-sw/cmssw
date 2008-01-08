#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "VistaTools/FewKDE/interface/Kernel.hh"

#ifndef __AdaptiveKDE__
#define __AdaptiveKDE__


class AdaptiveKDE : public KernelDensityEstimate
{
friend std::istream &operator>>(std::istream&, AdaptiveKDE &);
friend std::ostream &operator<<(std::ostream&, const AdaptiveKDE &);

public:
  AdaptiveKDE();
  AdaptiveKDE(const std::vector<double>& _weights, 
			   const std::vector<Kernel>& _kernels);

  void derive(const std::vector<std::vector<double> >& data, 
	      const std::vector<double>& _wt = std::vector<double>(0));

  //double evaluate(const std::vector<double>& x) const;

  std::string print() const;
  void read(std::istream& fin);

};

#endif
