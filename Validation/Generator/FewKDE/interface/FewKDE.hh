#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "VistaTools/FewKDE/interface/Kernel.hh"
#include "VistaTools/FewKDE/interface/KernelDensityEstimate.hh"

#ifndef __FewKDE__
#define __FewKDE__


class FewKDE : public KernelDensityEstimate
{
friend std::istream &operator>>(std::istream&, FewKDE &);
friend std::ostream &operator<<(std::ostream&, const FewKDE &);

public:

  // Constructors
  FewKDE();
  FewKDE(const std::vector<double>& _weights, 
	 const std::vector<Kernel>& _kernels);
  FewKDE(double w1, const FewKDE& ke1, double w2, const FewKDE& ke2);

  void enclose(const std::vector<std::vector<double> >& data);
  void derive(const std::vector<std::vector<double> >& data, 
	      const std::vector<double>& _wt = std::vector<double>(0));
  double evaluate(const std::vector<double>& x) const;
  FewKDE collapse(const std::vector<bool>& integrateOutThisDirection) const;
  void setNFewKdeTrials(int _nFewKdeTrials = 10);

private:
  std::vector<std::vector<double> > boundaries;
  int n;
  int nFewKdeTrials;

};

#endif
