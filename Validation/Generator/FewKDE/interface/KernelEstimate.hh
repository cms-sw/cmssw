#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "VistaTools/FewKDE/interface/Kernel.hh"

#ifndef __KernelEstimate__
#define __KernelEstimate__


class KernelEstimate
{
friend std::istream &operator>>(std::istream&, KernelEstimate &);
friend std::ostream &operator<<(std::ostream&, const KernelEstimate &);
public:
  KernelEstimate();
  KernelEstimate(const std::vector<double>& _weights, const std::vector<Kernel>& _kernels);
  KernelEstimate(double w1, const KernelEstimate& ke1, double w2, const KernelEstimate& ke2);
  double evaluate(const std::vector<double>& x) const;
  void enclose(const std::vector<std::vector<double> >& data);
  void derive(const std::vector<std::vector<double> >& data, const std::vector<double>& _wt = std::vector<double>(0));
  KernelEstimate collapse(const std::vector<bool>& integrateOutThisDirection) const;
private:
  std::vector<Kernel> kernels;
  std::vector<double> weights;
  std::vector<std::vector<double> > boundaries;
  int n;
};

#endif
