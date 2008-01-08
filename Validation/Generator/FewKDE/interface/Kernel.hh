#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "VistaTools/Math_utils/interface/matrix.hh"

#ifndef __Kernel__
#define __Kernel__


class Kernel
{
friend std::istream &operator>>(std::istream&, Kernel &);
friend std::ostream &operator<<(std::ostream&, const Kernel &);
public:
  Kernel();
  Kernel(const std::vector<double>& _mu, 
	 const std::vector<std::vector<double> >& _sigma, 
	 const std::vector<std::vector<double> >& _boundaries = std::vector<std::vector<double> >(0));
  double evaluate(const std::vector<double>& x) const;
  Kernel collapse(const std::vector<bool>& integrateOutThisDirection) const;

private:
  std::vector<double> mu;
  double norm;
  std::vector<std::vector<double> > boundaries;
  matrix sigma, sigmaInv;
  double sigmaDet;
  int d;
};

#endif
