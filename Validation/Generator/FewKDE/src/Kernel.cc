#include "VistaTools/Math_utils/interface/Math.hh"
#include "VistaTools/Math_utils/interface/matrix.hh"
#include "VistaTools/FewKDE/interface/Kernel.hh"
#include <cfloat>
using namespace std;

#ifdef __MAKECINT__
#pragma link C++ class vector<double>;
#pragma link C++ class vector<vector<double> >;
#endif


// Read a kernel from an input stream
istream &operator>>(istream& fin, Kernel& kernel)
{
  int d;
  fin >> d;

  vector<double> mu(d);
  for(int i=0; i<d; i++)
    fin >> mu[i];

  vector<vector<double> > sigma(d);
  for(int i=0; i<d; i++)
    sigma[i] = vector<double>(d);
  for(int i=0; i<d; i++)
    for(int j=0; j<d; j++)
      fin >> sigma[i][j];

  vector<vector<double> > boundaries = vector<vector<double> >(d);
  for(int i=0; i<d; i++)
    boundaries[i] = vector<double>(2);
  for(int i=0; i<d; i++)
    fin >> boundaries[i][0] >> boundaries[i][1];

  kernel = Kernel(mu,sigma,boundaries);
  return(fin);
}

// Write a kernel to an output stream
ostream &operator<<(ostream& fout, const Kernel& kernel)
{
  int d = kernel.d;
  fout << d << " ";

  for(int i=0; i<d; i++)
    fout << kernel.mu[i] << " ";

  matrix sigma = kernel.sigma;

  for(int i=0; i<d; i++)
    for(int j=0; j<d; j++)
      fout << sigma[i][j] << " ";

  for(int i=0; i<d; i++)
    fout << kernel.boundaries[i][0] << " " << kernel.boundaries[i][1] << " ";
  return(fout);
}

// Constructors
Kernel::Kernel()
{
}

Kernel::Kernel(const vector<double>& _mu, const vector<vector<double> >& _sigma, const vector<vector<double> >& _boundaries)
{
  d = _mu.size();
  assert(d==_sigma.size());
  mu = _mu;
  sigmaDet = 1;
  sigma = matrix(0,0);
  sigmaInv = matrix(0,0);
  if(d>0)
    {
      for(int i=0; i<d; i++)
	for(int j=i+1; j<d; j++)
	  assert(_sigma[i][j]==_sigma[j][i]);
      sigma = matrix(_sigma);
      sigmaInv = sigma.safeInverse(sigmaDet);
    }
  if(_boundaries.empty()) // if no boundaries, set boundaries at infinity
    {
      boundaries= vector<vector<double> >(d);
      for(int i=0; i<d; i++)
	{
	  boundaries[i] = vector<double>(2);
	  boundaries[i][0] = -FLT_MAX;
	  boundaries[i][1] = FLT_MAX;
	}
    }  
  else
    boundaries = _boundaries;
  assert(d==boundaries.size());
  for(int i=0; i<d; i++)
    assert(boundaries[i][0] <= boundaries[i][1]);

  // Set normalization so that the integral of the kernel inside the boundaries is unity
  norm = 1/sqrt(pow(2*M_PI,1.*d)*sigmaDet);
  for(int i=0; i<d; i++)
    norm /=
      max(1e-6,
	  (0.5+((boundaries[i][1]>mu[i] ? 1 : -1)*erf(sqrt((boundaries[i][1]-mu[i])*sigmaInv[i][i]*(boundaries[i][1]-mu[i])/2)))/2)
	  - (0.5+((boundaries[i][0]>mu[i] ? 1 : -1)*erf(sqrt((boundaries[i][0]-mu[i])*sigmaInv[i][i]*(boundaries[i][0]-mu[i])/2)))/2)
	  );
  assert((norm>0)&&(norm<FLT_MAX));
}


// Evaluate the kernel at a point x
double Kernel::evaluate(const std::vector<double>& x) const
{
  for(int i=0; i<d; i++)
    if((x[i]<boundaries[i][0])||(x[i]>boundaries[i][1]))
      return(0);

  double tsqd = 0;
  for(int i=0; i<d; i++)
    for(int j=0; j<d; j++)
      tsqd += (x[i]-mu[i])*sigmaInv[i][j]*(x[j]-mu[j]);
  double ans = 1;
  if(d>0)
    ans = norm*exp(-tsqd/2.);
  return(ans);
}

// Take the projection of the kernel
// integrateOutThisDirection is a vector of dimension d
// each direction i for which integrateOutThisDirection[i]==true will be integrated over
Kernel Kernel::collapse(const vector<bool>& integrateOutThisDirection) const
{
  assert(d==integrateOutThisDirection.size());
  matrix _sigma = sigma;
  vector<double> _mu = mu;
  vector<vector<double> > _boundaries = boundaries;
  for(int i=d-1; i>=0; i--)
    if(integrateOutThisDirection[i])
      {
	_sigma.deletecolumn(i);
	_sigma.deleterow(i);
	_mu.erase(_mu.begin()+i);
	_boundaries.erase(_boundaries.begin()+i);
      }
  return(Kernel(_mu,_sigma.toSTLVector(),_boundaries));
}
