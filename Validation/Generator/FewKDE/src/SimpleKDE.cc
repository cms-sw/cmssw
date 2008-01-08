#include <algorithm>
#include <cfloat>
#include "VistaTools/Math_utils/interface/Math.hh"
#include "VistaTools/FewKDE/interface/SimpleKDE.hh"
using namespace std;

#ifdef __MAKECINT__
#pragma link C++ class vector<double>;
#pragma link C++ class vector<vector<double> >;
#endif


SimpleKDE::SimpleKDE() { }

// A SimpleKDE is made of a vector of Kernels and their weights
SimpleKDE::SimpleKDE(const std::vector<double>& _weights, 
		     const std::vector<Kernel>& _kernels)
{
  size_t n = _weights.size();
  assert(n==_kernels.size());
  weights = _weights;
  double totalWeight = 0.;
  for(size_t i=0; i<n; i++)
    {
      assert(weights[i]>=0);
      totalWeight += weights[i];
    }
  for(size_t i=0; i<n; i++) // normalize the sum of the weights to one
    weights[i] /= totalWeight;
  kernels = _kernels;
}



// Derive the SimpleKDE from a set of data.
// data.size() = N_data ; data[i].size() = d
// wt.size() = N_data
// wt[i] is the weight of the data point data[i]
void SimpleKDE::derive(const vector<vector<double> >& data, 
		       const vector<double>& _wt)
{
  if(data.empty())
    {
      weights = vector<double>(0);
      kernels = vector<Kernel>(0);
      return;
    }
  int d = data[0].size();
  if((d==0)||(data.size()==1))
    {
      weights = vector<double>(1,1);
      kernels = vector<Kernel>(1,Kernel(vector<double>(d),vector<vector<double> >(d,vector<double>(d))));
      return;
    }
  vector<double> wt = _wt;
  if(wt.size()==0)
    wt = vector<double>(data.size(),1);
  assert(wt.size()==data.size());
  double totalWeight = 0;
  for(size_t i=0; i<wt.size(); i++)
    {
      assert(wt[i]>=0);
      totalWeight += wt[i];
    }
  assert(totalWeight>0);
  for(size_t i=0; i<wt.size(); i++)
    wt[i] /= totalWeight;

  vector<vector<double> > sigma2 = Math::computeCovarianceMatrix(data,0,2);
  //  double N = Math::effectiveNumberOfEvents(wt);
  //  double h = pow(N,-1./(d+4));
  vector<vector<double> > modifiedSigma2 = sigma2;   // need to change sigma2 by h, depending upon how h* (optimal h) is defined.

  weights = wt;
  kernels = vector<Kernel>(data.size());
  for(size_t i=0; i<data.size(); i++)
    kernels[i] = Kernel(data[i], modifiedSigma2);

  return;
}


istream &operator>>(istream& fin, SimpleKDE& kernelEstimate)
{
  int n;
  fin >> n;
  vector<double> weights = vector<double>(n);
  vector<Kernel> kernels = vector<Kernel>(n);
  for(int i=0; i<n; i++)
    {
      fin >> weights[i];
      fin >> kernels[i];
    }
  kernelEstimate = SimpleKDE(weights,kernels);
  return(fin);
}


// Write a SimpleKDE to an output stream
ostream &operator<<(ostream& fout, const SimpleKDE& kernelEstimate)
{
  fout << kernelEstimate.weights.size() << endl;
  for(size_t i=0; i<kernelEstimate.weights.size(); i++)
    {
      fout << kernelEstimate.weights[i] << " ";
      fout << kernelEstimate.kernels[i] << endl;
    }
  return(fout);
}






