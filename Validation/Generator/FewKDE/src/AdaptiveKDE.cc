#include <algorithm>
#include <cfloat>
#include "VistaTools/Math_utils/interface/Math.hh"
#include "VistaTools/FewKDE/interface/SimpleKDE.hh"
#include "VistaTools/FewKDE/interface/AdaptiveKDE.hh"
using namespace std;

#ifdef __MAKECINT__
#pragma link C++ class vector<double>;
#pragma link C++ class vector<vector<double> >;
#endif


AdaptiveKDE::AdaptiveKDE() { }


// An AdaptiveKDE is made of a vector of Kernels and their weights
AdaptiveKDE::AdaptiveKDE(const std::vector<double>& _weights, 
			 const std::vector<Kernel>& _kernels)
{
  int n = _weights.size();
  assert(n==_kernels.size());
  weights = _weights;
  double totalWeight = 0.;
  for(int i=0; i<n; i++)
    {
      assert(weights[i]>=0);
      totalWeight += weights[i];
    }
  for(int i=0; i<n; i++) // normalize the sum of the weights to one
    weights[i] /= totalWeight;
  kernels = _kernels;
}



// Derive the AdaptiveKDE from a set of data.
// data.size() = N_data ; data[i].size() = d
// wt.size() = N_data
// wt[i] is the weight of the data point data[i]
void AdaptiveKDE::derive(const vector<vector<double> >& data, 
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

  SimpleKDE simpleKDE;
  simpleKDE.derive(data, _wt);

  vector<double> mu0 = Math::computeMedian(data);
  vector<vector<double> > sigma2 = Math::computeCovarianceMatrix(data,0,2);

  return;
}


istream &operator>>(istream& fin, AdaptiveKDE& kernelEstimate)
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
  kernelEstimate = AdaptiveKDE(weights,kernels);
  return(fin);
}


// Write a AdaptiveKDE to an output stream
ostream &operator<<(ostream& fout, const AdaptiveKDE& kernelEstimate)
{
  int n = kernelEstimate.weights.size();
  fout << n << endl;
  for(int i=0; i<n; i++)
    {
      fout << kernelEstimate.weights[i] << " ";
      fout << kernelEstimate.kernels[i] << endl;
    }
  return(fout);
}






