#include <algorithm>
#include <cfloat>
#include "VistaTools/Math_utils/interface/Math.hh"
#include "VistaTools/FewKDE/interface/FewKDE.hh"
using namespace std;

#ifdef __MAKECINT__
#pragma link C++ class vector<double>;
#pragma link C++ class vector<vector<double> >;
#endif



class GlobalLikelihood: public Math::FunctionObject
{

public:
  GlobalLikelihood(const vector<double>& _wt, const vector<vector<double> >& _data, const vector<vector<double> >& _boundaries) 
  { 
    wt = _wt; 
    effectiveNumberOfEvents =  Math::effectiveNumberOfEvents(wt);
    data = _data; 
    boundaries = _boundaries; 
  }
  double operator()(const vector<double>& x);
  void unflattenParameters(vector<double>& wt, vector<vector<double> >& mu, vector<vector<vector<double> > >& sigma, const vector<double>& x, int d);
  void flattenParameters(const vector<double>& wt, const vector<vector<double> >& mu, const vector<vector<vector<double> > >& sigma, vector<double>& x);

private:
  vector<double> wt;
  double effectiveNumberOfEvents;
  vector<vector<double> > data;
  vector<vector<double> > boundaries;
};

// Flatten wt, mu, and sigma into a "flat" vector x
void GlobalLikelihood::flattenParameters(const vector<double>& wt, const vector<vector<double> >& mu, const vector<vector<vector<double> > >& sigma, vector<double>& x)
{
  int n = wt.size();
  assert(mu.size()>0);
  int d = mu[0].size();
  x = vector<double>(n*(1+d+d));
  int k=0;
  for(int i=0; i<n; i++)
    {
      x[k++] = wt[i];
      for(int j=0; j<d; j++)
	x[k++] = mu[i][j];
      for(int j=0; j<d; j++)
	  x[k++] = sigma[i][j][j];
    }
  return;
}

// Unflatten the "flat" vector x into the objects wt, mu, and sigma
void GlobalLikelihood::unflattenParameters(vector<double>& wt, vector<vector<double> >& mu, vector<vector<vector<double> > >& sigma, const vector<double>& x, int d)
{
  int n = x.size()/(1+d+d);
  assert(x.size()%n==0);
  int k=0;
  wt = vector<double>(n);
  mu = vector<vector<double> >(n, vector<double>(d));
  sigma = vector<vector<vector<double> > >(n,vector<vector<double> >(d,vector<double>(d)));
  for(int i=0; i<n; i++)
    {
      wt[i] = fabs(x[k++]);
      for(int j=0; j<d; j++)
	{
	  mu[i][j] = x[k++];
	  double range = boundaries[j][1]-boundaries[j][0];
	  while((mu[i][j]<boundaries[j][0]-range)||
		(mu[i][j]>boundaries[j][1]+range))
	    if(mu[i][j]<boundaries[j][0]-range)
	      mu[i][j] = 2*(boundaries[j][0]-range)-mu[i][j];
	    else
	      mu[i][j] = 2*(boundaries[j][1]+range)-mu[i][j];
	}
      for(int j=0; j<d; j++)
	sigma[i][j][j] = fabs(x[k++]);
    }
  assert(k==x.size());
  return;
}

//Compute the probability of seeing the original sample of events given this kernel estimate
double GlobalLikelihood::operator()(const vector<double>& x)
{
  assert(data.size()>0);
  int d = data[0].size();
  vector<double> kernelWeights;
  vector<vector<double> > mu;
  vector<vector<vector<double> > > sigma;
  unflattenParameters(kernelWeights, mu, sigma, x, d);
  vector<Kernel> kernels = vector<Kernel>(0);
  int n = kernelWeights.size();
  for(int i=0; i<n; i++)
    kernels.push_back(Kernel(mu[i],sigma[i],boundaries));
  FewKDE kernelEstimate(kernelWeights,kernels);

  vector<double> sumPerKernel(n), sumSqdPerKernel(n);
  double ans = 0.;
  for(int i=0; i<data.size(); i++)
    {
      double x = max(kernelEstimate.evaluate(data[i]),1e-10);
      ans += wt[i]*log(x);
      for(int j=0; j<n; j++)
	{
	  sumPerKernel[j] += wt[i]*kernels[j].evaluate(data[i]);
	  sumSqdPerKernel[j] += pow(wt[i]*kernels[j].evaluate(data[i]),2.);
	}
    }
  vector<double> numberOfEventsInKernel(n);
  for(int j=0; j<n; j++)
    if(sumSqdPerKernel[j]>0)
      numberOfEventsInKernel[j] = pow(sumPerKernel[j],2.)/sumSqdPerKernel[j];
    else
      numberOfEventsInKernel[j]=0;
  ans = exp(-ans);
  if(*min_element(numberOfEventsInKernel.begin(), numberOfEventsInKernel.end()) < pow(effectiveNumberOfEvents,1./(1.5)))
    ans = FLT_MAX/max(1.,(*min_element(numberOfEventsInKernel.begin(), numberOfEventsInKernel.end())));
  // print(x);  cout << "ans = " << ans << endl;
  return(ans); 
}


FewKDE::FewKDE()
{  
  n = 0;
  setNFewKdeTrials();
}


// A FewKDE is made of a vector of Kernels and their weights
FewKDE::FewKDE(const std::vector<double>& _weights, const std::vector<Kernel>& _kernels)
{
  n = _weights.size();
  setNFewKdeTrials();
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

FewKDE::FewKDE(double w1, const FewKDE& ke1, double w2, const FewKDE& ke2)
{
  setNFewKdeTrials((int)(max(ke1.nFewKdeTrials,ke2.nFewKdeTrials)));
  vector<Kernel> kernels = ke1.kernels;
  kernels.insert(kernels.end(), ke2.kernels.begin(), ke2.kernels.end());
  vector<double> weights;
  for(int i=0; i<ke1.weights.size(); i++)
    weights.push_back(w1*ke1.weights[i]);
  for(int i=0; i<ke2.weights.size(); i++)
    weights.push_back(w2*ke2.weights[i]);
  *this = FewKDE(weights,kernels);
}

// Evaluate the FewKDE at the point x
double FewKDE::evaluate(const std::vector<double>& x) const
{
  double ans = 0;
  for(int i=0; i<n; i++)
    ans += weights[i]*kernels[i].evaluate(x);
  return(ans);
}


void FewKDE::enclose(const std::vector<std::vector<double> >& data)
{
  if(data.empty())
    return;
  int d = data[0].size();

  // compute boundaries
  if(boundaries.empty())
    boundaries = vector<vector<double> >(d);
  vector<vector<double> > dataTranspose(d);
  for(int i=0; i<d; i++)
    {
      dataTranspose[i] = vector<double>(data.size());
      for(int j=0; j<data.size(); j++)
	dataTranspose[i][j] = data[j][i];
    }
  for(int i=0; i<d; i++)
    {
      if(boundaries[i].empty())
	{
	  boundaries[i] = vector<double>(2);
	  boundaries[i][0] = FLT_MAX;
	  boundaries[i][1] = FLT_MIN;
	}
      double low = *min_element(dataTranspose[i].begin(), dataTranspose[i].end());
      double high = *max_element(dataTranspose[i].begin(), dataTranspose[i].end());
      if(boundaries[i][0]>low)
	boundaries[i][0]=low;
      if(boundaries[i][1]<high)
	boundaries[i][1]=high;
    }
  return;
}


/* Derive the FewKDE from a set of data.
   data.size() = N_data ; data[i].size() = d
   wt.size() = N_data
   wt[i] is the weight of the data point data[i]
*/

void FewKDE::derive(const vector<vector<double> >& data, 
		    const vector<double>& _wt)
{
  if(data.empty())
    {
      n=0;
      weights = vector<double>(0);
      kernels = vector<Kernel>(0);
      return;
    }
  int d = data[0].size();
  if((d==0)||(data.size()==1))
    {
      n=1;
      weights = vector<double>(1,1);
      kernels = vector<Kernel>(1,Kernel(vector<double>(d),vector<vector<double> >(d,vector<double>(d))));
      return;
    }
  vector<double> wt = _wt;
  if(wt.size()==0)
    wt = vector<double>(data.size(),1);
  assert(wt.size()==data.size());
  double totalWeight = 0;
  for(int i=0; i<wt.size(); i++)
    {
      assert(wt[i]>=0);
      totalWeight += wt[i];
    }
  assert(totalWeight>0);
  for(int i=0; i<wt.size(); i++)
    wt[i] /= totalWeight;

  enclose(data);

  vector<double> mu0 = Math::computeMedian(data);
  vector<vector<double> > sigma1 = Math::computeCovarianceMatrix(data,0,1);
  vector<vector<double> > sigma2 = Math::computeCovarianceMatrix(data,0,2);
  for(int i=0; i<d; i++)
    for(int j=0; j<d; j++)
      if(i!=j)
	sigma2[i][j] = 0;

  GlobalLikelihood globalLikelihood(wt,data,boundaries);

  vector<double> x, dx;  

  double bestL = FLT_MAX;
  for(int nKernels=1; nKernels<=5; nKernels+=2) //(int)(1+d*log10(Math::effectiveNumberOfEvents(wt)));
    {
      // cout << "            placing " << nKernels << " kernels: " << flush;
      for(int trial=0; trial<nFewKdeTrials; trial++)
	{
	  // cout << trial << " " << flush;
	  vector<double> wt;
	  vector<vector<double> > mu;
	  vector<vector<vector<double> > > sigma;
	  for(int k=0; k<nKernels; k++)
	    {
	      vector<double> mu_k = Math::randMultiGauss(mu0,sigma1);
	      vector<vector<double> > sigma_k = sigma2;
	      for(int i=0; i<d; i++)
		sigma_k[i][i] *= 0.8+0.4*drand48();
	      double wt_k = drand48();
	      wt.push_back(wt_k);
	      mu.push_back(mu_k);
	      sigma.push_back(sigma_k);
	    }
	  globalLikelihood.flattenParameters(wt, mu, sigma, x);
	  double L = Math::minimize(x,&globalLikelihood);
	  globalLikelihood.unflattenParameters(wt, mu, sigma, x, d);
	  if(L<bestL)
	    {
	      bestL = L;
	      kernels = vector<Kernel>(0);
	      weights = wt;
	      n = weights.size();
	      totalWeight = Math::computeSum(weights);
	      for(int i=0; i<weights.size(); i++)
		weights[i] /= totalWeight;
	      for(int i=0; i<n; i++)
		kernels.push_back(Kernel(mu[i],sigma[i],boundaries));
	    }
	}
      // cout << endl;
    }
  // cout << "bestL = " << bestL << endl; cout << *this << endl;

  return;
}


// Read a FewKDE from an input stream
istream &operator>>(istream& fin, FewKDE& kernelEstimate)
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
  kernelEstimate = FewKDE(weights,kernels);
  return(fin);
}

// Write a FewKDE to an output stream
ostream &operator<<(ostream& fout, const FewKDE& kernelEstimate)
{
  fout << kernelEstimate.n << endl;
  for(int i=0; i<kernelEstimate.n; i++)
    {
      fout << kernelEstimate.weights[i] << " ";
      fout << kernelEstimate.kernels[i] << endl;
    }
  return(fout);
}

// Integrate out the dimensions i of the FewKDE for which integrateOutThisDirection[i] == true
FewKDE FewKDE::collapse(const std::vector<bool>& integrateOutThisDirection) const
{
  std::vector<double> _weights = weights;
  std::vector<Kernel> _kernels;
  for(int i=0; i<n; i++)
    _kernels.push_back(kernels[i].collapse(integrateOutThisDirection));
  
  return(FewKDE(_weights, _kernels));
}

void FewKDE::setNFewKdeTrials(int _nFewKdeTrials)
{
  assert(_nFewKdeTrials>=1);
  nFewKdeTrials = _nFewKdeTrials;
}



