/******************************
Implementation of Math namespace, containing several handy numerical routines
Bruce Knuteson 2003
******************************/

#include <numeric>
#include <cassert>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <stdio.h>
#include <unistd.h>
#include <cfloat>
#include "VistaTools/Math_utils/interface/Math.hh"
//#include "gsl/gsl_integration.h"
//#include "gsl/gsl_errno.h"
//#include "gsl/gsl_multimin.h"
using namespace std;

#ifdef __MAKECINT__
#pragma link C++ class vector<double>;
#pragma link C++ class vector<vector<double> >;
#endif


string Math::spacePad(int n)
{
  string ans = "";
  for(int i=0; i<n; i++)
    ans += " ";
  return(ans);
}

/// Replace substring x of the string s with the substring y
string Math::replaceSubString(string s, string x, string y)
{
  size_t pos = s.find(x);
  size_t pos1 = 0;
  while(pos!=string::npos)
    {
      s.replace(pos+pos1,x.size(),y);
      pos1 += pos+y.size();
      string s1 = s.substr(pos1);
      pos = s1.find(x);
    }
  return(s);
}

bool Math::MPEquality(double a, double b, double tol)
{
  return(fabs(a-b)<tol);
}

bool Math::MPEquality(const vector<double>& a, const vector<double>& b, double tol)
{
  if(a.size()!=b.size())
    return(false);
  bool ans = true;
  for(size_t i=0; i<a.size(); i++)
    ans = ans && (MPEquality(a[i],b[i],tol));
  return(ans);
}

double Math::addInQuadrature(double a, double b)
{
  return(sqrt(pow(a,2.)+pow(b,2.)));
}

bool Math::isNaNQ(double x)
{
  bool ans = ((x<=0)||(x>=0));
  return(!ans);
}

double Math::deltaphi(double phi1, double phi2)
{
  double ans = fabs(phi1-phi2);
  while(ans>M_PI)
    ans = fabs(ans - 2*M_PI);
  return(ans);
}

extern "C" double poissonConvolutedWithPoisson(double x, void * params)
{
  assert(x>=0);
  double b = ((double *) params)[0];
  assert(b>=0);
  double wt = ((double *) params)[1]; // weight of each monte carlo event
  assert(wt>0);
  int n = (int)(((double*) params)[2]); // number of events observed in the data
  assert(n>=0);
  double f = exp(-x + n * log(x) - lgamma(n+1)) * exp( - x/wt + (b/wt)*log(x/wt) - lgamma((b/wt)+1) ) / wt;
  return f;
}

extern "C" double poissonConvolutedWithGaussian(double x, void * params)
{
  double b = ((double *) params)[0];
  assert(b>=0);
  double deltab = ((double *) params)[1];
  assert(deltab>0);
  int n = (int)(((double*) params)[2]);
  assert(n>=0);
  double f = exp(-fabs(x) + n * log(fabs(x)) - lgamma(n+1)) * 1/(sqrt(2*M_PI)*deltab) * exp(-(pow(x-b,2)/(2*deltab*deltab)));
  return f;
}

 
double Math::roughMagnitudeOfDiscrepancy(int N, double b, double deltab)
{
  double sigma = sqrt(deltab*deltab + max(1.*N,b));
  double ans = (N-b)/sigma;
  return(ans);
}


double Math::poisson(int n, double mean)
{
  //returns the probability to pull n from a Poisson distribution of mean "mean".
  double ans;
  //  if (n <= 5000)
    ans = exp(-mean + n*log(mean) - lgamma(n+1));
    //  else {
    //use stirling's approximation, ln(n!) ~= n*ln(n)-n
    //    ans = exp(-mean + n*log(mean) - (n*log(n)-n));
    //  }
    if(ans > 1) ans=1; //just in case. something like that may happen if we use stirling's approximation.
  return(ans);
}

double Math::ppg(int d, double b0, double deltab)
{ // ppg stands for probability poisson gaussian.
// probability to observe d if bkg=b0+/-deltab. Convolute the poisson with gaussian. The gaussian represents the uncertainty in b0 by deltab. We pull b from that gaussian, and then we pull d from a poisson of mean b.
  // We use a numerical approximation to do the integral of the convolution.
  int k=20; //the bigger the k the finer the sampling, the longer it takes. The integration step is equal to deltab/k.
  int s=8; // number of standard deviations (deltab) up and down from b0 to sample. The more the better and the slower.
  double renormalizationFactor=2/(1+erf(b0/sqrt(2.)/deltab)); // this factor increases the positive piece of the gaussian G(b-b0,sigmab) to compensate for the area that is lost because it falls on the negative side of b.
  double ans = 0;
  for(int n=-s*k; n<=s*k; ++n){
    if(b0+n*deltab/k > 0)
      ans += 0.5*(erf((n+0.5)/k/sqrt(2.))-erf((n-0.5)/k/sqrt(2.))) * poisson(d,b0+n*deltab/k) * renormalizationFactor ;
  }
  return(ans);
}

double Math::accurateMagnitudeOfDiscrepancy(int d, double b0, double deltab)
{
  int lowerSumLimit=0;
  int upperSumLimit=0;
  double sign=0;
  double sigmaInGaussianApproximation=sqrt(deltab*deltab+b0);
  double effectProbability=0;
  if(d>b0) {
    sign=1;
    lowerSumLimit=d;
    upperSumLimit=(int)(b0+10*sigmaInGaussianApproximation);
    double tmpppg=0;
    for (int n=lowerSumLimit; n<=upperSumLimit; ++n) {
      tmpppg=ppg(n,b0,deltab);
      effectProbability += tmpppg;
      if (tmpppg/effectProbability < 1e-50) break;
    }
  }
  else {
    sign=-1;
    upperSumLimit=d;
    lowerSumLimit=(int) max(0,b0-10*sigmaInGaussianApproximation);
    double tmpppg=0;
    for (int n=upperSumLimit; n>=lowerSumLimit; --n) {
      tmpppg=ppg(n,b0,deltab);
      effectProbability += tmpppg;
      if (tmpppg/effectProbability < 1e-50) break;
    }
  }

  return(sign*prob2sigma(effectProbability)); //return number of sigmas, with sign to indicate excess or deficit...
}


double Math::probOfThisEffect(double b, int N, double deltab, string opt)
{

  // Return the probability of background b +- deltab fluctuating up to or above the observed number of events N

  double answer = 0.;
  assert(N>=0);
  if((N==0)&&(opt==">="))
    return(1.);

  /* This does a rough integration of the poisson probability p(bprime,N)
     convoluted with probability density gauss(b, deltab) for bprime */

  const int numberOfIntegrationPoints = 10;
  double integrationPoints[numberOfIntegrationPoints] = 
  {-1.64485,-1.03643,-0.67449,-0.38532,-0.125661,0.125661,0.38532,0.67449,1.03643,1.64485}; 
  // this from Mathematica:  aerf[y_]:=InverseErf[y*2.-1]*Sqrt[2.]; Table[aerf[t+.05], {t, 0., .9, .1}];

  // Comment out the above, and uncomment the below for better accuracy
  /*
   const int numberOfIntegrationPoints = 100;
   double integrationPoints[numberOfIntegrationPoints] = 
  { -2.57583, -2.17009, -1.95996, -1.81191, -1.6954, -1.59819, -1.5141, -1.43953, -1.3722, -1.31058, -1.25357, -1.20036, -1.15035, -1.10306, -1.05812, -1.01522, -0.974114, -0.934589, -0.896473, -0.859617, -0.823894, -0.789192, -0.755415, -0.722479, -0.690309, -0.658838, -0.628006, -0.59776, -0.568051, -0.538836, -0.510073, -0.481727, -0.453762, -0.426148, -0.398855, -0.371856, -0.345126, -0.318639, -0.292375, -0.266311, -0.240426, -0.214702, -0.189118, -0.163658, -0.138304, -0.113039, -0.0878448, -0.0627068, -0.0376083, -0.0125335, 0.0125335, 0.0376083, 0.0627068, 0.0878448, 0.113039, 0.138304, 0.163658, 0.189118, 0.214702, 0.240426, 0.266311, 0.292375, 0.318639, 0.345126, 0.371856, 0.398855, 0.426148, 0.453762, 0.481727, 0.510073, 0.538836, 0.568051, 0.59776, 0.628006, 0.658838, 0.690309, 0.722479, 0.755415, 0.789192, 0.823894, 0.859617, 0.896473, 0.934589, 0.974114, 1.01522, 1.05812, 1.10306, 1.15035, 1.20036, 1.25357, 1.31058, 1.3722, 1.43953, 1.5141, 1.59819, 1.6954, 1.81191, 1.95996, 2.17009, 2.57583 };
  // this from Mathematica:  aerf[y_]:=InverseErf[y*2.-1]*Sqrt[2.]; Table[aerf[t+.005], {t, 0., .99, .01}] //N[#,4]& ;
  */

  for(int i=0; i<numberOfIntegrationPoints; i++)
    {
      double bprime = b + integrationPoints[i]*deltab;
      if(bprime<0)
	bprime = fabs(bprime); // reflect about zero
      double ans = (N==0);

      if((bprime < 25.)&&(bprime>0.)) // the problem is Poisson
	{
	  if(opt==">=")
	    {
	      ans = 1.;
	      for(int k=0; k<N; k++)
		{
		  double logfact=0.;
		  for(int j=1; j<=k; j++)
		    logfact+=log((double)j);
		  ans-=exp(-bprime+ k*log(bprime)-logfact);
		}
	    }
	  if(opt=="==")
	    ans=exp(-bprime+ N*log(bprime)-lgamma(N+1));
	}
      if(bprime >= 25.) // the problem is Gaussian
	{
	  if(opt==">=")
	    {
	      double t = (N - bprime) / sqrt(bprime);
	      ans = sigma2prob(t);
	    }
	  if(opt=="==")
	    {
	      double deltabprime = sqrt(bprime);
	      ans = 1./(sqrt(2*M_PI)*deltabprime)*exp(-(N-bprime)*(N-bprime)/(2*deltabprime*deltabprime));
	    }
	}

      // Check to make sure that the answer is physical (it should always be)

      if((ans>1.)&&(ans<1.+1.e-12))
	ans=1.;
      if((ans<0.)&&(ans>-1.e-12))
	ans=0.;
      assert(ans<=1.);
      assert(ans>=0.);

      answer += ans;
    }

  answer /= numberOfIntegrationPoints;
  return(answer);
}

double Math::bkgFromEffect(double effect, int N)
{

  /* This is a very inefficient way to invert the function probOfThisEffect.
     This function is the inverse of probOfThisEffect in the sense that
     bkgFromEffect(probOfThisEffect(b, N), N) == b */

  double bkgHi=4.*N;
  double bkgLo=0.;
  double bkgMid=0.;

  // using the fact that probOfThisEffect(b,N) is monotonic in b, we perform a binary search for the root of the equation probOfThisEffect(b,N) == effect

  while((bkgHi-bkgLo)>.000001)
    {
      bkgMid=(bkgHi+bkgLo)/2.;
      double effectTmp=Math::probOfThisEffect(bkgMid,N);
      if(effectTmp<effect)
	bkgLo=bkgMid;
      else
	bkgHi=bkgMid;
    }
  return(bkgMid);
}

double Math::sigma2prob(double s)
{

  // Convert s, in units of standard deviation, to a probability

  double x = s/sqrt(2.);
  double p=.5*erfc(x); // equivalent to p=.5*(1-erf(x)); 
  return(p);
}

double Math::prob2sigma(double p)
{

  /* This is a very inefficient way to invert the function sigma2prob.
     This function is the inverse of sigma2prob in the sense that
     prob2sigma(sigma2prob(s)) == s and 
     sigma2prob(prob2sigma(p)) == p if p<.5 */

  double sigmaLo=-10.;
  double sigmaHi=10.;
  double sigma=0;
  double deltaSigma=100; //something big
  double tol=.0001;

  // using the fact that sigma2prob(s) is monotonic in s, we perform a binary search for the root of the equation sigma2prob(s) == p 

  while(deltaSigma>tol)
    {
      double ps=sigma2prob(sigma);
      if(p<ps)
	sigmaLo=sigma;
      else
	sigmaHi=sigma;
      sigma=(sigmaHi+sigmaLo)/2.;
      deltaSigma=fabs((sigmaHi-sigmaLo)/sigma);
    }
  return(sigma);
}

double Math::gasdev(double mu, double sigma)
{

  // This function returns a random double from a gaussian distribution with mean mu and standard deviation sigma.  
  // Taken from Numerical Recipes in C (http://www.nr.com)

        static int iset=0;
        static double gset;
        double fac,rsq,v1,v2;
        double ans;
        if  (iset == 0) {
                do {
                        v1=2.0*(double)drand48()-1.0;
                        v2=2.0*(double)drand48()-1.0;
                        rsq=v1*v1+v2*v2;
                } while (rsq >= 1.0 || rsq == 0.0);
                fac=sqrt(-2.0*log(rsq)/rsq);
                gset=v1*fac;
                iset=1;
                ans=v2*fac;
        } else {
                iset=0;
                ans=gset;
        }
        ans=ans*sigma+mu;
        return(ans);
}

vector<double> Math::randMultiGauss(const vector<double>& mu, const vector<vector<double> >& sigma)
{
  if((sigma.size()==0)||(mu.size()==0))
    return(vector<double>(0));
  assert(sigma.size()==sigma[0].size());
  assert(sigma.size()==mu.size());
  int d=mu.size();
  assert(d>0);
  vector<double> ans = vector<double>(d,0.);
  vector<double> r = vector<double>(d);
  for(int i=0; i<d; i++)
    r[i] = gasdev(0,1);
  for(int i=0; i<d; i++)
    for(int j=0; j<d; j++)
      ans[i] += sigma[i][j] * r[j];
  for(int i=0; i<d; i++)
    ans[i] += mu[i];
  return(ans);
}  


vector<double> Math::randMultiGauss(const vector<double>& mu, const matrix & sigma)
{
  assert(sigma.nrows()==sigma.ncols());
  assert(sigma.nrows()==mu.size());
  int d=mu.size();
  if(d==0)
    return(vector<double>(0));
  assert(d>0);
  vector<double> ans = vector<double>(d,0.);
  vector<double> r = vector<double>(d);
  for(int i=0; i<d; i++)
    r[i] = gasdev(0,1);
  for(int i=0; i<d; i++)
    for(int j=0; j<d; j++)
      ans[i] += sigma[i][j] * r[j];
  for(int i=0; i<d; i++)
    ans[i] += mu[i];
  return(ans);
}

double Math::expdev(double lambda)
{
  return(-log(drand48())*lambda);
}

double Math::computeSum(const vector<double>& x)
{
  double ans = 0.;
  for(size_t i=0; i<x.size(); i++)
    ans += x[i];
  return(ans);
}
double Math::computeAverage(const vector<double>& x)
{
  if(x.size()==0)
    return(0);
  return(computeSum(x)/x.size());
}
double Math::computeAverage(vector<int> x)
{
  double ans = 0.;
  for(size_t i=0; i<x.size(); i++)
    ans += x[i];
  ans /= x.size();
  return(ans);
}
long double Math::computeAverage(vector<long double> x)
{
  long double ans = 0.;
  for(size_t i=0; i<x.size(); i++)
    ans += x[i];
  ans /= x.size();
  return(ans);
}

double Math::computeRMS(vector<double> x)
{
  double average = computeAverage(x);
  double rms = 0.;
  for(size_t i=0; i<x.size(); i++)
    rms += pow(x[i]-average,2);
  rms = sqrt(rms/x.size());
  return(rms);
}
long double Math::computeRMS(vector<long double> x)
{
  long double average = computeAverage(x);
  long double rms = 0.;
  for(size_t i=0; i<x.size(); i++)
    rms += pow(x[i]-average,2);
  rms = sqrt(rms/x.size());
  return(rms);
}

double Math::effectiveNumberOfEvents(const vector<double> & wt)
{
  double weightSumSqd=0, totalWeight=0, Neff=0;
  for(size_t i=0; i<wt.size(); i++)
    totalWeight += wt[i];
  for(size_t i=0; i<wt.size(); i++)
    weightSumSqd += pow(wt[i]/totalWeight,2.);
  if(weightSumSqd>0)
    Neff = 1./weightSumSqd;
  assert((size_t)Neff<=wt.size());
  return(Neff);
} 

double Math::computeMCerror(double sum, double n, double epsilonWt)
{
  if(n<1)
    n=1;
  double error =
    1 * sum / sqrt(n) + 
    2 * epsilonWt; 
  return(error);
}

double Math::computeMCerror(const vector<double> & wt, double epsilonWt)
{
  double sum = Math::computeSum(wt);
  double n = Math::effectiveNumberOfEvents(wt)+1;
  return(Math::computeMCerror(sum,n,epsilonWt));
}


int Math::poisson(double mu)
{

  // This function returns an integer pulled from a poisson distribution with mean mu

  assert(mu >= 0.);
  if(mu == 0)
    return(0);

  // If mu is large, the problem is gaussian

  if(mu > 12.)
    {
      int ans = (int)(gasdev(mu, sqrt(mu))+.5);
      if(ans<0)
	ans = 0;
      return(ans);
    }

  // If mu is small, the problem is poisson

  double r = drand48();
  int n = 0;
  double lognfactorial = 0;
  double P = 0.;
  while( r > (P += exp(-mu + n*log(mu) - lognfactorial) ))
    {
      n++;
      lognfactorial += log((double)n);
    }
  return(n);
}	   

int Math::fluctuate(double mean, double systematicError)
{

  // This function is similar to poisson, but it convolves the poisson with a gaussian to account for the systematic error

  double nEventsExpected = gasdev(mean, systematicError);
  if(((mean>=0.)&&(nEventsExpected<0))||
     ((mean<0)&&(nEventsExpected>0)))
    nEventsExpected = 0.;
  int answer = poisson(abs(nEventsExpected));
  if(nEventsExpected<0)
    answer*= -1;
  return(answer);
}

double Math::gamma(double x)
{

  // Compute the gamma function for integer or half-integer arguments

  assert(x>.49);

  // gamma(.5) == sqrt(pi)

  if((x>.49)&&(x<.51))
    return(sqrt(3.141592));
  
  // gamma(1.) == 1.

  if(x<1.01)
    return(1.);

  // recursion relation:  gamma(x) = (x-1) gamma(x-1)

  return((x-1)*gamma(x-1));
}
  
double Math::volumeOfUnitSphere(int d)
{

  // Compute the volume of the d-dimensional unit sphere

  double ans= pow(3.141592,d/2.)/((d/2.)*gamma(d/2.));
  return(ans);
}

double Math::toleranceRound(double x, double tol)
{
  double ans = ((long int)((x+(x<0 ? -1 : +1)*tol/2)/tol))*tol;
  return(ans);
}

double Math::sigFigRound(double x, int nSigFigs)
{

  // Round x to nSigFigs significant digits
  
  if(x==0)
    return(0);
  int sign = (x>0) - (x<0);
  x = fabs(x);
  int a = (int)(log10(x) + (x>=1.));
  double ans = (long int)(x/pow(10.,a-nSigFigs) + .5) * pow(10.,a-nSigFigs);
  ans *= sign;
  return(ans);
}

double Math::nice(double x, int addedprecision)
{
  if(x==0)
    return(0);
  int prec = 2;
  double sgn = x/fabs(x);
  x = fabs(x);
  if(x!=0)
    prec = (int)floor(-log10(x)+1.5);
  if(prec<1) prec=1;
  prec += addedprecision;
  double ans = sgn*((int)(pow(10.,prec)*x+0.5))/pow(10.,prec);
  return ans;
}

string Math::ftoa(double x)
{
  char a[100];
  sprintf(a,"%g",x);
  return(a);
}


vector<int> Math::getDigits(int n, int base, int size)
{
  assert(base>0);
  assert(size>=0);
  assert(n>=0);
  if(base==1)
    return(vector<int>(size,0));
  vector<int> ans;
  int nDigits = 0;
  int baseToTheN = 1;
  while(true)
    {
      if(n>=baseToTheN)
	{
	  baseToTheN *= base;
	  ans.push_back((n%baseToTheN)/(baseToTheN/base));
	  nDigits++;
	}
      else
	break;
    }
  for(int i=nDigits; i<size; i++)
    ans.push_back(0);
  return(ans);
}

vector<int> Math::getDigits(int n, vector<int> base, int size)
{
  for(size_t i=0; i<base.size(); i++)
    assert(base[i]>1);
  assert(size>=0);
  assert(n>=0);
  vector<int> ans;
  size_t nDigits = 0;
  int baseToTheN = 1;
  while(true)
    {
      if(n>=baseToTheN)
	{
	  if(nDigits<base.size())
	    {
	      baseToTheN *= base[nDigits];
	      ans.push_back((n%baseToTheN)/(baseToTheN/base[nDigits]));
	      nDigits++;
	    }
	  else
	    {
	      ans.push_back(n/(baseToTheN));
	      nDigits++;
	      break;
	    }
	}
      else
	break;
    }
  for(int i=nDigits; i<size; i++)
    ans.push_back(0);
  return(ans);
}

vector<int> Math::integerNthRoot(int a, int n)
{
  vector<int> ans(n, (int)pow(a, 1./n));
  for(int i=0; i<n; i++)
    {
      int m=1;
      for(int j=0; j<n; j++)
	if(i==j)
	  m *= ans[j]+1;
	else
	  m *= ans[j];
      if(m < a)
	ans[i]++;
    }
  return(ans);
}

double Math::calculateSmoothingParameter(double fractionalWeight, int nvars)
{

  // Calculate h, the smoothing parameter in PDE.  A discussion of this extension to traditional kernel estimation methods may be found in the Sherlock algorithms note.

  assert(fractionalWeight>=0.);
  double h = 1.0*pow(fractionalWeight,1./(nvars+4));
  return(h);
}


double Math::eta2theta(double eta) { return 2.*atan(exp(-eta)) ; }
double Math::theta2eta(double theta) { return -log(tan(theta/2.)) ; }

void Math::tossAwayTail(vector<vector<double> >& events, double alpha) // toss away a fraction alpha of the events, from the tails of the distributions
{
  int nevents = events.size();
  if(nevents==0)
    return;
  int nvars = events[0].size();
  vector<vector<vector<double> > > a = vector<vector<vector<double> > >(nvars);
  for(int i=0; i<nvars; i++)
    {
      a[i] = vector<vector<double> >(nevents);
      for(int j=0; j<nevents; j++)
	{
	  a[i][j] = vector<double>(2);
	  a[i][j][0] = events[j][i];
	  a[i][j][1] = j;
	}
    }
  for(int i=0; i<nvars; i++)
    sort(a[i].begin(), a[i].end());
  vector<int> eventsToToss;
  for(int k=0; k<nevents*alpha/(2*nvars); k++)
    {
      for(int i=0; i<nvars; i++)
	{
	  eventsToToss.push_back((int)a[i][0][1]);
	  eventsToToss.push_back((int)a[i][a[i].size()-1][1]);
	  a[i].erase(a[i].begin());
	  a[i].erase(a[i].end()-1);
	}
    }
  vector<vector<double> > remainingEvents;
  for(int i=0; i<nevents; i++)
    {
      bool keepThisEvent = true;
      for(size_t j=0; j<eventsToToss.size(); j++)
	if(i==eventsToToss[j])
	  keepThisEvent = false;
      if(keepThisEvent)
	remainingEvents.push_back(events[i]);
    }
  events = remainingEvents;
  return;
}


vector<double> Math::computeMedian(vector<vector<double> > events)
{
  int nevents = events.size();
  if(nevents==0)
    return(vector<double>(0));
  int nvars = events[0].size();
  vector<vector<double> > a = vector<vector<double> >(nvars);
  for(int i=0; i<nvars; i++)
    {
      a[i] = vector<double>(nevents);
      for(int j=0; j<nevents; j++)
	a[i][j] = events[j][i];
    }
  vector<double> median = vector<double>(nvars);
  for(int i=0; i<nvars; i++)
    {
      sort(a[i].begin(), a[i].end());
      median[i] = a[i][a[i].size()/2];
    }
  return(median);
}

double Math::sgnpow(double x, double n)
{
  if(x==0)
    return(0);
  return( x/fabs(x) * pow(fabs(x),n) );
}

vector<vector<double> > Math::computeCovarianceMatrix(vector<vector<double> > events, double alpha, int lNorm)
{
  tossAwayTail(events,alpha);
  vector<double> average = computeMedian(events);
  if(events.size()==0)
    return(vector<vector<double> >(0));
  int nvars = events[0].size();
  vector<vector<double> > covarianceMatrix = vector<vector<double> >(nvars);
  for(int j=0; j<nvars; j++)
    {
      covarianceMatrix[j] = vector<double>(nvars);
      for(int k=0; k<nvars; k++)
	for(size_t i=0; i<events.size(); i++)
	  covarianceMatrix[j][k] += sgnpow((events[i][j] - average[j])*(events[i][k] - average[k]),lNorm/2.)/events.size();
    }
  return(covarianceMatrix);
}


vector<vector<double> > Math::computeCorrelationMatrix(vector<vector<double> > events, double alpha, int lNorm)
{
  vector<vector<double> > covarianceMatrix = computeCovarianceMatrix(events, alpha, lNorm);
  int nvars = covarianceMatrix.size();
  vector<vector<double> > correlationMatrix = covarianceMatrix;
  for(int j=0; j<nvars; j++)
    for(int k=0; k<nvars; k++)
      correlationMatrix[j][k] = covarianceMatrix[j][k] / sqrt(covarianceMatrix[j][j]*covarianceMatrix[k][k]); 

  return(correlationMatrix);
}


vector<string> Math::vectorizeString(string s, string separator)
{
  vector<string> ans;
  size_t i1 = 0;
  for(size_t i2=0; i2<s.size(); i2++)
    {
      if((i2+separator.size()<s.size()) &&
	 (s.substr(i2,separator.size())==separator))
	{
	  ans.push_back(s.substr(i1,i2-i1));
	  i1=i2+separator.size();
	  i2 += separator.size()-1;
	}
    }
  if(i1<s.size())
    ans.push_back(s.substr(i1));
  return(ans);
}

void Math::loadMatrixFromFile(string filename, vector<vector<double> >& events)
{
  int nvars;
  events = vector<vector<double> >(0);
  ifstream fin(filename.c_str());
  if(!fin)
    return;

  // count the number of columns in the first line

  string s;
  getline(fin,s);
  nvars=0;
  for(size_t i=0;i<s.size();i++)
    if((s.substr(i,1)!=" ")&&(s.substr(i,1)!="\t")&&
       ((i==0)||(s.substr(i-1,1)==" ")||(s.substr(i-1,1)=="\t"))) nvars++;
  fin.close();
  if(nvars==0)
    return;

  // read in the data

  ifstream fdata(filename.c_str());
  vector<double> v(nvars);
  while(fdata >> v[0])
    {
      for(int i=1; i<nvars; i++)
	fdata >> v[i];
      events.push_back(v);
    }
  fdata.close();

  return;
}

void Math::loadMatrixFromFile(string filename, vector<vector<double> >& events, vector<double>& weights)
{
  int nvars;
  events = vector<vector<double> >(0);
  ifstream fin(filename.c_str());
  if(!fin)
    return;

  // count the number of columns in the first line

  string s;
  getline(fin,s);
  nvars=-1;
  for(size_t i=0;i<s.size();i++)
    if((s.substr(i,1)!=" ")&&(s.substr(i,1)!="\t")&&
       ((i==0)||(s.substr(i-1,1)==" ")||(s.substr(i-1,1)=="\t"))) nvars++;
  fin.close();
  if(nvars<=0)
    return;

  // read in the data

  ifstream fdata(filename.c_str());
  vector<double> v(nvars);
  double w;
  while(fdata >> w)
    {
      weights.push_back(w);
      for(int i=0; i<nvars; i++)
	fdata >> v[i];
      events.push_back(v);
    }
  fdata.close();

  return;
}

// Numerical recipes in C

#define NRANSI
#define MINMAX +1 // +-1 for minimization, maximization, respectively

double Math::amotry(matrix & p, vector<double> & y, vector<double> & psum, int ndim, Math::FunctionObject* funk, int ihi, double fac)
{
	int j;
	double fac1,fac2,ytry;
	vector<double> ptry(ndim);

	fac1=(1.0-fac)/ndim;
	fac2=fac1-fac;
	for (j=0;j<ndim;j++) ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
	ytry=MINMAX*funk->operator()(ptry);
	if (ytry < y[ihi]) {
		y[ihi]=ytry;
		for (j=0;j<ndim;j++) {
			psum[j] += ptry[j]-p[ihi][j];
			p[ihi][j]=ptry[j];
		}
	}
	return ytry;
}


//#define NMAX 5000
//#define ftol 1e-6
#define GET_PSUM \
					for (j=0;j<ndim;j++) {\
					for (sum=0.0,i=0;i<mpts;i++) sum += p[i][j];\
					psum[j]=sum;}
#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}

void Math::amoeba(matrix & p, vector<double> & y, int ndim, 
	Math::FunctionObject* funk, double ftol, int NMAX, bool stoppable)
{
  int ihi,ilo,inhi,j,mpts=ndim+1;
  double rtol,sum,swap,ysave,ytry;
  vector<double> psum(ndim);

  int nfunk=0;
  for(size_t i=0; i<y.size(); i++) y[i]= MINMAX*y[i];
  int i;

  GET_PSUM
    for (;;) {
      ilo=0;
      ihi = y[0] > y[1] ? (inhi=1,0) : (inhi=0,1);
      for (i=0;i<mpts;i++) {
	if (y[i] <= y[ilo]) ilo=i;
	if (y[i] > y[ihi]) {
	  inhi=ihi;
	  ihi=i;
	} else if (y[i] > y[inhi] && i != ihi) inhi=i;
      }
      //cout << "y[ihi]=" << y[ihi] << endl;
      //cout << "y[ilo]=" << y[ilo] << endl;
      rtol=2.0*fabs(y[ihi]-y[ilo])/(fabs(y[ihi])+fabs(y[ilo]));
      if ((rtol < ftol)||(nfunk>=NMAX)||
	  (stoppable&&(ifstream("stop").good()))) {
	//cout << "rtol=" << rtol << endl;
	SWAP(y[0],y[ilo])
	  for (i=0;i<ndim;i++) SWAP(p[0][i],p[ilo][i])
	    for (size_t i=0; i<y.size(); i++) y[i]= MINMAX * y[i];
	break;
      }
      //		/*
      if (nfunk >= NMAX) 
	{
	  cout << "NMAX exceeded";
	  exit(1);
	}
      //		*/
      nfunk += 2;
      ytry=amotry(p,y,psum,ndim,funk,ihi,-1.0);
      if (ytry <= y[ilo])
	ytry=amotry(p,y,psum,ndim,funk,ihi,2.0);
      else if (ytry >= y[inhi]) {
	ysave=y[ihi];
	ytry=amotry(p,y,psum,ndim,funk,ihi,0.5);
	if (ytry >= ysave) {
	  for (i=0;i<mpts;i++) {
	    if (i != ilo) {
	      for (j=0;j<ndim;j++)
		p[i][j]=psum[j]=0.5*(p[i][j]+p[ilo][j]);
	      y[i]=MINMAX*funk->operator()(psum);
	    }
	  }
	  nfunk += ndim;
	  GET_PSUM
	    }
      } else --(nfunk);
    }
  // cout << "function calls: " << nfunk << endl;
}


#undef SWAP
#undef GET_PSUM
#undef NMAX
#undef NRANSI

double Math::minimize(vector<double>& x, Math::FunctionObject* funk, vector<double> dx, double tol, bool stoppable)
{
  int n = x.size();
  matrix p(n+1,n);
  double ans = 0;
  if(dx.empty())
    {
      dx = vector<double>(n);
      for(int i=0; i<n; i++)
	dx[i] = 0.1*x[i];
    }
  
  for(int i=0; i<n+1; i++)
    for(int j=0; j<n; j++)
      if(i==j+1)
	p[i][j] = x[j] + dx[j];
      else
	p[i][j] = x[j];
  vector<double> y(n+1);
  for(int i=0; i<n+1; i++)
    y[i] = funk->operator()(p[i]);
  Math::amoeba(p, y, n, funk, tol, 10000, stoppable);
  x = p[0];
  ans = y[0];  
  return(ans);
}


double Math::binomialError(double p, int N)
{
  assert(p>=0);
  assert(p<=1);
  double ans = sqrt(p*(1-p)/N);
  return(ans);
}

double Math::deltaR(double phi1, double eta1, double phi2, double eta2)
{
  double dphi = fabs(phi1-phi2);
  while(dphi>M_PI) dphi = fabs(2*M_PI-dphi);
  double deta = fabs(eta1-eta2);
  double dR = sqrt(dphi*dphi + deta*deta);
  return(dR);
}

void Math::makeNiceHistogramRange(vector<double> a, int& nbins, double& lo, double& hi)
{
  if(a.empty())
    {
      lo=hi=nbins=0;
      return;
    }
  sort(a.begin(), a.end());
  for(size_t i=0; i<a.size(); i++)
    if(a[i]==FLT_MAX)
      a.erase(a.begin()+i,a.end());

  if(a[0]==a[a.size()-1])
    {
      nbins = 1;
      lo = a[0] -1;
      hi = a[0] +1;
      return;
    }
  nbins = (int)(pow((double)a.size(),1./3));
  lo = max(a[0],a[a.size()/2]-3*(a[a.size()/2]-a[a.size()/5]));
  hi = min(a[a.size()-1],a[a.size()/2]+3*(a[4*a.size()/5]-a[a.size()/2]));
  double binWidth = (hi-lo)/nbins;
  double orderOfMagnitude = pow(10.,(double)((int)(log10(binWidth)-(log10(binWidth)<0))));
  double x = binWidth / orderOfMagnitude;
  
  assert(x>=1);
  if(x>5) 
    x=10;
  else
    if(x>2) x=5;
    else
      if(x>1) x=2;
  binWidth = x * orderOfMagnitude;
  lo = ((int)(lo/binWidth))*binWidth;
  hi = ((int)(hi/binWidth+1))*binWidth;
  nbins = (int)((hi-lo)/binWidth+0.01);
  return;
}
  
void Math::makeNiceHistogramRange(
				  const vector<double>& bkgWeights,
				  const vector<double>& sigWeights,
				  const vector<vector<double> >& bkgEvents, 
				  const vector<vector<double> >& sigEvents, 
				  const vector<vector<double> >& dataEvents,
				  vector<vector<double> >& range, vector<int>& nbins		
				  )
{

  double s = Math::computeSum(sigWeights);
  double b = Math::computeSum(bkgWeights);
  //int N = dataEvents.size();

  int d = 0;
  if(!dataEvents.empty())
    d = dataEvents[0].size();
  else if(!sigEvents.empty())
    d = sigEvents[0].size();
  else if(!bkgEvents.empty())
    d = bkgEvents[0].size();
	    
  // loop over variables
  range = vector<vector<double> >(0);
  nbins = vector<int>(0);
  for(int k=0; k<d; k++)
    {		
      vector<double> xrange(2); 
      xrange[0]=+1.e8; xrange[1]=-1.e8;
      vector< pair<double,double> > sigDist;
      vector< pair<double,double> > bkgDist;
      vector< pair<double,double> > dataDist;
      for(size_t i=0; i<sigEvents.size(); i++)
	if(sigEvents[i][k]!=FLT_MAX)
	  sigDist.push_back(pair<double,double>(sigEvents[i][k],sigWeights[i]/s));
      for(size_t i=0; i<bkgEvents.size(); i++)
	if(bkgEvents[i][k]!=FLT_MAX)
	  bkgDist.push_back(pair<double,double>(bkgEvents[i][k],bkgWeights[i]/b));
      for(size_t i=0; i<dataEvents.size(); i++)
	if(dataEvents[i][k]!=FLT_MAX)
	  dataDist.push_back(pair<double,double>(dataEvents[i][k],1./dataEvents.size()));
      if(!sigDist.empty())
	sort(sigDist.begin(), sigDist.end());
      if(!bkgDist.empty())
	sort(bkgDist.begin(), bkgDist.end());
      if(!dataDist.empty())
	sort(dataDist.begin(), dataDist.end());
      double wts=0., wtb=0., wtd=0.;
      double fractionOfMCEventsThatSpillOffPlot = 0.01; // 0.05
      double fractionOfDataEventsThatSpillOffPlot = 0.03; // 0.01
      for(size_t i=0; i<sigDist.size(); i++)
	{
	  if((wts<fractionOfMCEventsThatSpillOffPlot)&&(wts+sigDist[i].second>=fractionOfMCEventsThatSpillOffPlot))
	    if(xrange[0]>sigDist[i].first)
	      xrange[0]=sigDist[i].first;
	  if((wts<(1-fractionOfMCEventsThatSpillOffPlot))&&(wts+sigDist[i].second>=(1-fractionOfMCEventsThatSpillOffPlot)))
	    if(xrange[1]<sigDist[i].first)
	      xrange[1]=sigDist[i].first;
	  wts += sigDist[i].second;
	}
      for(size_t i=0; i<bkgDist.size(); i++)
	{
	  if((wtb<fractionOfMCEventsThatSpillOffPlot)&&(wtb+bkgDist[i].second>=fractionOfMCEventsThatSpillOffPlot))
	    if(xrange[0]>bkgDist[i].first)
	      xrange[0]=bkgDist[i].first;
	  if((wtb<(1-fractionOfMCEventsThatSpillOffPlot))&&(wtb+bkgDist[i].second>=(1-fractionOfMCEventsThatSpillOffPlot)))
	    if(xrange[1]<bkgDist[i].first)
	      xrange[1]=bkgDist[i].first;
	  wtb += bkgDist[i].second;
	}
      //assert(Math::MPEquality(wts,1.));
      //assert(Math::MPEquality(wtb,1.));
      for(size_t i=0; i<dataDist.size(); i++)
	{
	  if(fractionOfDataEventsThatSpillOffPlot==0)
	    {
	      if(dataEvents[i][k]<xrange[0])
		xrange[0]=dataEvents[i][k];
	      if(dataEvents[i][k]>xrange[1])
		xrange[1]=dataEvents[i][k];
	    }
	  else
	    {
	      if((wtd<fractionOfDataEventsThatSpillOffPlot)&&(wtd+dataDist[i].second>=fractionOfDataEventsThatSpillOffPlot))
		if(xrange[0]>dataDist[i].first)
		  xrange[0]=dataDist[i].first;
	      if((wtd<1-fractionOfDataEventsThatSpillOffPlot)&&(wtd+dataDist[i].second>=1-fractionOfDataEventsThatSpillOffPlot))
		if(xrange[1]<dataDist[i].first)
		  xrange[1]=dataDist[i].first;
	      wtd += dataDist[i].second;
	    }
	}
      xrange[1] = xrange[1] + 0.05*(xrange[1]-xrange[0]); 
      xrange[0] = xrange[0] - 0.05*(xrange[1]-xrange[0]); 
      if(xrange[0]==xrange[1])
	{
	  xrange[0] -= 1.;
	  xrange[1] += 1.;
	}
      range.push_back(xrange);
      //int nx = (int)(pow((double)dataEvents.size()+25,0.5));
      int nx=(int)(10+30*(1-exp(-(double)dataEvents.size()/100)));
      nbins.push_back(nx);
    }
  return;
}



vector<vector<int> > Math::permutationSet(vector<int> q) // returns a vector<vector<int> > of size q.size() factorial by q.size()
{
  if(q.size()==0)
    return(vector<vector<int> >(1,vector<int>(0)));
  vector<vector<int> > ans;
  for(size_t i=0; i<q.size(); i++)
    {
      vector<int> p;
      for(size_t j=0; j<q.size(); j++)
	if(j!=i)
	  p.push_back(q[j]);
      vector<vector<int> > r = permutationSet(p);
      for(size_t j=0; j<r.size(); j++)
	r[j].push_back(q[i]);
      ans.insert(ans.end(), r.begin(), r.end());
    }
  return(ans);
}

vector<vector<int> > Math::permutationSet(int n)
{
  vector<int> q(n);
  for(int i=0; i<n; i++)
    {
      q[i] = i;
    }
  return(permutationSet(q));
}
  

vector<string> Math::getFilesInDirectory(string dir, string pattern)
{
  string tmpFileName = getTmpFilename();
  bool containsWildCard = false;
  if(dir=="")
    dir=".";
  vector<string> ans;
  for(size_t i=0; i<pattern.length(); i++)
    if((pattern.substr(i,1)=="*")||
       (pattern.substr(i,1)=="["))
      containsWildCard = true;
  if(containsWildCard)
    {
      ::system(("cd "+dir+"; echo "+pattern+" > "+tmpFileName+";").c_str());
      ifstream ftmp(tmpFileName.c_str());
      string file;
      while(ftmp >> file)
	if(file!=pattern)
	  ans.push_back(file);
      ftmp.close();
      ::system(("rm "+tmpFileName).c_str());
    }
  else
    {
      if(access((dir+"/"+pattern).c_str(),F_OK)==0)
	ans.push_back(pattern);
    }
  return(ans);
}
    
vector<double> Math::putIntoBins(vector<double> binEdges, vector<double> points, vector<double> wt)
{
  sort(binEdges.begin(), binEdges.end());
  vector<double> ans(binEdges.size()+1);
  size_t n = wt.size();
  assert(n==points.size());
  for(size_t i=0; i<n; i++)
    ans[upper_bound(binEdges.begin(), binEdges.end(), points[i])-binEdges.begin()] += wt[i];
  return(ans);
}

vector<int> Math::putIntoBins(vector<double> binEdges, vector<double> points)
{
  sort(binEdges.begin(), binEdges.end());
  vector<int> ans(binEdges.size()+1);
  int n = points.size();
  for(int i=0; i<n; i++)
    ans[upper_bound(binEdges.begin(), binEdges.end(), points[i])-binEdges.begin()]++;
  return(ans);
}

 
int Math::intpow(int i, int j)
{
  int ans = 1;
  for(int l=0; l<j; l++)
    ans *= i;
  return(ans);
}

string Math::getTmpFilename()
{
  // use both the process ID and a random number to ensure uniqueness of the file name
  string ans = "/tmp/tmp_XYZ_"+Math::ftoa(getpid())+"_"+Math::ftoa((int)(drand48()*1e6))+".txt";
  return(ans);
}


void Math::makeDifferent(vector<double>& x, double tol)
{
  vector<double> y=x;
  sort(y.begin(), y.end());
  assert(y==x); // ensure that x is sorted

  for(size_t i=0; i<x.size()-1; i++)
    {
      if(x[i+1]-x[i] < tol)
	{
	  size_t j=i+1;
	  while((j<x.size()-1)&&(x[j+1]-x[j]<tol))
	    j++;
	  for(size_t k=i+1; k<=j; k++)
	    x[k] += tol;
	}	      
    }
  return;
}

string Math::system(string command)
{
  string tmpFileName = Math::getTmpFilename();
  ::system((command+" > "+tmpFileName).c_str());
  ifstream ftmp(tmpFileName.c_str());
  string ans="", s="";
  while(getline(ftmp,s))
    ans += s+"\n";
  ftmp.close();
  ::system(("rm "+tmpFileName).c_str());
  return(ans);
}


double Math::innerProduct(vector<double> a, vector<double> b)
{
  assert(a.size()==b.size());
  assert(a.size()>=1);
  double ans=0;
  for (size_t i=0; i<a.size(); i++)
    ans += a[i]*b[i];
  return ans;
}

vector<double> Math::outerProduct(vector<double> a, vector<double> b)
{
  assert(a.size() == b.size());
  assert(a.size() == 3);
  vector<double> ans;
  double ans0=a[1]*b[2]-a[2]*b[1];
  double ans1=a[2]*b[0]-a[0]*b[2];
  double ans2=a[0]*b[1]-a[1]*b[0];
  ans.push_back(ans0);
  ans.push_back(ans1);
  ans.push_back(ans2);
  return ans;
}

double Math::norm(vector<double> a) {
  double ans=sqrt(Math::innerProduct(a,a));
  return(ans);
}


vector<double> Math::normalizeVector(vector<double> a) {
  // take any vector and divide it by its norm, to return a unit vector.
  double norm=sqrt(Math::innerProduct(a,a));
  if (norm == 0)
    return (a);

  assert(norm > 0);
  vector<double> ans;
  size_t imax=a.size();
  double normi=1./norm;
  for (size_t i=0; i<imax ; ++i )
    ans.push_back(a[i]*normi);
  return(ans);
}

double Math::planesAngle(vector<double> a1, vector<double> a2, vector<double> b1, vector<double> b2) {
  // a1 and a2 define plane a, while b1 and b2 define plane b. Each plane is characterized uniquely by its normal unit vector na and nb. This function will return the accute angle between the directions of na and nb. The fact that we don't return just the angle which can be between 0 and pi, is that we don't consider the planes to have "up" or "down" side. In other words, we want the function to return the same if I call it with (a1,a2,b1,b2) or (a2,a1,b1,b2) or (a2,a1,b2,b1) or (a1,a2,b2,b1)
  vector<double> na = Math::normalizeVector(Math::outerProduct(a1,a2));
  vector<double> nb = Math::normalizeVector(Math::outerProduct(b1,b2));
  double dotProduct = fabs(Math::innerProduct(na,nb));  //fabs to keep the accute angle
  if(dotProduct>1) dotProduct=1;
  double ans=acos(dotProduct); 
  return(ans);
}


int Math::criticalD(double b, double pvalueTarget) {
  // Georgios: If we expect b, what is the minimum number of data we need to observe to get a pvalue <= pvalueTarget ?
  //cout << "\ncriticalD: b=" << b << " pvalueTarget=" << pvalueTarget << endl;
  int d = (int) max(0, (int)floor(b + Math::prob2sigma(pvalueTarget)*sqrt(b)) ) ;
  //cout << "Starting d= " << d << endl;
  if ( Math::probOfThisEffect(b,d,0,">=") < pvalueTarget ) {
    double pval=Math::probOfThisEffect(b,d,0,">=");
    while ( pval < pvalueTarget ) {
      //cout << "d= " << d << " pval=" << pval;
      --d;
      double previousPval=pval;
      pval=Math::probOfThisEffect(b,d,0,">=");
      if ( previousPval==pval ) break; //we don't have precision to get any nearer pvalueTarget
    }
  }
  else {
    double pval=Math::probOfThisEffect(b,d,0,">=");
    while ( pval > pvalueTarget ) {
      //cout << "d= " << d << " pval=" << pval;
      ++d;
      double previousPval=pval;
      pval=Math::probOfThisEffect(b,d,0,">=");
      if ( previousPval==pval ) break; //we don't have precision to get any nearer pvalueTarget
    }
  }
  //cout << " Returning: " << d << endl;
  return(d);
}



/// The rest of this file is no longer used, but retained just in case


#if 0
double Math::poissonConvolutedWithPoissonIntegrated(double b, double wtMC, int n)
{
  if(n<0)
    cout << "n = " << n << endl;
  assert(n>=0);
  assert(b>=0);
  assert(wtMC>0);

  double result, error;
  size_t neval;
  double params[3] = {b, wtMC, n};
  double nMC = b/wtMC;
  double deltab = wtMC*sqrt(nMC);
  double low, high;
  double nSigma = 5.0;
  if(nMC>25) // large MC statistics
    {
      low = b - nSigma*deltab;
      high = b + nSigma*deltab;
    }
  else // small MC statistics
    {
      low = 0;
      high = b + nSigma*deltab + nSigma*5*wtMC;
    }

  gsl_function F;
  F.function = &poissonConvolutedWithPoisson;
  F.params = &params;

  //gsl_set_error_handler_off();
  //gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
  //gsl_integration_qng(&F, low, high, 1e-5, 0, &result, &error, &neval); // bktemp 1e-8
    gsl_integration_qng(&F, low, high, 1e-5, 0, &result, &error, &neval); // bktemp 1e-8
  //gsl_integration_qag(&F, low, high, 1e-10, 0, 100, 6, w, &result, &error);
  //gsl_integration_workspace_free(w);
  return(result);
}

double Math::poissonConvolutedWithGaussianIntegrated(double b, double deltab, int n)
{
  assert(n>=0);
  assert(b>=0);
  assert(deltab>=0);
  if(deltab==0)
    return( exp(-b + n * log(b) - lgamma(n+1)) );

  double result, error;
  size_t neval;
  double params[3] = {b, deltab, n};

  gsl_function F;
  F.function = &poissonConvolutedWithGaussian;
  F.params = &params;

  //gsl_set_error_handler_off();
  //gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
  gsl_integration_qng(&F, b-5*deltab, b+5*deltab, 1e-5, 0, &result, &error, &neval); // bktemp 1e-8
  //gsl_integration_qag(&F, b-5*deltab, b+5*deltab, 1e-10, 0, 100, 6, w, &result, &error);
  //gsl_integration_workspace_free(w);
  return(result);
}


// Define the KeepNSmallest class methods

Math::KeepNSmallest::KeepNSmallest(int N)
{
  this->N = N;
  values = vector<double>(N);
  numbers = vector<int>(N);
  for(int i=0; i<N; i++)
    values[i]=1.e8;
  return;
}

void Math::KeepNSmallest::update(double value, int number)
{
  for(int k=0; k<N; k++)
    if(value < values[k])
      {
	for(int l=N-2; l>=k; l--)
	  {
	    values[l+1] = values[l];
	    numbers[l+1] = numbers[l];
	  }
	values[k] = value;
	numbers[k] = number;
	break;
      }
  return;
}	

vector<int> Math::KeepNSmallest::getNumbers()
{
  return(numbers);
}

vector<double> Math::KeepNSmallest::getValues()
{
  return(values);
}

Math::KeepWSmallest::KeepWSmallest(double W, int N)
{
  this->W = W;
  this->N = N;
}



Math::KeepWSmallest::KeepWSmallest(double W, int N)
{
  this->W = W;
  this->N = N;
  values = vector<double>(0);
  numbers = vector<int>(0);
  weights = vector<double>(0);
  totalWeight = 0.;
  return;
}

void Math::KeepWSmallest::update(double value, int number, double weight)
{
  double accumulatedWeight = 0.;
  bool wasInserted = false;
  if((!values.empty())&&(value<values[values.size()-1]))
    for(int k=0; k<values.size(); k++)
      {
	if((value < values[k])&&((accumulatedWeight+weight<W)||(k<N)))
	  {
	    values.push_back(0);
	    numbers.push_back(0);
	    weights.push_back(0);
	    for(int l=values.size()-2; l>=k; l--)
	      {
		values[l+1] = values[l];
		numbers[l+1] = numbers[l];
		weights[l+1] = weights[l];
	      }
	    values[k] = value;
	    numbers[k] = number;
	    weights[k] = weight;
	    wasInserted = true;
	    totalWeight += weight;
	    break;
	  }
	else
	  accumulatedWeight += weights[k];
      }
  if((!wasInserted)&&((totalWeight+weight<W)||(values.size()<N)))
    {
      values.push_back(value);
      numbers.push_back(number);
      weights.push_back(weight);
      totalWeight += weight;
    }
  if(wasInserted)
    {
      accumulatedWeight = 0.;
      for(int i=0; i<values.size(); i++)
	{
	  accumulatedWeight += weights[i];
	  if((accumulatedWeight>W)&&(i>=N))
	    {
	      for(int j=values.size()-1; j>=i; j--)
		{
		  values.pop_back();
		  numbers.pop_back();
		  totalWeight -= weights[j];
		  weights.pop_back();
		}
	      break;
	    }
	}
    }
  return;
}	

vector<int> Math::KeepWSmallest::getNumbers()
{
  return(numbers);
}

vector<double> Math::KeepWSmallest::getValues()
{
  return(values);
}


// Numerical Recipes in C routines
extern void lubksb(float **a, int n, int *indx, float b[]);
extern void ludcmp(float **a, int n, int *indx, float *d);

vector<double> Math::linearEquationSolve(matrix A, vector<double> B)
{
  // This is straight from Numerical Recipes in C, section 2.3, on LU decomposition
  assert(A.nrows() == A.ncols());
  float ** a, *b, d;
  int n, *indx;
  n = A.nrows();
  a = new float * [n+1];
  b = new float [n+1];
  indx = new int [n+1];
  a[0] = new float [n+1];
  for(int i=0; i<n; i++)
    {
      a[i+1] = new float [n+1];
      for(int j=0; j<n; j++)
	a[i+1][j+1] = A[i][j];
      b[i+1] = B[i];
    }
  ludcmp(a, n, indx, &d);
  lubksb(a, n, indx, b);
  vector<double> ans(n);
  for(int i=0; i<n; i++)
    ans[i] = b[i+1];
  for(int i=0; i<n; i++)
    delete a[i+1];
  delete a;
  delete b;
  delete indx;
  return(ans);
}


#endif
 
