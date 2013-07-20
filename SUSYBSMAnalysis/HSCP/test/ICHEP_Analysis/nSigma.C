//--------------------------------------------------------------------------------------------
//
// nSigma.cc
// v1.1, updated by Greg Landsberg 5/21/09
//
// This Root code computes the probability for the expectd background Bkgr with the FRACTIONAL
// uncertainty sFfrac (i.e., B = Bkgr*(1 +/- sBfrac)) to fluctuate to or above the
// observed number of events nobs
//
// To find 3/5 sigma evidence/discovery points, one should use nobs = int(<S+B>),
// where <S+B> is the expected mean of the signal + background.
//
// Usage: nSigma(Double_t Bkgr, Int_t nobs, Double_t sBfrac) returns the one sided probability
// of an upward backround fluctuations, expressed in Gaussian sigmas. It is suggested to run
// this code in the compiled mode, i.e. .L nSigma.cc++
//
// 5 sigma corresponds to the p-value of 2.85E-7; 3 sigma corresponds to p-value of 1.35E-3
//
//---------------------------------------------------------------------------------------------
#include "TMath.h"
#include "TF1.h"

Double_t nSigma(Double_t Bkgr, Int_t nobs, Double_t sBfrac);
Double_t Poisson(Double_t Mu, Int_t n);
Double_t PoissonAve(Double_t Mu, Int_t n, Double_t ErrMu);
Double_t Inner(Double_t *x, Double_t *par);
Double_t ErfcInverse(Double_t x);

static const Double_t Eps = 1.e-9;

Double_t nSigma(Double_t Bkgr, Int_t nobs, Double_t sBfrac) {
	//caluculate poisson probability 
	Double_t probLess = 0.;
	Int_t i = nobs;
	Double_t eps = 0;
	do {
		eps = 2.*PoissonAve(Bkgr, i++, sBfrac*Bkgr);
		probLess += eps;
	} while (eps > 0.);
//	
	return TMath::Sqrt(2.)*ErfcInverse(probLess);	
}

Double_t Poisson(Double_t Mu, Int_t n)
{
	Double_t logP;
//
	logP = -Mu + n*TMath::Log(Mu);
	for (Int_t i = 2; i <= n; i++) logP -= TMath::Log((Double_t) i);
//
	return TMath::Exp(logP);
}

Double_t PoissonAve(Double_t Mu, Int_t n, Double_t ErrMu)
{
	Double_t par[3], retval;
	par[0]=Mu;  // background value
	par[1]=ErrMu;  // background error
	par[2]=n; // n
	TF1 *in = new TF1("Inner",Inner,0.,Mu + 5.*ErrMu,3);   
	Double_t low = Mu > 5.*ErrMu ? Mu - 5.*ErrMu : 0.;
	if (ErrMu < Eps) {
		Double_t x[1];
		x[0] = Mu;
		par[1] = 1./sqrt(2.*TMath::Pi());
		retval = Inner(x,par);
	} else retval = in->Integral(low,Mu+5.*ErrMu,par);
	delete in;
	return retval;
}

Double_t Inner(Double_t *x, Double_t *par)
{
    Double_t B, sB;
    B = par[0];
    sB = par[1];
    Int_t n = par[2];
//
    return 1./sqrt(2.*TMath::Pi())/sB*exp(-(x[0]-B)*(x[0]-B)/2./sB/sB)*Poisson(x[0],n);
}

Double_t ErfcInverse(Double_t x)
{
	Double_t xmin = 0., xmax = 20.;
	Double_t sig = xmin;
	if (x >=1) return sig;
//
	do {
		Double_t erf = TMath::Erfc(sig);
		if (erf > x) {
			xmin = sig;
			sig = (sig+xmax)/2.;
		} else {
			xmax = sig;
			sig = (xmin + sig)/2.;
		}
	} while (xmax - xmin > Eps);
	return sig;
}