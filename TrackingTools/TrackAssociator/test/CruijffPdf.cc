#include "CruijffPdf.h"
#include "RooRealVar.h"
#include "RooRealConstant.h"
using namespace RooFit;

ClassImp(CruijffPdf)

CruijffPdf::CruijffPdf(const char *name, const char *title,
	     RooAbsReal& _m, RooAbsReal& _m0, 
	     RooAbsReal& _sigmaL, RooAbsReal& _sigmaR,
	     RooAbsReal& _alphaL, RooAbsReal& _alphaR)
  :
  RooAbsPdf(name, title),
  m("m", "Dependent", this, _m),
  m0("m0", "M0", this, _m0),
  sigmaL("sigmaL", "SigmaL", this, _sigmaL),
  sigmaR("sigmaR", "SigmaR", this, _sigmaR),
  alphaL("alphaL", "AlphaL", this, _alphaL),
  alphaR("alphaR", "AlphaR", this, _alphaR)
{
}

CruijffPdf::CruijffPdf(const CruijffPdf& other, const char* name) :
  RooAbsPdf(other, name), m("m", this, other.m), m0("m0", this, other.m0),
  sigmaL("sigmaL", this, other.sigmaL), sigmaR("sigmaR", this, other.sigmaR), 
  alphaL("alphaL", this, other.alphaL), alphaR("alphaR", this, other.alphaR)
{
}

Double_t CruijffPdf::evaluate() const 
{
  double dx = (m-m0) ;
  double sigma = dx<0 ? sigmaL: sigmaR ;
  double alpha = dx<0 ? alphaL: alphaR ;
  double f = 2*sigma*sigma + alpha*dx*dx ;
  return exp(-dx*dx/f) ;
}
