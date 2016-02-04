#ifndef __ROO_MYPDF_RDL__
#define __ROO_MYPDF_RDL__

// Developed by Wouter Hulsbergen

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;
class RooAbsReal;

class CruijffPdf : public RooAbsPdf {
public:
  CruijffPdf(const char *name, const char *title, RooAbsReal& _m,
		RooAbsReal& _m0, 
		RooAbsReal& _sigmaL, RooAbsReal& _sigmaR,
		RooAbsReal& _alphaL, RooAbsReal& _alphaR) ;
  
  CruijffPdf(const CruijffPdf& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { 
    return new CruijffPdf(*this,newname); }

  inline virtual ~CruijffPdf() { }

protected:

  RooRealProxy m;
  RooRealProxy m0;
  RooRealProxy sigmaL;
  RooRealProxy sigmaR;
  RooRealProxy alphaL;
  RooRealProxy alphaR;

  Double_t evaluate() const;

private:
  
  ClassDef(CruijffPdf,0)
};

#endif
