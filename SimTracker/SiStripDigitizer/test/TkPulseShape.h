
#include "TF1.h"

namespace TkPulseShape {

Double_t fpeak(Double_t *x, Double_t *par)
{
  if(x[0]+par[1]<0) return par[0];
  return par[0]+par[2]*(x[0]+par[1])*TMath::Exp(-(x[0]+par[1])/par[3]);
}

Double_t fdeconv(Double_t *x, Double_t *par)
{
  Double_t xm = par[4]*(x[0]-25);
  Double_t xp = par[4]*(x[0]+25);
  Double_t xz = par[4]*x[0];
  return 1.2131*TkPulseShape::fpeak(&xp,par)-1.4715*TkPulseShape::fpeak(&xz,par)+0.4463*TkPulseShape::fpeak(&xm,par);
}

Double_t fpeak_convoluted(Double_t *x, Double_t *par)
{
 TF1 f("peak_convoluted",TkPulseShape::fpeak,0,200,4);
 return f.Integral(x[0]-par[4]/2.,x[0]+par[4]/2.,par,1.)/(par[4]);
}

Double_t fdeconv_convoluted(Double_t *x, Double_t *par)
{
  Double_t xm = (x[0]-25);
  Double_t xp = (x[0]+25);
  Double_t xz = x[0];
  return 1.2131*TkPulseShape::fpeak_convoluted(&xp,par)-1.4715*TkPulseShape::fpeak_convoluted(&xz,par)+0.4463*TkPulseShape::fpeak_convoluted(&xm,par);
}

TF1* GetDeconvFitter() 
{
  TF1* deconv_fitter = new TF1("deconv_fitter",TkPulseShape::fdeconv_convoluted,-50,50,5);
  deconv_fitter->SetParLimits(0,-10,10); // baseline
  deconv_fitter->SetParLimits(1,-100,0); // -position
  deconv_fitter->SetParLimits(2,0,20);   // amplitude (20 -> ~255)
  deconv_fitter->SetParLimits(3,5,100);  // time constant
  deconv_fitter->SetParLimits(4,0,50);   // smearing
  deconv_fitter->SetParameters(0.,-50,10,50,10);
  return deconv_fitter;
}

TF1* GetPeakFitter()
{
   TF1* peak_fitter = new TF1("peak_fitter",TkPulseShape::fpeak_convoluted,1,300,5);
   peak_fitter->SetParLimits(0,-10,10);
   peak_fitter->SetParLimits(1,-100,0);
   peak_fitter->SetParLimits(2,0,200);
   peak_fitter->SetParLimits(3,5,100);
   peak_fitter->FixParameter(4,22.5);
   peak_fitter->SetParameters(0.,-35,10,50,10);
   return peak_fitter;
}
}
