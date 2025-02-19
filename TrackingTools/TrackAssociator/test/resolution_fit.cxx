#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooDataHist.h"
#include "RooAddPdf.h"
#include "RooGlobalFunc.h"
#include "CruijffPdf.h"
#include "TH1F.h"
// best for fits with a core gaussian and some outliers
RooPlot* double_gauss_fit(TH1* histo, TString title = "",
			  double min_mean1=-100, double max_mean1=100,
			  double min_mean2=-100, double max_mean2=100,
			  double min_sigma1=0.001, double max_sigma1=10,
			  double min_sigma2=1, double max_sigma2=10)
{
   using namespace RooFit;
   RooRealVar x("x","x",0);
   RooRealVar mean1("mean1","mean1",min_mean1,max_mean1);
   RooRealVar mean2("mean2","mean2",min_mean2,max_mean2);
   RooRealVar sigma1("sigma1","sigma1",min_sigma1,max_sigma1);
   RooRealVar sigma2("sigma2","sigma2",min_sigma2,max_sigma2);
   RooGaussian pdf1("gaus1","gaus1",x,mean1,sigma1);
   RooGaussian pdf2("gaus2","gaus2",x,mean2,sigma2);
   RooRealVar frac("frac","frac",0,1);
   RooAddPdf pdf("pdf","pdf",pdf1,pdf2,frac);
   RooDataHist data("data","data",x,histo);
   pdf.fitTo(data,RooFit::Minos(kFALSE));
   RooPlot* frame = x.frame();
   data.plotOn(frame);
   data.statOn(frame,What("N"));
   pdf.paramOn(frame,Format("NEA",AutoPrecision(2)));
   pdf.plotOn(frame);
   frame->SetTitle(title);
   frame->Draw();
   return frame;
}

// best for asymmetrical distributions with long tails
RooPlot* cruijff_fit(TH1* histo, TString title = "",
		     double min_mean=0, double max_mean=10,
		     double min_sigmaL=0.001, double max_sigmaL=10,
		     double min_sigmaR=0.001, double max_sigmaR=10)
{
   using namespace RooFit;
   RooRealVar x("x","x",0);
   RooRealVar mean("mean","mean",min_mean,max_mean);
   RooRealVar sigmaL("sigmaL","sigmaL",min_sigmaL,max_sigmaL);
   RooRealVar sigmaR("sigmaR","sigmaR",min_sigmaR,max_sigmaR);
   RooRealVar alphaL("alphaL","alphaL",0,1); 
   RooRealVar alphaR("alphaR","alphaR",0,1);
   CruijffPdf pdf("pdf","pdf",x,mean,sigmaL,sigmaR,alphaL,alphaR);
   RooDataHist data("data","data",x,histo);
   pdf.fitTo(data,RooFit::Minos(kFALSE));
   RooPlot* frame = x.frame();
   data.plotOn(frame);
   data.statOn(frame,What("N"));
   pdf.paramOn(frame,Format("NEA",AutoPrecision(2)));
   pdf.plotOn(frame);
   frame->SetTitle(title);
   frame->Draw();
   return frame;
}

