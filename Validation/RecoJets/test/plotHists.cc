#include <math.h>
#include <iostream>
#include <string>
#include <vector>

#include "TROOT.h"
#include "TFile.h"
#include "TKey.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TFrame.h"
#include "TPostScript.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TText.h"

class PlotPoint {
public:
  double x;
  double y;
  double meanerror;
  double syserror;
  double width;
  double widtherror;
  bool valid;
};

class PlotPoints {
public:
  std::vector<PlotPoint> points;
  std::string name;
};

TH1D* getSlice (const TH2F& fHist, int fSlice, int fNSlices=1) {
  char title[1024];
  sprintf (title, "%s_slice_y%d", fHist.GetName(), fSlice);
  return fHist.ProjectionY (title, fSlice, fSlice+fNSlices-1, "e");
}

TH1D* getSlice2 (const TH2F& fHist, int fSlice, int fNSlices=1) {
  char title[1024];
  sprintf (title, "%s_slice_x%d", fHist.GetName(), fSlice);
  return fHist.ProjectionX (title, fSlice, fSlice+fNSlices-1, "e");
}

TH1D* getSlice (const TH3F& fHist, int fSliceXMin, int fSliceXMax, int fSliceYMin, int fSliceYMax) {
  char title[1024];
  sprintf (title, "%s_slice_xy%d_%d_%d_%d", fHist.GetName(), fSliceXMin, fSliceXMax, fSliceYMin, fSliceYMax);
  return fHist.ProjectionZ (title, fSliceXMin, fSliceXMax, fSliceYMin, fSliceYMax, "e");
}

PlotPoint limitedFit (const TH1D& fHist, int fFirstBin, int fLastBin) {
  static int counter = 0;
  TH1D hist (fHist);
  if (fFirstBin < 1) fFirstBin = 1;
  if (fLastBin > fHist.GetNbinsX()) fLastBin = fHist.GetNbinsX();
  double xmin = hist.GetBinLowEdge(fFirstBin);
  double xmax = hist.GetBinLowEdge(fLastBin+1);
  TF1 gauss ("g","gaus");
  //  hist.Fit (&gauss, "QN", "", xmin, xmax);
  hist.Fit (&gauss, "Q", "", xmin, xmax);
  // draw histogram
  if (0)
  {
    gStyle->SetOptFit (111);
    TCanvas c1 ("c1","A Simple Graph with error bars",500,500);
    hist.Draw();
    c1.Update();
    char histname [1024];
    sprintf (histname, "%s_%d.gif", fHist.GetName(), counter++);
    c1.Print (histname);
  }
  double chi2 = gauss.GetChisquare();
  double mean = gauss.GetParameter(1);
  double width = gauss.GetParameter(2);
  double meanError = gauss.GetParError(1);
  double widthError = gauss.GetParError(2);
  PlotPoint result;
  result.x = 0;
  result.y = mean;
  result.meanerror = meanError;
  result.width = width;
  result.widtherror = widthError;
  result.valid = true;
  bool printout=false;
  if (printout)
  std::cout << "limitedFit-> " << fFirstBin << '/' << fLastBin
	    << ", chi2: " << chi2 
	    << ", result: " << result.y << '/' << result.meanerror << '/' 
	    << result.width << '/' << result.widtherror << std::endl;
  return result;
}

PlotPoint fitHist (const TH1D& fHist) {
  PlotPoint result;
  // look for maximum
  int maxBin = 0;
  double maxValue = fHist.GetBinContent (0);
  for (int ibin = 0; ibin < fHist.GetNbinsX(); ++ibin) {
    if (fHist.GetBinContent (ibin) > maxValue) {
      maxBin = ibin;
      maxValue = fHist.GetBinContent (maxBin);
    }
  }
  if (maxValue < 10) { // low statistics
    result.valid = false;
    return result;
  }
  double binSize = fHist.GetBinWidth(0);
  double histSigma = fHist.GetRMS();
  int fitWindow = (int) floor (histSigma/binSize);
  if (fitWindow < 2) {
    fitWindow = 2;
    std::cout << "fitHist-> window is too narrow (1): " << histSigma << '/' << fitWindow << std::endl;
  }
  // collect fit statistics
  PlotPoint fit0 = limitedFit (fHist, maxBin-fitWindow, maxBin+fitWindow); // central
  //refit
  fitWindow = (int) floor (2.*fit0.width/binSize);
  if (fitWindow < 2) {
    fitWindow = 2;
    std::cout << "fitHist-> window is too narrow (2): " << histSigma << '/' << fitWindow << std::endl;
  }
  fit0 = limitedFit (fHist, maxBin-fitWindow, maxBin+fitWindow); // central
  int shift = (int) floor (fit0.width/binSize/2.+0.5);
  std::cout << "fitHist-> width/binsize/window/shift: " << fit0.width << '/' 
	    << binSize << '/' << fitWindow << '/' << shift << std::endl;
  if (shift < 1) shift = 1;
  PlotPoint fitLeft = limitedFit (fHist, maxBin-fitWindow-shift, maxBin+fitWindow-shift); // left
  PlotPoint fitRight = limitedFit (fHist, maxBin-fitWindow+shift, maxBin+fitWindow+shift); // right
  PlotPoint fitNarrow = limitedFit (fHist, maxBin-fitWindow+shift, maxBin+fitWindow-shift); // narrow
  PlotPoint fitWide = limitedFit (fHist, maxBin-fitWindow-shift, maxBin+fitWindow+shift); // wide
  result.x = 0;
  result.y = (fit0.y + fitLeft.y + fitRight.y + fitNarrow.y + fitWide.y)/5.;
  result.meanerror = fit0.meanerror;
  result.width = fit0.width;
  result.widtherror = fit0.widtherror;
  result.valid = fit0.valid && fitLeft.valid && fitRight.valid && fitNarrow.valid && fitWide.valid;
  double sysMean = fabs ((fitLeft.y - fitRight.y));
  result.syserror = sysMean;
  double sysWidth = fabs ((fitLeft.width - fitRight.width));
  result.widtherror = sqrt (fit0.widtherror*fit0.widtherror + sysWidth*sysWidth);
  if ((result.syserror+result.meanerror) > 5 * fabs(result.y)) { // too big error
    result.valid = false;
  }
  return result;
}

PlotPoints profileFit (const TH2F& fHist) {
  PlotPoints result;
  int nSlices = fHist.GetNbinsX();
  for (int iSlice = 1; iSlice <= nSlices; ++iSlice) {
    TH1D* slice = getSlice (fHist, iSlice);
    std::cout << "profileFit-> processing hist " << fHist.GetName() << ", slice " << iSlice << std::endl;
    PlotPoint fit = fitHist (*slice);
    fit.x = fHist.GetBinCenter (fHist.GetBin (iSlice, 0));
    result.points.push_back (fit);
    delete slice;
  }
  result.name = "somename";
  return result;
}

double plusepsilon (double x) {
  return x+fabs(x)*1.e-5;
}

double minusepsilon (double x) {
  return x-fabs(x)*1.e-5;
}

PlotPoints profileFitX (const TH3F& fHist, double fYmin, double fYmax, bool fSymmetric = true) {
  PlotPoints result;
  TAxis* xAxis = fHist.GetXaxis();
  TAxis* yAxis = fHist.GetYaxis();
  int sliceYMin = 0;
  int sliceYMax = 0;
  int sliceYMin2 = 0;
  int sliceYMax2 = 0;
  for (int iy = 1; iy < yAxis->GetNbins(); ++iy) {
//      std::cout << "profileFitX->" << iy << '/' << yAxis->GetBinLowEdge (iy) << '/' 
//  	      << fYmin << '/' << sliceYMin << '/' << (yAxis->GetBinLowEdge (iy) <= fYmin) << std::endl;
    if (yAxis->GetBinLowEdge (iy) <= plusepsilon(fYmin)) sliceYMin = iy;
  }
  for (int iy = yAxis->GetNbins(); iy >=1; --iy) {
//      std::cout << "profileFitX_r->" << iy << '/' << yAxis->GetBinLowEdge (iy+1) << '/' 
//  	      << fYmax << '/' << sliceYMax << '/' << (yAxis->GetBinLowEdge (iy+1) >= fYmax) << std::endl;
    if (yAxis->GetBinLowEdge (iy+1) >= minusepsilon(fYmax)) sliceYMax = iy;
  }
  std::cout << "profileFitX-> ymin/ymax/binmin/binmax: " << fYmin << '/' << fYmax << '/' << sliceYMin << '/' << sliceYMax << std::endl;
  if (fSymmetric) {
    for (int iy = 1; iy < yAxis->GetNbins(); ++iy) {
      if (yAxis->GetBinLowEdge (iy) <= plusepsilon(-fYmax)) sliceYMin2 = iy;
    }
    for (int iy = yAxis->GetNbins(); iy >=1; --iy) {
      if (yAxis->GetBinLowEdge (iy+1) >= minusepsilon(-fYmin)) sliceYMax2 = iy;
    }
    std::cout << "profileFitX-> ymin/ymax/binmin/binmax 2: " << -fYmax << '/' << -fYmin << '/' << sliceYMin2 << '/' << sliceYMax2 << std::endl;
  }
  for (int iSlice = 1; iSlice <= xAxis->GetNbins(); ++iSlice) {
    TH1D* slice = getSlice (fHist, iSlice, iSlice, sliceYMin, sliceYMax);
    if (fSymmetric) {
      TH1D* slice2 = getSlice (fHist, iSlice, iSlice, sliceYMin2, sliceYMax2);
      slice->Add (slice2);
      delete slice2;
    }
    std::cout << "profileFitX-> processing hist " << fHist.GetName() << ", slice " << iSlice << std::endl;
    PlotPoint fit = fitHist (*slice);
    fit.x = xAxis->GetBinCenter (iSlice);
    result.points.push_back (fit);
    delete slice;
  }
  result.name = "somename";
  return result;
}

PlotPoints profileFitY (const TH3F& fHist, double fXmin, double fXmax, bool fSymmetric = true) {
  PlotPoints result;
  TAxis* xAxis = fHist.GetXaxis();
  TAxis* yAxis = fHist.GetYaxis();
  int sliceXMin = 0;
  int sliceXMax = 0;
  int sliceXMin2 = 0;
  int sliceXMax2 = 0;
  for (int ix = 1; ix < xAxis->GetNbins(); ++ix) {
    if (xAxis->GetBinLowEdge (ix) <= plusepsilon(fXmin)) sliceXMin = ix;
  }
  for (int ix = xAxis->GetNbins(); ix >=1; --ix) {
    if (xAxis->GetBinLowEdge (ix+1) >= minusepsilon(fXmax)) sliceXMax = ix;
  }
  std::cout << "profileFitY-> xmin/xmax/binmin/binmax: " << fXmin << '/' << fXmax << '/' << sliceXMin << '/' << sliceXMax << std::endl;
  if (fSymmetric) {
    for (int ix = 1; ix < xAxis->GetNbins(); ++ix) {
      if (xAxis->GetBinLowEdge (ix) <= plusepsilon(-fXmax)) sliceXMin2 = ix;
    }
    for (int ix = xAxis->GetNbins(); ix >=1; --ix) {
      if (xAxis->GetBinLowEdge (ix+1) >= minusepsilon(-fXmin)) sliceXMax2 = ix;
    }
    std::cout << "profileFitY-> xmin/xmax/binmin/binmax 2: " << -fXmax << '/' << -fXmin << '/' << sliceXMin2 << '/' << sliceXMax2 << std::endl;
  }
  for (int iSlice = 1; iSlice <= yAxis->GetNbins(); ++iSlice) {
    TH1D* slice = getSlice (fHist, sliceXMin, sliceXMax, iSlice, iSlice);
    if (fSymmetric) {
      TH1D* slice2 = getSlice (fHist, sliceXMin2, sliceXMax2, iSlice, iSlice);
      slice->Add (slice2);
      delete slice2;
    }
    std::cout << "profileFitY-> processing hist " << fHist.GetName() << ", slice " << iSlice << std::endl;
    PlotPoint fit = fitHist (*slice);
    fit.x = yAxis->GetBinCenter (iSlice);
    result.points.push_back (fit);
    delete slice;
  }
  result.name = "somename";
  return result;
}

PlotPoints getEfficiency (const TH1* fHistNumerator, const TH1* fHistDenumerator) {
  PlotPoints result;
  for (int ibin = 1; ibin <= fHistNumerator->GetNbinsX(); ++ibin) {
    PlotPoint point;
    double nn = fHistNumerator->GetBinContent (ibin);
    double dn = fHistDenumerator->GetBinContent (ibin);
    point.x = fHistNumerator->GetBinCenter (ibin);
    point.width = point.syserror = point.widtherror = 0.;
    point.valid = true;
    if (nn > 0 && dn > 0) {
      point.y = nn / dn;
      if (nn > dn/2) point.meanerror = sqrt (dn - nn) / dn;
      else point.meanerror = sqrt (nn) / dn;
    }
    else {
      point.valid = false;
      point.y = 0;
      point.meanerror = 0;
    }
    result.points.push_back (point);
  }
  return result;
}

PlotPoints getEfficiency (const TH2F& fHistNumerator, const TH2F& fHistDenumerator, int fSlice, int fNSlices=1) {
  PlotPoints result;
  TH1D* numerator = getSlice (fHistNumerator, fSlice, fNSlices);
  TH1D* denumerator = getSlice (fHistDenumerator, fSlice, fNSlices);
  return getEfficiency (numerator, denumerator);
}

PlotPoints getEfficiency2 (const TH2F& fHistNumerator, const TH2F& fHistDenumerator, int fSlice, int fNSlices=1) {
  PlotPoints result;
  TH1D* numerator = getSlice2 (fHistNumerator, fSlice, fNSlices);
  TH1D* denumerator = getSlice2 (fHistDenumerator, fSlice, fNSlices);
  return getEfficiency (numerator, denumerator);
}

std::vector<std::string> getAllKeys (const TDirectory* fDir, const std::string& fClassName) {
  std::cout << "getAllKeys-> " << fDir->GetName() << ", " <<  fClassName << std::endl;
  //  fDir->ls();
  std::vector<std::string> result;
  TIter next (fDir->GetListOfKeys ());
  for (TKey* key = 0; (key = (TKey *) next());) {
    std::cout << "key from list: " << key->GetName()  << '/' << key->GetClassName () << std::endl;
    if (fClassName == key->GetClassName ()) {
      result.push_back (std::string (key->GetName ()));
    } 
  }
  return result;
} 

std::vector<std::string> getAllObjects (const TDirectory* fDir, const std::string& fClassName) {
  std::cout << "getAllObjects-> " << fDir->GetName() << ", " <<  fClassName << std::endl;
  //  fDir->ls();
  std::vector<std::string> result;
  TIter next (fDir->GetList ());
  for (TObject* obj = 0; (obj = (TObject *) next());) {
    std::cout << "name from list: " << obj->GetName()  << '/' << obj->ClassName () << std::endl;
    if (fClassName == obj->ClassName ()) {
      result.push_back (std::string (obj->GetName ()));
    } 
  }
  return result;
} 

TObject* getObject (TDirectory* fDir, const std::vector <std::string>& fObjectName) {
  TObject* result = 0; // nothing so far
  TDirectory* dir = fDir;
  for (unsigned i = 0; i < fObjectName.size (); ++i) {
    dir->GetObject (fObjectName[i].c_str(), result);
    if (result) {
      if (i < fObjectName.size () - 1) {
	dir = (TDirectory*) result;
	result = 0;
      }
    }
    else {
      std::cerr << "getObject-> Can not find (sub)dir/object " << fObjectName[i] << " in directory " << dir->GetName () << std::endl;
      return 0;
    }
  }
  return result;
}

enum PlotOptions {MEAN, WIDTH, RELWIDTH, MEANLOG, WIDTHLOG, RELWIDTHLOG};

void plotPoints (PlotOptions fOption,
		 const std::vector<const PlotPoints*> fPlots, 
		 const std::string& fLabelX, const std::string& fLabelY, 
		 const std::string& fFilename) {
  int color [4] = {2,4,6,3};
  TCanvas* c1 = new TCanvas ("c1","A Simple Graph with error bars",1000,1000);
  
  c1->SetGrid();
  c1->SetFillColor(0);
  c1->GetFrame()->SetFillColor(21);
  c1->GetFrame()->SetBorderSize(12);
//   c1->SetFillColor(42);
//   c1->GetFrame()->SetFillColor(21);
//   c1->GetFrame()->SetBorderSize(12);
  
  TPad* pad = new TPad("pad", "pad", 0, 0, 1, 1, 0);
  switch (fOption) {
  case MEANLOG:
  case WIDTHLOG:
  case RELWIDTHLOG:
    pad->SetLogx ();
    break;
  default:
    pad->SetLogx (false);
  }
  pad->SetGrid();
  pad->Draw();
  pad->cd();
  
  TMultiGraph* mg = new TMultiGraph;
  TLegend* legend = new TLegend (0.2,0.8,0.4,1);


    Float_t x[10][1024];
    Float_t y[10][1024];
    Float_t ex[10][1024];
    Float_t ey[10][1024];

  for (unsigned iplot = 0; iplot < fPlots.size(); ++iplot) {
    const PlotPoints& plot = *(fPlots[iplot]);
    int nPoints = 0;
    double xmin = 0;
    double xmax = 0;
    for (unsigned i = 0; i < plot.points.size(); ++i) {
      if (plot.points[i].valid) {
	switch (fOption) {
	case MEAN:
	case WIDTH:
	case RELWIDTH:
	  x[iplot][nPoints] = plot.points[i].x;
	  break;
	case MEANLOG:
	case WIDTHLOG:
	case RELWIDTHLOG:
	  x[iplot][nPoints] = pow (10., plot.points[i].x);
	}
	ex[iplot][nPoints] = 0;
	switch (fOption) {
	case MEAN:
	case MEANLOG:
	  std::cout << "NP/mean/err: " << nPoints << '/' << plot.points[i].y << '/' << plot.points[i].meanerror << std::endl;
	  y[iplot][nPoints] = plot.points[i].y;
	  ey[iplot][nPoints] = sqrt (plot.points[i].meanerror*plot.points[i].meanerror + 
			      plot.points[i].syserror*plot.points[i].syserror);
	  break;
	case WIDTH:
	case WIDTHLOG:
	  y[iplot][nPoints] = plot.points[i].width;
	  ey[iplot][nPoints] = plot.points[i].widtherror;
	  break;
	case RELWIDTH:
	case RELWIDTHLOG:
	  if (fabs (plot.points[i].y) < 1e-5) continue;
	  y[iplot][nPoints] = plot.points[i].width/plot.points[i].y;
	  std::cout << "NP/width/y/rel: " << nPoints << '/' << plot.points[i].width << '/' << plot.points[i].y << '/' << y[iplot][nPoints] << std::endl;
	  ey[iplot][nPoints] = plot.points[i].widtherror/plot.points[i].y;
	  std::cout << "NP/widtherror/y/relerror: " << nPoints << '/' << plot.points[i].widtherror << '/' << plot.points[i].y << '/' << ey[iplot][nPoints] << std::endl;
	  break;
// 	default:
// 	  std::cout << "Unknown option... " << std::endl;
// 	  y[iplot][nPoints] = 0;
// 	  ey[iplot][nPoints] = 0;
	}
	if (nPoints > 0) {
	  if (x[iplot][nPoints] < xmin) xmin = x[iplot][nPoints];
	  if (x[iplot][nPoints] > xmax) xmax = x[iplot][nPoints];
	}
	else {
	  xmin = x[iplot][nPoints];
	  xmax = x[iplot][nPoints];
	}
	nPoints++;
      }
    } 
    TGraphErrors* gr = new TGraphErrors (nPoints,x[iplot],y[iplot],ex[iplot],ey[iplot]);
    std::cout << "made graph " << plot.name << " npoints:" << nPoints << std::endl;
    gr->Print();
    gr->SetTitle("");
    gr->SetMarkerColor(color[iplot]);
    gr->SetMarkerStyle(21);
    gr->SetLineColor(color[iplot]);
    gr->SetLineWidth(2);
    mg->Add (gr);
    legend->AddEntry (gr, plot.name.c_str());
  }
  mg->Draw("ALP");
  mg->GetXaxis()->SetTitle (fLabelX.c_str());
  mg->GetXaxis()->SetMoreLogLabels ();
  mg->GetYaxis()->SetTitle (fLabelY.c_str());

  legend->Draw();
  
  c1->Update();
  
  std::string filename = fFilename + ".gif";
  c1->Print (filename.c_str());
}

void plotEscale (TDirectory* fDir) {
  std::vector <const PlotPoints*> graphs;
  TH3F* hist;
  fDir->GetObject ("EScale", hist);
  PlotPoints EScale_B = profileFitX (*hist,0, 1.4, true);
  EScale_B.name = "0<|eta|<1.4";
  graphs.push_back(&EScale_B);
  PlotPoints EScale_E = profileFitX (*hist, 1.4, 3, true);
  EScale_E.name = "1.4<|eta|<3";
  graphs.push_back(&EScale_E);
  PlotPoints EScale_F = profileFitX (*hist, 3, 5, true);
  EScale_F.name = "3<|eta|<5";
  graphs.push_back(&EScale_F);
  plotPoints (MEANLOG, graphs, "pT_GenJet (GeV/c)", "(ET_CaloJet)/(ET_GenJet)", "EScale");
}

void plotEscaleVsEta (TDirectory* fDir) {
  std::vector <const PlotPoints*> graphs;
  TH3F* hist;
  fDir->GetObject ("EScale", hist);
  PlotPoints pp_1 = profileFitY (*hist, 0.75, 1.25, false);
  pp_1.name = "5.6<pT(gen)<18";
  graphs.push_back(&pp_1);
  PlotPoints pp_2 = profileFitY (*hist, 1.25, 1.75, false);
  pp_2.name = "18<pT(gen)<56";
  graphs.push_back(&pp_2);
  PlotPoints pp_3 = profileFitY (*hist, 1.75, 2.25, false);
  pp_3.name = "56<pT(gen)<178";
  graphs.push_back(&pp_3);
  plotPoints (MEAN, graphs, "eta", "(ET_CaloJet)/(ET_GenJet)", "EScale_Eta");
}

void plotEResolution (TDirectory* fDir) {
  std::vector <const PlotPoints*> graphs;
  TH3F* hist;
  fDir->GetObject ("EScale", hist);
  PlotPoints EScale_B = profileFitX (*hist,0, 1.4, true);
  EScale_B.name = "0<|eta|<1.4";
  graphs.push_back(&EScale_B);
  PlotPoints EScale_E = profileFitX (*hist, 1.4, 3, true);
  EScale_E.name = "1.4<|eta|<3";
  graphs.push_back(&EScale_E);
  PlotPoints EScale_F = profileFitX (*hist, 3, 5, true);
  EScale_F.name = "3<|eta|<5";
  graphs.push_back(&EScale_F);
  plotPoints (RELWIDTHLOG, graphs, "pT_GenJet (GeV/c)", "sigma(ET_Calo/ET_Gen)/(ET_Calo/ET_Gen)", "EResolution");
}

void plotPhiResolution (TDirectory* fDir) {
  std::vector <const PlotPoints*> graphs;
  TH3F* hist;
  fDir->GetObject ("DeltaPhi", hist);
  PlotPoints DeltaPhi_B = profileFitX (*hist, 0, 1.4, true);
  DeltaPhi_B.name = "0<|eta|<1.4";
  graphs.push_back(&DeltaPhi_B);
  PlotPoints DeltaPhi_E = profileFitX (*hist, 1.4, 3, true);
  DeltaPhi_E.name = "1.4<|eta|<3";
  graphs.push_back(&DeltaPhi_E);
  PlotPoints DeltaPhi_F = profileFitX (*hist, 3, 5, true);
  DeltaPhi_F.name = "3<|eta|<5";
  graphs.push_back(&DeltaPhi_F);
  plotPoints (WIDTHLOG, graphs, "pT_GenJet (GeV/c)", "sigma(phi)", "DeltaPhi");
}

void plotEtaResolution (TDirectory* fDir) {
  std::vector <const PlotPoints*> graphs;
  TH3F* hist;
  fDir->GetObject ("DeltaEta", hist);
  PlotPoints DeltaEta_B = profileFitX (*hist, 0, 1.4, true);
  DeltaEta_B.name = "0<|eta|<1.4";
  graphs.push_back(&DeltaEta_B);
  PlotPoints DeltaEta_E = profileFitX (*hist, 1.4, 3, true);
  DeltaEta_E.name = "1.4<|eta|<3";
  graphs.push_back(&DeltaEta_E);
  PlotPoints DeltaEta_F = profileFitX (*hist, 3, 5, true);
  DeltaEta_F.name = "3<|eta|<5";
  graphs.push_back(&DeltaEta_F);
  plotPoints (WIDTHLOG, graphs, "pT_GenJet (GeV/c)", "sigma(eta)", "DeltaEta");
}

void plotEtaMean (TDirectory* fDir) {
  std::vector <const PlotPoints*> graphs;
  TH3F* hist;
  fDir->GetObject ("DeltaEta", hist);
  PlotPoints DeltaEta_1 = profileFitY (*hist, 0.75, 1.25, false);
  DeltaEta_1.name = "5.6<pT(gen)<18";
  graphs.push_back(&DeltaEta_1);
  PlotPoints DeltaEta_2 = profileFitY (*hist, 1.25, 1.75, false);
  DeltaEta_2.name = "18<pT(gen)<56";
  graphs.push_back(&DeltaEta_2);
  PlotPoints DeltaEta_3 = profileFitY (*hist, 1.75, 2.25, false);
  DeltaEta_3.name = "56<pT(gen)<178";
  graphs.push_back(&DeltaEta_3);
  plotPoints (MEAN, graphs, "eta", "delta(eta)", "DeltaEtaMean");
}

void plotPhiMean (TDirectory* fDir) {
  std::vector <const PlotPoints*> graphs;
  TH3F* hist;
  fDir->GetObject ("DeltaPhi", hist);
  PlotPoints DeltaEta_1 = profileFitY (*hist, 0.75, 1.25, false);
  DeltaEta_1.name = "5.6<pT(gen)<18";
  graphs.push_back(&DeltaEta_1);
  PlotPoints DeltaEta_2 = profileFitY (*hist, 1.25, 1.75, false);
  DeltaEta_2.name = "18<pT(gen)<56";
  graphs.push_back(&DeltaEta_2);
  PlotPoints DeltaEta_3 = profileFitY (*hist, 1.75, 2.25, false);
  DeltaEta_3.name = "56<pT(gen)<178";
  graphs.push_back(&DeltaEta_3);
  plotPoints (MEAN, graphs, "eta", "delta(phi)", "DeltaPhiMean");
}

void plotEtaEff (TDirectory* fDir) {
  std::vector <const PlotPoints*> graphs;
  TH2F* hist1;
  fDir->GetObject ("MatchedGenJetEta", hist1);
  TH2F* hist2;
  fDir->GetObject ("GenJetEta", hist2);
  PlotPoints p1 = getEfficiency (*hist1, *hist2, 2, 2);
  p1.name = "5.6<pT(gen)<18";
  graphs.push_back(&p1);
  PlotPoints p2 = getEfficiency (*hist1, *hist2, 4, 2);
  p2.name = "18<pT(gen)<56";
  graphs.push_back(&p2);
  PlotPoints p3 = getEfficiency (*hist1, *hist2, 6, 2);
  p3.name = "56<pT(gen)<178";
  graphs.push_back(&p3);
  plotPoints (MEAN, graphs, "eta", "efficiency", "EtaEff");
}

void plotPtEff (TDirectory* fDir) {
  std::vector <const PlotPoints*> graphs;
  TH2F* hist1;
  fDir->GetObject ("MatchedGenJetEta", hist1);
  TH2F* hist2;
  fDir->GetObject ("GenJetEta", hist2);
  PlotPoints p1 = getEfficiency2 (*hist1, *hist2, 19, 14);
  p1.name = "0<|eta|<1.4";
  graphs.push_back(&p1);
  PlotPoints p2 = getEfficiency2 (*hist1, *hist2, 11, 8);
  p2.name = "1.4<|eta|<3";
  graphs.push_back(&p2);
  PlotPoints p3 = getEfficiency2 (*hist1, *hist2, 1, 10);
  p3.name = "3<|eta|<5";
  graphs.push_back(&p3);
  plotPoints (MEANLOG, graphs, "pT_GenJet (GeV/c)", "efficiency", "PtEff");
}





int main (int argn, char* argv []) {
  int result = 0; // OK

  std::string inputFileName (argv[1]);
  std::cout << "Processing file " << inputFileName << std::endl;
  TFile* inputFile = TFile::Open (inputFileName.c_str());
  if (inputFile) {
    std::cout << "ls for the file:" << std::endl;
    inputFile->ls ();

    std::vector<std::string> dirName1 = getAllKeys (inputFile, "TDirectory");
    for (unsigned idir = 0; idir < dirName1.size(); idir++) {
      TDirectory* dir1 = 0;
      inputFile->GetObject (dirName1[idir].c_str(), dir1);
      if (dir1) {
	std::vector<std::string> dirName2 = getAllKeys (dir1, "TDirectory");
	for (unsigned idir2 = 0; idir2 < dirName1.size(); ++idir2) {
	  TDirectory* dir2 = 0;
	  dir1->GetObject (dirName2[idir2].c_str(), dir2);
	  if (dir2) {
  	    plotEResolution (dir2);
     	    plotEtaResolution (dir2);
     	    plotPhiResolution (dir2);
   	    plotEtaEff (dir2);
  	    plotEtaMean (dir2);
 	    plotPhiMean (dir2);
   	    plotPtEff (dir2);
     	    plotEscale (dir2);
	    plotEscaleVsEta (dir2);
	  }
	}
      }
      else {
	std::cerr << "Can not find dir1: " << dirName1[idir] << std::endl;
      }
    }
  }
  else {
    std::cerr << " Can not open input file " << inputFileName << std::endl;
    result = 1;
  }
  return result;
}
