// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

// data dirs
TString theDirName = "Images";
//

// data files
TString theDetectorFileName_old;
TString theDetectorFileName_new;
TString theDetector;
//
TFile* theDetectorFile_old;
TFile* theDetectorFile_new;
//

// histograms
TProfile* prof_x0_det_total_old;
TProfile* prof_x0_det_SUP_old; 
TProfile* prof_x0_det_SEN_old;
TProfile* prof_x0_det_CAB_old;
TProfile* prof_x0_det_COL_old;
TProfile* prof_x0_det_ELE_old;
TProfile* prof_x0_det_OTH_old;
TProfile* prof_x0_det_AIR_old;
TProfile* prof_x0_str_total_old;
TProfile* prof_x0_str_SUP_old; 
TProfile* prof_x0_str_SEN_old;
TProfile* prof_x0_str_CAB_old;
TProfile* prof_x0_str_COL_old;
TProfile* prof_x0_str_ELE_old;
TProfile* prof_x0_str_OTH_old;
TProfile* prof_x0_str_AIR_old;
//
TProfile* prof_x0_det_total_new;
TProfile* prof_x0_det_SUP_new; 
TProfile* prof_x0_det_SEN_new;
TProfile* prof_x0_det_CAB_new;
TProfile* prof_x0_det_COL_new;
TProfile* prof_x0_det_ELE_new;
TProfile* prof_x0_det_OTH_new;
TProfile* prof_x0_det_AIR_new;
TProfile* prof_x0_str_total_new;
TProfile* prof_x0_str_SUP_new; 
TProfile* prof_x0_str_SEN_new;
TProfile* prof_x0_str_CAB_new;
TProfile* prof_x0_str_COL_new;
TProfile* prof_x0_str_ELE_new;
TProfile* prof_x0_str_OTH_new;
TProfile* prof_x0_str_AIR_new;
//
unsigned int iFirst = 1;
unsigned int iLast  = 9;
//

using namespace std;

// Main
TrackerMaterialBudgetComparison(TString detector) {
  // detector
  theDetector = detector;
  if(
     theDetector!="TIB" && theDetector!="TIDF" && theDetector!="TIDB" && theDetector!="InnerServices"
     && theDetector!="TOB" && theDetector!="TEC" && theDetector!="TkStrct" 
     && theDetector!="PixBar" && theDetector!="PixFwdPlus" && theDetector!="PixFwdMinus" 
     && theDetector!="Tracker" && theDetector!="TrackerSum"
     && theDetector!="Pixel" && theDetector!="Strip"
     && theDetector!="InnerTracker"
     ){
    cerr << "MaterialBudget - ERROR detector not found " << theDetector << endl;
    break;
  }
  //
  
  // files
  theDetectorFileName_old = "matbdg_" + theDetector + "_old.root";
  theDetectorFileName_new = "matbdg_" + theDetector + "_new.root";
  if(theDetector == "TrackerSum") {
    iFirst = 1;
    iLast  = 9;
    theDetectorFileName_old = "matbdg_TIB_old.root";
    theDetectorFileName_new = "matbdg_TIB_new.root";
  }
  if(theDetector == "Pixel") {
    iFirst = 8;
    iLast  = 9;
    theDetectorFileName_old = "matbdg_PixBar_old.root";
    theDetectorFileName_new = "matbdg_PixBar_new.root";
  }
  if(theDetector == "Strip") {
    iFirst = 1;
    iLast  = 5;
    theDetectorFileName_old = "matbdg_TIB_old.root";
    theDetectorFileName_new = "matbdg_TIB_new.root";
  }
  if(theDetector == "InnerTracker") {
    iFirst = 1;
    iLast  = 3;
    theDetectorFileName_old = "matbdg_TIB_old.root";
    theDetectorFileName_new = "matbdg_TIB_new.root";
  }
  cout << "*** Open file... " << endl;
  cout << " old: " << theDetectorFileName_old << endl;
  cout << " new: " << theDetectorFileName_new << endl;
  cout << "***" << endl;
  //
  
  // open root files
  theDetectorFile_old = new TFile(theDetectorFileName_old);
  theDetectorFile_new = new TFile(theDetectorFileName_new);
  //
  
  // plots
  createPlots("x_vs_eta");
  createPlots("x_vs_phi");
  //  createPlots("x_vs_R");
  createPlots("l_vs_eta");
  createPlots("l_vs_phi");
  //  createPlots("l_vs_R");
  //  create2DPlots("x_vs_eta_vs_phi");
  //  create2DPlots("l_vs_eta_vs_phi");
  create2DPlots("x_vs_z_vs_R");
  create2DPlots("l_vs_z_vs_R");
  create2DPlots("x_vs_z_vs_Rsum");
  create2DPlots("l_vs_z_vs_Rsum");
  //
}

// Plots
void createPlots(TString plot) {
  unsigned int plotNumber = 0;
  TString abscissaName = "dummy";
  TString ordinateName = "dummy";
  if(plot.CompareTo("x_vs_eta") == 0) {
    plotNumber = 10;
    abscissaName = TString("#eta");
    ordinateName = TString("x/X_{0}");
  } else if(plot.CompareTo("x_vs_phi") == 0) {
    plotNumber = 20;
    abscissaName = TString("#varphi [rad]");
    ordinateName = TString("x/X_{0}");
  } else if(plot.CompareTo("x_vs_R") == 0) {
    plotNumber = 40;
    abscissaName = TString("R [mm]");
    ordinateName = TString("x/X_{0}");
  } else if(plot.CompareTo("l_vs_eta") == 0) {
    plotNumber = 1010;
    abscissaName = TString("#eta");
    ordinateName = TString("x/#lambda_{0}");
  } else if(plot.CompareTo("l_vs_phi") == 0) {
    plotNumber = 1020;
    abscissaName = TString("#varphi [rad]");
    ordinateName = TString("x/#lambda_{0}");
  } else if(plot.CompareTo("l_vs_R") == 0) {
    plotNumber = 1040;
    abscissaName = TString("R [mm]");
    ordinateName = TString("x/#lambda_{0}");
  } else {
    cout << " error: chosen plot name not known " << plot << endl;
    return;
  }
  
  // get TProfiles
  prof_x0_det_total_old = (TProfile*)theDetectorFile_old->Get(Form("%u", plotNumber));
  prof_x0_det_SUP_old   = (TProfile*)theDetectorFile_old->Get(Form("%u", 100 + plotNumber));
  prof_x0_det_SEN_old   = (TProfile*)theDetectorFile_old->Get(Form("%u", 200 + plotNumber));
  prof_x0_det_CAB_old   = (TProfile*)theDetectorFile_old->Get(Form("%u", 300 + plotNumber));
  prof_x0_det_COL_old   = (TProfile*)theDetectorFile_old->Get(Form("%u", 400 + plotNumber));
  prof_x0_det_ELE_old   = (TProfile*)theDetectorFile_old->Get(Form("%u", 500 + plotNumber));
  prof_x0_det_OTH_old   = (TProfile*)theDetectorFile_old->Get(Form("%u", 600 + plotNumber));
  prof_x0_det_AIR_old   = (TProfile*)theDetectorFile_old->Get(Form("%u", 700 + plotNumber));
  //
  prof_x0_det_total_new = (TProfile*)theDetectorFile_new->Get(Form("%u", plotNumber));
  prof_x0_det_SUP_new   = (TProfile*)theDetectorFile_new->Get(Form("%u", 100 + plotNumber));
  prof_x0_det_SEN_new   = (TProfile*)theDetectorFile_new->Get(Form("%u", 200 + plotNumber));
  prof_x0_det_CAB_new   = (TProfile*)theDetectorFile_new->Get(Form("%u", 300 + plotNumber));
  prof_x0_det_COL_new   = (TProfile*)theDetectorFile_new->Get(Form("%u", 400 + plotNumber));
  prof_x0_det_ELE_new   = (TProfile*)theDetectorFile_new->Get(Form("%u", 500 + plotNumber));
  prof_x0_det_OTH_new   = (TProfile*)theDetectorFile_new->Get(Form("%u", 600 + plotNumber));
  prof_x0_det_AIR_new   = (TProfile*)theDetectorFile_new->Get(Form("%u", 700 + plotNumber));
  //
  
  // histos
  TH1D* hist_x0_total_old = (TH1D*)prof_x0_det_total_old->ProjectionX();
  TH1D* hist_x0_SUP_old   = (TH1D*)prof_x0_det_SUP_old->ProjectionX();
  TH1D* hist_x0_SEN_old   = (TH1D*)prof_x0_det_SEN_old->ProjectionX();
  TH1D* hist_x0_CAB_old   = (TH1D*)prof_x0_det_CAB_old->ProjectionX();
  TH1D* hist_x0_COL_old   = (TH1D*)prof_x0_det_COL_old->ProjectionX();
  TH1D* hist_x0_ELE_old   = (TH1D*)prof_x0_det_ELE_old->ProjectionX();
  TH1D* hist_x0_OTH_old   = (TH1D*)prof_x0_det_OTH_old->ProjectionX();
  TH1D* hist_x0_AIR_old   = (TH1D*)prof_x0_det_AIR_old->ProjectionX();
  //
  TH1D* hist_x0_total_new = (TH1D*)prof_x0_det_total_new->ProjectionX();
  TH1D* hist_x0_SUP_new   = (TH1D*)prof_x0_det_SUP_new->ProjectionX();
  TH1D* hist_x0_SEN_new   = (TH1D*)prof_x0_det_SEN_new->ProjectionX();
  TH1D* hist_x0_CAB_new   = (TH1D*)prof_x0_det_CAB_new->ProjectionX();
  TH1D* hist_x0_COL_new   = (TH1D*)prof_x0_det_COL_new->ProjectionX();
  TH1D* hist_x0_ELE_new   = (TH1D*)prof_x0_det_ELE_new->ProjectionX();
  TH1D* hist_x0_OTH_new   = (TH1D*)prof_x0_det_OTH_new->ProjectionX();
  TH1D* hist_x0_AIR_new   = (TH1D*)prof_x0_det_AIR_new->ProjectionX();
  //
  
  if(theDetector=="TrackerSum" || theDetector=="Pixel" || theDetector=="Strip" || theDetector=="InnerTracker") {
    TString subDetector = "TIB";
    for(unsigned int i_detector=iFirst; i_detector<=iLast; i_detector++) {
      switch(i_detector) {
      case 1: {
	subDetector = "TIDF";
	break;
      }
      case 2: {
	subDetector = "TIDB";
	break;
      }
      case 3: {
	subDetector = "InnerServices";
	break;
      }
      case 4: {
	subDetector = "TOB";
	break;
      }
      case 5: {
	subDetector = "TEC";
	break;
      }
      case 6: {
	subDetector = "TkStrct";
	break;
      }
      case 7: {
	subDetector = "PixBar";
	break;
      }
      case 8: {
	subDetector = "PixFwdPlus";
	break;
      }
      case 9: {
	subDetector = "PixFwdMinus";
	break;
      }
      default: cout << " something wrong" << endl;
      }
      // file name
      TString subDetectorFileName_old = "matbdg_" + subDetector + "_old.root";
      TString subDetectorFileName_new = "matbdg_" + subDetector + "_new.root";
      // open file
      TFile* subDetectorFile_old = new TFile(subDetectorFileName_old);
      TFile* subDetectorFile_new = new TFile(subDetectorFileName_new);
      cout << "*** Open file... " << endl;
      cout << " old: " << subDetectorFileName_old << endl;
      cout << " new: " << subDetectorFileName_new << endl;
      cout << "***" << endl;
      // subdetector profiles
      prof_x0_det_total_old = (TProfile*)subDetectorFile_old->Get(Form("%u", plotNumber));
      prof_x0_det_SUP_old   = (TProfile*)subDetectorFile_old->Get(Form("%u", 100 + plotNumber));
      prof_x0_det_SEN_old   = (TProfile*)subDetectorFile_old->Get(Form("%u", 200 + plotNumber));
      prof_x0_det_CAB_old   = (TProfile*)subDetectorFile_old->Get(Form("%u", 300 + plotNumber));
      prof_x0_det_COL_old   = (TProfile*)subDetectorFile_old->Get(Form("%u", 400 + plotNumber));
      prof_x0_det_ELE_old   = (TProfile*)subDetectorFile_old->Get(Form("%u", 500 + plotNumber));
      prof_x0_det_OTH_old   = (TProfile*)subDetectorFile_old->Get(Form("%u", 600 + plotNumber));
      prof_x0_det_AIR_old   = (TProfile*)subDetectorFile_old->Get(Form("%u", 700 + plotNumber));
      //
      prof_x0_det_total_new = (TProfile*)subDetectorFile_new->Get(Form("%u", plotNumber));
      prof_x0_det_SUP_new   = (TProfile*)subDetectorFile_new->Get(Form("%u", 100 + plotNumber));
      prof_x0_det_SEN_new   = (TProfile*)subDetectorFile_new->Get(Form("%u", 200 + plotNumber));
      prof_x0_det_CAB_new   = (TProfile*)subDetectorFile_new->Get(Form("%u", 300 + plotNumber));
      prof_x0_det_COL_new   = (TProfile*)subDetectorFile_new->Get(Form("%u", 400 + plotNumber));
      prof_x0_det_ELE_new   = (TProfile*)subDetectorFile_new->Get(Form("%u", 500 + plotNumber));
      prof_x0_det_OTH_new   = (TProfile*)subDetectorFile_new->Get(Form("%u", 600 + plotNumber));
      prof_x0_det_AIR_new   = (TProfile*)subDetectorFile_new->Get(Form("%u", 700 + plotNumber));
      // add to summary histogram
      hist_x0_total_old->Add( (TH1D*)prof_x0_det_total_old->ProjectionX("B"), +1.000 );
      hist_x0_SUP_old->Add(   (TH1D*)prof_x0_det_SUP_old->ProjectionX("B")  , +1.000 );
      hist_x0_SEN_old->Add(   (TH1D*)prof_x0_det_SEN_old->ProjectionX("B")  , +1.000 );
      hist_x0_CAB_old->Add(   (TH1D*)prof_x0_det_CAB_old->ProjectionX("B")  , +1.000 );
      hist_x0_COL_old->Add(   (TH1D*)prof_x0_det_COL_old->ProjectionX("B")  , +1.000 );
      hist_x0_ELE_old->Add(   (TH1D*)prof_x0_det_ELE_old->ProjectionX("B")  , +1.000 );
      hist_x0_OTH_old->Add(   (TH1D*)prof_x0_det_OTH_old->ProjectionX("B")  , +1.000 );
      hist_x0_AIR_old->Add(   (TH1D*)prof_x0_det_AIR_old->ProjectionX("B")  , +1.000 );
      //
      hist_x0_total_new->Add( (TH1D*)prof_x0_det_total_new->ProjectionX("B"), +1.000 );
      hist_x0_SUP_new->Add(   (TH1D*)prof_x0_det_SUP_new->ProjectionX("B")  , +1.000 );
      hist_x0_SEN_new->Add(   (TH1D*)prof_x0_det_SEN_new->ProjectionX("B")  , +1.000 );
      hist_x0_CAB_new->Add(   (TH1D*)prof_x0_det_CAB_new->ProjectionX("B")  , +1.000 );
      hist_x0_COL_new->Add(   (TH1D*)prof_x0_det_COL_new->ProjectionX("B")  , +1.000 );
      hist_x0_ELE_new->Add(   (TH1D*)prof_x0_det_ELE_new->ProjectionX("B")  , +1.000 );
      hist_x0_OTH_new->Add(   (TH1D*)prof_x0_det_OTH_new->ProjectionX("B")  , +1.000 );
      hist_x0_AIR_new->Add(   (TH1D*)prof_x0_det_AIR_new->ProjectionX("B")  , +1.000 );
      //
    }
  }
  //
  
  // Draw

  // Comparison
  // canvas
  TCanvas can_comparison("TkMB_Comparison","TkMB_Comparison",1200,800);
  can_comparison.Range(0,0,25,25);
  can_comparison.Divide(4,2);
  can_comparison.SetFillColor(kWhite);
  can_comparison.SetGridy(1);
  can_comparison.SetLogy(0);
  // canvas
  TCanvas can_ratio("TkMB_ComparisonRatio","TkMB_ComparisonRatio",1200,800);
  can_ratio.Range(0,0,25,25);
  can_ratio.Divide(4,2);
  can_ratio.SetFillColor(kWhite);
  can_ratio.SetGridy(1);
  can_ratio.SetLogy(0);
  // settings
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);
  
  for(unsigned int i_category=1; i_category<=8; i_category++) {
    TH1D* histo_old;
    TH1D* histo_new;
    switch(i_category) {
    case 1: {
      histo_old = hist_x0_SUP_old;
      histo_new = hist_x0_SUP_new;
      break;
    }
    case 2: {
      histo_old = hist_x0_SEN_old;
      histo_new = hist_x0_SEN_new;
      break;
    }
    case 3: {
      histo_old = hist_x0_CAB_old;
      histo_new = hist_x0_CAB_new;
      break;
    }
    case 4: {
      histo_old = hist_x0_COL_old;
      histo_new = hist_x0_COL_new;
      break;
    }
    case 5: {
      histo_old = hist_x0_ELE_old;
      histo_new = hist_x0_ELE_new;
      break;
    }
    case 6: {
      histo_old = hist_x0_OTH_old;
      histo_new = hist_x0_OTH_new;
      break;
    }
    case 7: {
      histo_old = hist_x0_AIR_old;
      histo_new = hist_x0_AIR_new;
      break;
    }
    case 8: {
      histo_old = hist_x0_total_old;
      histo_new = hist_x0_total_new;
      break;
    }
    }
    
    // Ratio
    TH1D* histo_ratio = new TH1D(*histo_new);
    //
    
    // canvas
    can_comparison.cd();
    can_comparison.cd(i_category);
    //
    // Compare
    histo_new->SetMarkerColor(2);  // red
    histo_new->SetLineColor(102);  // dark red
    histo_new->SetFillColor(0);    // white
    histo_new->SetMarkerStyle(20); // cyrcles
    histo_new->SetMarkerSize(0.3); // 
    histo_old->SetLineColor(4);    // blue
    histo_old->SetFillStyle(3002); // small points
    histo_old->SetFillColor(4);    // blue
    histo_old->SetLineWidth(1.0);  // 
    //
    // Draw
    histo_old->GetXaxis()->SetTitle(abscissaName);
    histo_old->GetYaxis()->SetTitle(ordinateName);
    // edges
    histo_old->GetMaximum() > histo_new->GetMaximum() ?
      ( histo_new->SetMaximum(histo_old->GetMaximum()) ) :
      ( histo_old->SetMaximum(histo_new->GetMaximum()) );
    //
    //
    histo_old->Draw("HIST");
    histo_new->Draw("HIST P E1 SAME");
    //
    // perform chi2 test between obtained and nominal histograms
    double compatibilityFactor = histo_new->KolmogorovTest(histo_old,"");
    std::cout << " Compatibility of " << histo_new->GetName()
	      << " with nominal distribution " << histo_old->GetName() << " is " << compatibilityFactor << std::endl;
    //
    
    // Legenda
    TLegend* theLegend = new TLegend(0.80, 0.80, 0.99, 0.99);
    theLegend->AddEntry( histo_old , Form("OLD (%f)", histo_old->GetSumOfWeights() ) , "l" );
    theLegend->AddEntry( histo_new , Form("NEW (%f)", histo_new->GetSumOfWeights() ) , "p" );
    theLegend->SetHeader( Form("KF: %f",compatibilityFactor) );
    theLegend->Draw();
    //
    
    // Ratio
    // canvas
    can_ratio.cd();
    can_ratio.cd(i_category);
    //
    // Compare
    histo_ratio->Divide(histo_old);
    histo_ratio->SetMarkerColor(4);   // blue
    histo_ratio->SetLineColor(102);   // dark red
    histo_ratio->SetFillColor(0);     // white
    histo_ratio->SetMarkerStyle(20);  // cyrcles
    histo_ratio->SetMarkerSize(0.2);  // 
    histo_ratio->SetLineWidth(0.8);  // 
    //
    // Draw
    histo_ratio->GetXaxis()->SetTitle(abscissaName);
    histo_ratio->GetYaxis()->SetTitle( Form( "%s Ratio (New/Old)",  ordinateName.Data() ));
    histo_ratio->Draw("HIST P E1");
    //
    
  }
  
  // Store Comparison
  can_comparison.Update();
  can_comparison.SaveAs( Form( "%s/%s_Comparison_%s.eps",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can_comparison.SaveAs( Form( "%s/%s_Comparison_%s.gif",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  can_comparison.SaveAs( Form( "%s/%s_Comparison_%s.pdf",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //
  
  // Store Ratio
  can_ratio.Update();
  can_ratio.SaveAs( Form( "%s/%s_ComparisonRatio_%s.eps",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can_ratio.SaveAs( Form( "%s/%s_ComparisonRatio_%s.gif",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  can_ratio.SaveAs( Form( "%s/%s_ComparisonRatio_%s.pdf",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //
  
}

void create2DPlots(TString plot) {
  unsigned int plotNumber = 0;
  TString abscissaName = "dummy";
  TString ordinateName = "dummy";
  Int_t zLog = 0;
  Int_t zCol = 0; // 0 linear grayscale, 1 quadratic grayscale
  Double_t histoMin = -1.;
  Double_t histoMax = -1.;
  if(plot.CompareTo("x_vs_eta_vs_phi") == 0) {
    plotNumber = 30;
    abscissaName = TString("#eta");
    ordinateName = TString("#varphi");
    quotaName    = TString("x/X_{0}");
  } else if(plot.CompareTo("l_vs_eta_vs_phi") == 0) {
    plotNumber = 1030;
    abscissaName = TString("#eta");
    ordinateName = TString("#varphi");
    quotaName    = TString("x/#lambda_{0}");
  } else if(plot.CompareTo("x_vs_z_vs_Rsum") == 0) {
    plotNumber = 50;
    abscissaName = TString("z [mm]");
    ordinateName = TString("R [mm]");
    quotaName    = TString("#Sigmax/X_{0}");
    zLog = 1;
    zCol = 1;
    histoMin = 0.4762;
    histoMax = 2.1;
  } else if(plot.CompareTo("x_vs_z_vs_R") == 0) {
    plotNumber = 60;
    abscissaName = TString("z [mm]");
    ordinateName = TString("R [mm]");
    quotaName    = TString("1/X_{0}");
    zLog = 1;
    histoMin = 1E-4;
    histoMax = 1E+4;
  } else if(plot.CompareTo("l_vs_z_vs_Rsum") == 0) {
    plotNumber = 1050;
    abscissaName = TString("z [mm]");
    ordinateName = TString("R [mm]");
    quotaName    = TString("#Sigmax/#lambda_{0}");
    zLog = 1;
    histoMin = 0.4762;
    histoMax = 2.1;
  } else if(plot.CompareTo("l_vs_z_vs_R") == 0) {
    plotNumber = 1060;
    abscissaName = TString("z [mm]");
    ordinateName = TString("R [mm]");
    quotaName    = TString("1/#lambda_{0}");
    zLog = 1;
    histoMin = 1E-4;
    histoMax = 1E+4;
  } else {
    cout << " error: chosen plot name not known " << plot << endl;
    return;
  }
  
  // get TProfiles
  prof2d_x0_det_total_old = (TProfile2D*)theDetectorFile_old->Get(Form("%u", plotNumber));
  prof2d_x0_det_total_new = (TProfile2D*)theDetectorFile_new->Get(Form("%u", plotNumber));
  
  // histos
  TH2D* hist_x0_total_old = (TH2D*)prof2d_x0_det_total_old->ProjectionXY();
  TH2D* hist_x0_total_new = (TH2D*)prof2d_x0_det_total_new->ProjectionXY();
  //
  
  if(theDetector=="TrackerSum" || theDetector=="Pixel" || theDetector=="Strip" || theDetector=="InnerTracker") {
    TString subDetector = "TIB";
    for(unsigned int i_detector=iFirst; i_detector<=iLast; i_detector++) {
      switch(i_detector) {
      case 1: {
	subDetector = "TIDF";
	break;
      }
      case 2: {
	subDetector = "TIDB";
	break;
      }
      case 3: {
	subDetector = "InnerServices";
	break;
      }
      case 4: {
	subDetector = "TOB";
	break;
      }
      case 5: {
	subDetector = "TEC";
	break;
      }
      case 6: {
	subDetector = "TkStrct";
	break;
      }
      case 7: {
	subDetector = "PixBar";
	break;
      }
      case 8: {
	subDetector = "PixFwdPlus";
	break;
      }
      case 9: {
	subDetector = "PixFwdMinus";
	break;
      }
      default: cout << " something wrong" << endl;
      }
      // file name
      TString subDetectorFileName_old = "matbdg_" + subDetector + "_old.root";
      TString subDetectorFileName_new = "matbdg_" + subDetector + "_new.root";
      // open file
      TFile* subDetectorFile_old = new TFile(subDetectorFileName_old);
      TFile* subDetectorFile_new = new TFile(subDetectorFileName_new);
      cout << "*** Open file... " << endl;
      cout << " old: " << subDetectorFileName_old << endl;
      cout << " new: " << subDetectorFileName_new << endl;
      cout << "***" << endl;
      // subdetector profiles
      prof2d_x0_det_total_old = (TProfile2D*)subDetectorFile_old->Get(Form("%u", plotNumber));
      prof2d_x0_det_total_new = (TProfile2D*)subDetectorFile_new->Get(Form("%u", plotNumber));
      // add to summary histogram
      hist_x0_total_old->Add( (TH2D*)prof2d_x0_det_total_old->ProjectionXY("B"), +1.000 );
      hist_x0_total_new->Add( (TH2D*)prof2d_x0_det_total_new->ProjectionXY("B"), +1.000 );
    }
  }
  //
  
  // properties
  gStyle->SetPalette(1);
  gStyle->SetStripDecimals(false);
  //
  
  // Create "null" histo
  TH2F *frame = new TH2F("frame","",10,-3100.,3100.,10,-50.,1400.); 
  frame->SetMinimum(0.1);
  frame->SetMaximum(10.);
  frame->GetXaxis()->SetTickLength(frame->GetXaxis()->GetTickLength()*0.50);
  frame->GetYaxis()->SetTickLength(frame->GetXaxis()->GetTickLength()/4.);

  // Ratio
  TH2D* histo_ratio = new TH2D(*hist_x0_total_new);
  TString hist2dTitle = Form( "Material Budget Ratio (New/Old) (%s) ",quotaName.Data() ) + theDetector + Form( ";%s;%s;%s",abscissaName.Data(),ordinateName.Data(),quotaName.Data() );
  frame->SetTitle(hist2dTitle);
  frame->SetTitleOffset(0.5,"Y");
  histo_ratio->Divide(hist_x0_total_old);
  //
  
  //Set minimum and maximum
  if ( histoMin != -1. ) histo_ratio->SetMinimum(histoMin);      
  if ( histoMax != -1. ) histo_ratio->SetMaximum(histoMax);      

  // canvas
  TCanvas can("can","can",2480+248,580+58+58);
  can.SetTopMargin(0.1);
  can.SetBottomMargin(0.1);
  can.SetLeftMargin(0.04);
  can.SetRightMargin(0.06);
  can.SetFillColor(kWhite);
  gStyle->SetOptStat(0);
  //
  
  // Draw
  gStyle->SetPalette(1);

  // Log?
  can.SetLogz(zLog);

  // Draw in colors
  frame->Draw();
  histo_ratio->Draw("COLZsame");

  // Store
  can.Update();

  //Aesthetic
  TPaletteAxis *palette = 
    (TPaletteAxis*)histo_ratio->GetListOfFunctions()->FindObject("palette");
  palette->SetX1NDC(0.945);
  palette->SetX2NDC(0.96);
  palette->SetY1NDC(0.1);
  palette->SetY2NDC(0.9);
  palette->GetAxis()->SetTickSize(.01);
  if ( zLog==1 ) {  palette->GetAxis()->SetLabelOffset(-0.0045); }
  palette->GetAxis()->SetTitle("");
  TLatex *paletteTitle = new TLatex(3450.,1400.,quotaName); 
  paletteTitle->SetTextAngle(90.);
  paletteTitle->SetTextSize(0.05);
  paletteTitle->SetTextAlign(31);
  paletteTitle->Draw();     
  histo_ratio->GetYaxis()->SetTickLength(histo_ratio->GetXaxis()->GetTickLength()/4.);
  histo_ratio->GetYaxis()->SetTickLength(histo_ratio->GetXaxis()->GetTickLength()/4.);
  histo_ratio->SetTitleOffset(0.5,"Y");
  histo_ratio->GetXaxis()->SetNoExponent(true);
  histo_ratio->GetYaxis()->SetNoExponent(true);
  if ( zLog == 1 && abs(histo_ratio->GetMaximum()/histo_ratio->GetMinimum())<1000. ) { 
    palette->GetAxis()->SetMoreLogLabels(kTRUE); 
    palette->GetAxis()->SetNoExponent(kTRUE);
  }


  // Color plots not optimized... skipping for now
  //  can.SaveAs( Form( "%s/%s_ComparisonRatio_%s_col.eps",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can.SaveAs( Form( "%s/%s_ComparisonRatio_%s_col.gif",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //  can.SaveAs( Form( "%s/%s_ComparisonRatio_%s_col.pdf",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //
  
  // Double color palette
  Int_t ncol = 50;
  Int_t colors[100]; 
  TColor *col; 
  Double_t dg=1/(Double_t)(ncol-1);
  Double_t gray=0.;

  for (Int_t i=0; i<ncol; i++) {     
    colors[i]= i+100; 
    col = gROOT->GetColor(colors[i]);  
    if ( zCol == 0. ) { col->SetRGB(gray, gray, 1.); }
    if ( zCol == 1. ) { col->SetRGB(gray*gray, gray*gray, 1.); }
//    cout << i << " " << gray << "\n";
    gray = gray+dg;
  }
  gray = 1.;
  for (Int_t i=ncol; i<2*ncol; i++) {     
    colors[i]= i+100; 
    col = gROOT->GetColor(colors[i]);  
    if ( zCol == 0. ) { col->SetRGB(1., gray, gray); }
    if ( zCol == 1. ) { col->SetRGB(1., gray*gray, gray*gray); }
//    cout << i << " " << gray << "\n";;
    gray = gray-dg;
  }

  histo_ratio->SetContour(100);      
  gStyle->SetPalette(100,colors);
  
  // Store
  can.Update();

  //Add eta labels
  drawEtaValues();

  can.Modified();

  can.SaveAs( Form( "%s/%s_ComparisonRatio_%s.eps",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  can.SaveAs( Form( "%s/%s_ComparisonRatio_%s.gif",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  can.SaveAs( Form( "%s/%s_ComparisonRatio_%s.pdf",  theDirName.Data(), theDetector.Data(), plot.Data() ) );
  //

  // restore properties
  gStyle->SetStripDecimals(true);
  //
}

void drawEtaValues(){

  //Add eta labels
  Float_t etas[33] = {-3.4, -3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4., -1.2, -1., -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8, 3.0, 3.4};
  Float_t etax = 2940.;
  Float_t etay = 1240.;
  Float_t lineL = 100.;
  Float_t offT = 10.;
  Float_t pi = 3.1415926;
  for (Int_t ieta=0; ieta<33; ieta++){
    Float_t th = 2*atan(exp(-etas[ieta]));
    Int_t talign = 21;

    //IP
    TLine *lineh = new TLine(-20.,0.,20.,0.); 
    lineh->Draw();  
    TLine *linev = new TLine(0.,-10.,0.,10.); 
    linev->Draw();  

    if ( etas[ieta]>-1.6 && etas[ieta]<1.6 ){
      Float_t x1 = etay/tan(th);
      Float_t y1 = etay;
    } else if ( etas[ieta]<=-1.6 ) {
      Float_t x1 = -etax;
      Float_t y1 = -etax*tan(th);
      talign = 11;
    } else if ( etas[ieta]>=1.6 ){
      Float_t x1 = etax;
      Float_t y1 = etax*tan(th);
      talign = 31;
    }
    Float_t x2 = x1+lineL*cos(th);
    Float_t y2 = y1+lineL*sin(th);
    Float_t xt = x2;
    Float_t yt = y2+offT;
    //      cout << isign << " " << th*180./pi << " " << x1 << " " << y1 << "\n";
    TLine *line1 = new TLine(x1,y1,x2,y2); 
    line1->Draw();  
    char text[20];
    int rc = sprintf(text, "%3.1f", etas[ieta]);
    if ( etas[ieta] == 0 ) {
      TLatex *t1 = new TLatex(xt,yt,"#eta = 0"); 
    } else {
      TLatex *t1 = new TLatex(xt,yt,text); 
    }
    t1->SetTextSize(0.03);
    t1->SetTextAlign(talign);
    t1->Draw();     
    
  }

}

