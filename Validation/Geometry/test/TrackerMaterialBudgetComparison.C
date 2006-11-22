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

using namespace std;

// Main
TrackerMaterialBudgetComparison(TString detector) {
  // detector
  theDetector = detector;
  if(
     theDetector!="TIB" && theDetector!="TIDF" && theDetector!="TIDB" 
     && theDetector!="TOB" && theDetector!="TEC" && theDetector!="TkStrct" 
     && theDetector!="PixBar" && theDetector!="PixFwdPlus" && theDetector!="PixFwdMinus" 
     && theDetector!="Tracker" && theDetector!="TrackerSum"
     && theDetector!="Pixel" && theDetector!="Strip"
     ){
    cerr << "MaterialBudget - ERROR detector not found " << theDetector << endl;
    break;
  }
  //
  
  // files
  theDetectorFileName_old = "matbdg_" + theDetector + "_old.root";
  theDetectorFileName_new = "matbdg_" + theDetector + "_new.root";
  unsigned int iFirst = 1;
  unsigned int iLast  = 8;
  if(theDetector == "TrackerSum") {
    iFirst = 1;
    iLast  = 8;
    theDetectorFileName_old = "matbdg_TIB_old.root";
    theDetectorFileName_new = "matbdg_TIB_new.root";
  }
  if(theDetector == "Pixel") {
    iFirst = 7;
    iLast  = 8;
    theDetectorFileName_old = "matbdg_PixBar_old.root";
    theDetectorFileName_new = "matbdg_PixBar_new.root";
  }
  if(theDetector == "Strip") {
    iFirst = 1;
    iLast  = 4;
    theDetectorFileName_old = "matbdg_TIB_old.root";
    theDetectorFileName_new = "matbdg_TIB_new.root";
  }
  cout << "*** Open file... " << endl;
  cout << " old: " << theDetectorFileName_old << endl;
  cout << " new: " << theDetectorFileName_new << endl;
  cout << "***" << endl;
  //
  
  // open root files
  TFile* theDetectorFile_old = new TFile(theDetectorFileName_old);
  TFile* theDetectorFile_new = new TFile(theDetectorFileName_new);
  //
  
  // get TProfiles
  prof_x0_det_total_old = (TProfile*)theDetectorFile_old->Get("10");
  prof_x0_det_SUP_old   = (TProfile*)theDetectorFile_old->Get("110");
  prof_x0_det_SEN_old   = (TProfile*)theDetectorFile_old->Get("210");
  prof_x0_det_CAB_old   = (TProfile*)theDetectorFile_old->Get("310");
  prof_x0_det_COL_old   = (TProfile*)theDetectorFile_old->Get("410");
  prof_x0_det_ELE_old   = (TProfile*)theDetectorFile_old->Get("510");
  prof_x0_det_OTH_old   = (TProfile*)theDetectorFile_old->Get("610");
  prof_x0_det_AIR_old   = (TProfile*)theDetectorFile_old->Get("710");
  //
  prof_x0_det_total_new = (TProfile*)theDetectorFile_new->Get("10");
  prof_x0_det_SUP_new   = (TProfile*)theDetectorFile_new->Get("110");
  prof_x0_det_SEN_new   = (TProfile*)theDetectorFile_new->Get("210");
  prof_x0_det_CAB_new   = (TProfile*)theDetectorFile_new->Get("310");
  prof_x0_det_COL_new   = (TProfile*)theDetectorFile_new->Get("410");
  prof_x0_det_ELE_new   = (TProfile*)theDetectorFile_new->Get("510");
  prof_x0_det_OTH_new   = (TProfile*)theDetectorFile_new->Get("610");
  prof_x0_det_AIR_new   = (TProfile*)theDetectorFile_new->Get("710");
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
  
  if(theDetector=="TrackerSum" || theDetector=="Pixel" || theDetector=="Strip") {
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
	subDetector = "TOB";
	break;
      }
      case 4: {
	subDetector = "TEC";
	break;
      }
      case 5: {
	subDetector = "TkStrct";
	break;
      }
      case 6: {
	subDetector = "PixBar";
	break;
      }
      case 7: {
	subDetector = "PixFwdPlus";
	break;
      }
      case 8: {
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
      prof_x0_det_total_old = (TProfile*)subDetectorFile_old->Get("10");
      prof_x0_det_SUP_old   = (TProfile*)subDetectorFile_old->Get("110");
      prof_x0_det_SEN_old   = (TProfile*)subDetectorFile_old->Get("210");
      prof_x0_det_CAB_old   = (TProfile*)subDetectorFile_old->Get("310");
      prof_x0_det_COL_old   = (TProfile*)subDetectorFile_old->Get("410");
      prof_x0_det_ELE_old   = (TProfile*)subDetectorFile_old->Get("510");
      prof_x0_det_OTH_old   = (TProfile*)subDetectorFile_old->Get("610");
      prof_x0_det_AIR_old   = (TProfile*)subDetectorFile_old->Get("710");
      //
      prof_x0_det_total_new = (TProfile*)subDetectorFile_new->Get("10");
      prof_x0_det_SUP_new   = (TProfile*)subDetectorFile_new->Get("110");
      prof_x0_det_SEN_new   = (TProfile*)subDetectorFile_new->Get("210");
      prof_x0_det_CAB_new   = (TProfile*)subDetectorFile_new->Get("310");
      prof_x0_det_COL_new   = (TProfile*)subDetectorFile_new->Get("410");
      prof_x0_det_ELE_new   = (TProfile*)subDetectorFile_new->Get("510");
      prof_x0_det_OTH_new   = (TProfile*)subDetectorFile_new->Get("610");
      prof_x0_det_AIR_new   = (TProfile*)subDetectorFile_new->Get("710");
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
  // canvas
  TCanvas can_comparison("TkMB_Comparison","TkMB_Comparison",1200,800);
  can_comparison.Range(0,0,25,25);
  can_comparison.Divide(4,2);
  can_comparison.SetFillColor(kWhite);
  can_comparison.SetGridy(1);
  can_comparison.SetLogy(0);
  can_comparison.cd();
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
    
    // canvas
    can_comparison.cd(i_category);
    //
    // Compare
    histo_new->SetMarkerColor(2); // red
    histo_new->SetLineColor(102); // dark red
    histo_new->SetFillColor(0); // white
    histo_new->SetMarkerStyle(20); // cyrcles
    histo_new->SetMarkerSize(0.3); // 
    histo_old->SetLineColor(4); // blue
    histo_old->SetFillColor(0); // white
    histo_old->SetLineWidth(1.0); // 
    //
    // Draw
    histo_old->Draw("HIST");
    histo_new->Draw("HIST P E1 SAME");
    //
    // perform chi2 test between obtained and nominal histograms
    double compatibilityFactor = histo_new->KolmogorovTest(histo_old,"");
    std::cout << " Compatibility of " << histo_new->GetName()
	      << " with nominal distribution " << histo_old->GetName() << " is " << compatibilityFactor << std::endl;
    //
    
    // Legenda
    TLegend* theLegend = new TLegend(0.60, 0.60, 0.89, 0.89);
    theLegend->AddEntry( histo_old , "OLD" , "l" );
    theLegend->AddEntry( histo_new , "NEW" , "p" );
    theLegend->SetHeader( Form("KF: %f",compatibilityFactor) );
    theLegend->Draw();
    //
  }
  
  // Store
  can_comparison.Update();
  can_comparison.SaveAs( Form("%s/%s_Comparison.eps",  theDirName.Data(), theDetector.Data()) );
  can_comparison.SaveAs( Form("%s/%s_Comparison.gif",  theDirName.Data(), theDetector.Data()) );
  //
  
}
