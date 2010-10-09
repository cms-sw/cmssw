#include <iostream>
#include <sstream>

#include "TSystem.h"
#include "TFile.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TH1F.h"
#include "TText.h"
#include "TPaveLabel.h"
#include "TLine.h"
#include "TLegend.h"
#include "TMath.h"
#include "PlotCompareUtility.h"
#include "HistoData.h"
using namespace std;

const bool KS_TEST = true;
const bool CHI2_TEST = false;

int main(int argc, char *argv[]) {

  // make sure command line arguments were supplied
  if (argc != 6) { cerr << "Usage: " << argv[0] << " [reference.root] [new-comparison.root] [root dir] [new-release] [old-release] \n"; return 1; }

  // create the comparison class
  PlotCompareUtility *pc = new PlotCompareUtility(argv[1],argv[2],argv[3],"METTask_");
  HistoData *hd;

  if (pc->GetStatus() != 0) { cout << "Final Result: no_data" << endl; return 0; }

  // add histogram information
  //Type = 0 (Do not rebin or zoom) , 1 (Rebin and Zoom, x-axis > 0 ) , 2 (Rebin and Zoom)



  hd = pc->AddHistoData("METPhiResolution_GenMETCalo"); hd->SetType(4);
  hd = pc->AddHistoData("METResolution_GenMETCalo"); hd->SetType(3);
  hd = pc->AddHistoData("METPhiResolution_GenMETTrue"); hd->SetType(4);
  hd = pc->AddHistoData("METResolution_GenMETTrue"); hd->SetType(3);
  hd = pc->AddHistoData("MET"); hd->SetType(1);
  hd = pc->AddHistoData("METPhi"); hd->SetType(4);
  hd = pc->AddHistoData("METSig"); hd->SetType(1);
  hd = pc->AddHistoData("MEx"); hd->SetType(2);
  hd = pc->AddHistoData("MEy"); hd->SetType(2);
  hd = pc->AddHistoData("SumET"); hd->SetType(2);



  if (pc->GetStatus() != 0) { cerr << "error encountered, exiting.\n"; return pc->GetStatus(); }

  int num_histos = pc->GetNumHistos();
  bool combinedFailed = false;
  float threshold = KS_TEST ?  pc->GetKSThreshold() : pc->GetChi2Threshold(); 

    // get the reference and comparison histograms
  int Nevents_ref = ((TH1F *)pc->GetRefHisto("Nevents"))->GetEntries();
  int Nevents_new = ((TH1F *)pc->GetNewHisto("Nevents"))->GetEntries();
  int Nevents = -1;
  if (Nevents_ref>Nevents_new) Nevents = Nevents_ref;
  else Nevents = Nevents_new;

  // create summary histograms
  TH1F h1dResults_passed("h1dResults_passed","",num_histos, 1, num_histos + 1);
  TH1F h1dResults_failed("h1dResults_failed","",num_histos, 1, num_histos + 1);

  // loop over the supplied list of histograms for comparison
  for (int index = 0; index < pc->GetNumHistos(); index++) {

    int number = index + 1;
    hd = pc->GetHistoData(number);
    //int type = hd->GetType();
    //types[index] = type;
    string name = hd->GetName();
    //string value = hd->GetValueX();
    cout << name << endl;

    // get the reference and comparison histograms
    TH1F *href = (TH1F *)pc->GetRefHisto(name);
    TH1F *hnew = (TH1F *)pc->GetNewHisto(name);


    // ignore if histogram is empty
    if (hnew->GetEntries() <= 1 || href->GetEntries() <= 1) {
      cerr << name << " error: no entries"; combinedFailed = true; continue;
    }

    // calculate and set range and number of bins
    double h1RMS =  hnew->GetRMS();
    double h2RMS =  href->GetRMS();
    double RMS = TMath::Max(h1RMS, h2RMS);
    double h1Mean =  hnew->GetMean();
    double h2Mean =  href->GetMean();
    double Mean = 0.5 * (h1Mean + h2Mean);
    double Nbins = href->GetNbinsX();
    double min = href->GetXaxis()->GetXmin();
    double max = href->GetXaxis()->GetXmax();
    double dX = max - min;
    double dNdX = 1;
    double NewMin = min;
    double NewMax = max;

    int rebinning = Nbins; 

    if (RMS>0 && hd->GetType() ) 
      {
	dNdX = 100. / ( 10 * RMS);
	NewMin = Mean - 10 * RMS;
	NewMax = Mean + 10 * RMS;
      }
    
    if ((dX * dNdX)>0  && hd->GetType() ) 
      rebinning = (int)(double(Nbins) / (dX * dNdX));
    
    if ( rebinning > 1 && hd->GetType() ) 
      {
	href->Rebin(rebinning);
	hnew->Rebin(rebinning);
      }

    if ( hd->GetType() == 1 )
      { 
	href->GetXaxis()->SetRangeUser(0.0, NewMax);
	hnew->GetXaxis()->SetRangeUser(0.0, NewMax);
      }
    else if ( hd->GetType() == 2 || hd->GetType() != 3 )
      {
	href->GetXaxis()->SetRangeUser(NewMin, NewMax);
	hnew->GetXaxis()->SetRangeUser(NewMin, NewMax);
      }

    // perform statistical tests
    double ks_score = hnew->KolmogorovTest(href,"D");
    double chi2_score = hnew->Chi2Test(href, "p");
    //double result = KS_TEST ? ks_score : chi2_score;
    double result = (ks_score>chi2_score) ? ks_score : chi2_score;
    
      href->SetNormFactor(Nevents_new);     	
      hnew->SetNormFactor(Nevents_new);
    //hnew->SetNormFactor(1);

    // ensure that the peaks of both histograms will be shown by making a dummy histogram
    float Nentries_ref = href->GetEntries();
    float Nentries_new = hnew->GetEntries();
    float XaxisMin_ref = 0, XaxisMax_ref = 0, YaxisMin_ref = 0, YaxisMax_ref = 0;
    float XaxisMin_new = 0, XaxisMax_new = 0, YaxisMin_new = 0, YaxisMax_new = 0;
    if (Nentries_ref>0) YaxisMax_ref = (href->GetMaximum()+TMath::Sqrt(href->GetMaximum()))*(Nentries_new/Nentries_ref);
    if (Nentries_new>0) YaxisMax_new = (hnew->GetMaximum()+TMath::Sqrt(hnew->GetMaximum()));

    XaxisMin_ref = href->GetXaxis()->GetXmin()>NewMin  ? href->GetXaxis()->GetXmin() : NewMin;
    XaxisMax_ref = href->GetXaxis()->GetXmax()<=NewMax ? href->GetXaxis()->GetXmax() : NewMax;
    YaxisMax_ref = (YaxisMax_ref>=YaxisMax_new) ? YaxisMax_ref : YaxisMax_new;

    if (TMath::Abs(XaxisMin_ref - XaxisMax_ref)<1E-6)
      {
	XaxisMin_ref = 0;
	XaxisMax_ref = 1;
      }
    
    TH1F *hdumb = new TH1F("hdumb","", rebinning, XaxisMin_ref, XaxisMax_ref);
    hdumb->SetMinimum(1E-1); //--For Rick
    hdumb->SetMaximum(1.05*YaxisMax_ref);
    //    if (href->GetMaximum() < hnew->GetMaximum())
    //  href->SetAxisRange(0, 1.1 * hnew->GetMaximum(), "Y");
        
    // set drawing options on the reference histogram
    href->SetStats(0);
    href->SetLineWidth(1);
    href->SetLineColor(14);
    href->SetMarkerColor(14);
    href->SetFillColor(17);
    //href->SetFillStyle(3004);
    href->GetXaxis()->SetTitle(name.c_str());
    href->GetYaxis()->SetTitle("Entries");
    href->GetYaxis()->SetTitleOffset(1.5);

    // set drawing options on the new histogram
    hnew->SetStats(0);
    hnew->SetLineWidth(1);
    hnew->SetFillStyle(3001);
    // set drawing options on the dummy histogram
    hdumb->SetStats(0);
    hdumb->GetXaxis()->SetTitle(name.c_str());
    hdumb->GetXaxis()->SetLabelSize(0.5 * hdumb->GetXaxis()->GetTitleSize());
    hdumb->GetXaxis()->SetTitleSize(0.6 * hdumb->GetXaxis()->GetTitleSize());
    hdumb->GetYaxis()->SetTitle("Entries");
    hdumb->GetYaxis()->SetTitleOffset(1.5);
    hdumb->GetYaxis()->SetLabelSize(0.5 * hdumb->GetXaxis()->GetTitleSize());
    
    stringstream ss_title;
    ss_title.precision(5);
    if (ks_score>chi2_score)
      ss_title << "KS Score = " << ks_score;
    else
      ss_title << "Chi^2 Score = " << chi2_score;
    TText canvas_title(0.1,0.97,ss_title.str().c_str());


    // determine if test is a "pass" or a "fail"
    if (result <= threshold) {

      canvas_title.SetTextColor(kRed);

      // make this histogram red to denote failure
      hnew->SetFillColor(kRed);
      hnew->SetLineColor(206);
      hnew->SetMarkerColor(206);

      // mark the entire sample as being 'not-compatible'
      combinedFailed = true;

      // set the summary bin to failed (only need to set titles for passed h1dResults)
      h1dResults_passed.GetXaxis()->SetBinLabel(number, name.c_str());
      h1dResults_failed.SetBinContent(number, result);

    } else {

      canvas_title.SetTextColor(kGreen);

      // make this histogram green to denote passing score
      hnew->SetFillColor(kGreen);
      hnew->SetLineColor(kGreen+2);  
      hnew->SetMarkerColor(kGreen+2);

      // set the summary bin to passed
      h1dResults_passed.GetXaxis()->SetBinLabel(number, name.c_str());
      h1dResults_passed.SetBinContent(number, result);

    }

    // setup canvas for displaying the compared histograms
    TCanvas histo_c("histo_c","histo_c",785,800);
    histo_c.Draw();

    TPad histo_p("histo_p","histo_p",0,0,1,0.99);
    histo_p.Draw();

    histo_c.cd();
    canvas_title.SetTextSize(0.025);
    canvas_title.Draw();

    histo_p.cd();

    if( hd->GetType() < 3 )histo_p.SetLogy(1); //--This is just for Dr. Rick
    hdumb->Draw();
    href->Draw("SAME");
    hnew->Draw("SAME");
    hnew->Draw("E1SAME");
    
    stringstream legend_new;
    stringstream legend_ref;
    legend_new << argv[4] << ": " << Nentries_new << " entries, " << Nevents_new << " events";
    legend_ref << argv[5] << ": " << Nentries_ref << " entries, " << Nevents_ref << " events";		
    
    TLegend l1(0.15,0.001,0.33, 0.06);
    l1.SetTextSize(0.02);
    l1.AddEntry(hnew, legend_new.str().c_str(),"lF");
    l1.AddEntry(href, legend_ref.str().c_str(),"lF");
    l1.SetFillColor(kNone);      
    l1.Draw("SAME");
    

    // print the result to gif
    string histo_name = name + ".gif";
    histo_c.Print(histo_name.c_str(),"gif");
    cout << "Result of comparison for " << name << ": ks score = " << ks_score << " : chi2 score = " << chi2_score << endl << endl;

  }

  // create summary canvas
  int summary_height = int(780 * float(num_histos) / 11); // 780;
  TCanvas main_c("main_c","main_c",799,summary_height);
  main_c.Draw();

  TPad main_p("main_p","main_p",0.01,0.01,0.99,0.94);
  main_p.SetLeftMargin(0.30);
  main_p.SetBottomMargin(0.15);
  main_p.SetLogx(1);
  main_p.SetGrid();
  main_p.SetFrameFillColor(10);
  main_p.Draw();

  main_c.cd();
  TText summary_title(.01, .95, "");
  summary_title.Draw("SAME");

  main_p.cd();

  // setup the passing test bars
  h1dResults_passed.SetStats(0);
  h1dResults_passed.GetXaxis()->SetLabelSize(0.03);
  h1dResults_passed.GetYaxis()->SetLabelSize(0.04);
  h1dResults_passed.GetYaxis()->SetTitle("Compatibility");
  h1dResults_passed.SetBarWidth(0.7);
  h1dResults_passed.SetBarOffset(0.1);
  h1dResults_passed.SetFillColor(kGreen);
  h1dResults_passed.SetLineColor(1);
  h1dResults_passed.GetYaxis()->SetRangeUser(1E-7,2);
  h1dResults_passed.Draw("hbar0");

  // setup the failing test bars
  h1dResults_failed.SetStats(0);
  h1dResults_failed.GetXaxis()->SetLabelSize(0.03);
  h1dResults_failed.GetYaxis()->SetLabelSize(0.04);
  h1dResults_failed.GetYaxis()->SetTitle("Compatibility");
  h1dResults_failed.SetBarWidth(0.7);
  h1dResults_failed.SetBarOffset(0.1);
  h1dResults_failed.SetFillColor(kRed);
  h1dResults_failed.SetLineColor(1);
  h1dResults_failed.GetYaxis()->SetRangeUser(1E-7,2);
  h1dResults_failed.Draw("hbar0SAME");

  // draw the pass/fail threshold line
  TLine l(threshold, 1, threshold, num_histos+1);
  l.SetLineColor(kRed);
  l.SetLineWidth(2);
  l.SetLineStyle(2);
  l.Draw("SAME"); 

  // print the results
  main_c.Update();
  main_c.Print("AllResults-1dHistoCheck.gif","gif");

  if (combinedFailed) cout << "Final Result: fail" << endl;
  else cout << "Final Result: pass" << endl;

  //delete pc;
  return 0;

}
