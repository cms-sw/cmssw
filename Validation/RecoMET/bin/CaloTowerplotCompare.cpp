#include <iostream>
#include <sstream>
#include <math.h>

#include "TSystem.h"
#include "TFile.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TText.h"
#include "TPaveLabel.h"
#include "TLine.h"
#include "TLegend.h"

#include "include/PlotCompareUtility.h"
#include "include/HistoData.h"
using namespace std;

const bool KS_TEST = false;
const bool CHI2_TEST = true;

int main(int argc, char *argv[]) {

  // make sure command line arguments were supplied
  if (argc != 3) { cerr << "Usage: " << argv[0] << " [reference.root] [new-comparison.root]\n"; return 1; }

  // create the comparison class
  PlotCompareUtility *pc = new PlotCompareUtility(argv[1],argv[2],"DQMData/RecoMETV/METTask/CaloTowers/caloTowers","METTask_CT_");
  HistoData *hd;

  if (pc->GetStatus() != 0) { cout << "Final Result: no_data" << endl; return 0; }

  // add histogram information
  //hd = pc->AddHistoData("et_ieta_iphi"); hd->SetType(1); hd->SetValueX("Integrated Et (GeV)"); //*
  hd = pc->AddHistoData("etvsieta"); hd->SetType(2); hd->SetValueX("CaloTower Et (GeV)");
  //hd = pc->AddHistoData("emEt_ieta_iphi"); hd->SetType(1); hd->SetValueX("Integrated emEt (GeV)"); //*
  hd = pc->AddHistoData("emEtvsieta"); hd->SetType(2); hd->SetValueX("CaloTower emEt (GeV)");
  //hd = pc->AddHistoData("hadEt_ieta_iphi"); hd->SetType(1); hd->SetValueX("Integrated hadEt (GeV)"); //*
  hd = pc->AddHistoData("hadEtvsieta"); hd->SetType(2); hd->SetValueX("CaloTower hadEt (GeV)");
  //hd = pc->AddHistoData("energy_ieta_iphi"); hd->SetType(1); hd->SetValueX("Integrated energy (GeV)"); //*
  hd = pc->AddHistoData("energyvsieta"); hd->SetType(2); hd->SetValueX("CaloTower energy (GeV)");
  //hd = pc->AddHistoData("outerEnergy_ieta_iphi"); hd->SetType(1); hd->SetValueX("Integrated outerEnergy (GeV)"); //*
  hd = pc->AddHistoData("outerEnergyvsieta"); hd->SetType(2); hd->SetValueX("CaloTower outerEnergy (GeV)");
  //hd = pc->AddHistoData("hadEnergy_ieta_iphi"); hd->SetType(1); hd->SetValueX("Integrated hadEnergy (GeV)"); //*
  hd = pc->AddHistoData("hadEnergyvsieta"); hd->SetType(2); hd->SetValueX("CaloTower hadEnergy (GeV)");
  //hd = pc->AddHistoData("emEnergy_ieta_iphi"); hd->SetType(1); hd->SetValueX("Integrated emEnergy (GeV)"); //*
  hd = pc->AddHistoData("emEnergyvsieta"); hd->SetType(2); hd->SetValueX("CaloTower emEnergy (GeV)");
  //hd = pc->AddHistoData("Occ_ieta_iphi"); hd->SetType(1); hd->SetValueX("Integrated Occupancy");
  hd = pc->AddHistoData("Occ_ieta_iphi"); hd->SetType(1); hd->SetValueX("i#phi");
  hd = pc->AddHistoData("Occvsieta"); hd->SetType(2); hd->SetValueX("CaloTower Occupancy");
  hd = pc->AddHistoData("SETvsieta"); hd->SetType(2); hd->SetValueX("#eta-Ring SumET (GeV)");
  hd = pc->AddHistoData("METvsieta"); hd->SetType(2); hd->SetValueX("#eta-Ring MET (GeV)");
  hd = pc->AddHistoData("MExvsieta"); hd->SetType(2); hd->SetValueX("#eta-Ring MEx (GeV)");
  hd = pc->AddHistoData("MEyvsieta"); hd->SetType(2); hd->SetValueX("#eta-Ring MEy (GeV)");
  hd = pc->AddHistoData("METPhivsieta"); hd->SetType(2); hd->SetValueX("#eta-Ring MET-phi (radians)");

  if (pc->GetStatus() != 0) { cerr << "error encountered, exiting.\n"; return pc->GetStatus(); }


  // create overview results summary histogram
  int num_histos = pc->GetNumHistos();
  TH1F h2dResults("h2dResults","h2dResults", num_histos, 1, num_histos + 1);

  // arrays/variables to store test scores and overall pass/fail evaluations
  vector<float> low_score;
  vector<float> high_score;
  bool combinedFailed = false;
  vector<bool> testFailed;
  vector<bool> empty;
  vector<int> types;

  for (int index = 0; index < num_histos; index++) {

    // initialize. memset to non-zero is dangerous with float types
    low_score.push_back(1);
    high_score.push_back(0);
    testFailed.push_back(false);
    empty.push_back(true);
    types.push_back(0);

  }

  float threshold = KS_TEST ? pc->GetKSThreshold() : pc->GetChi2Threshold();


  // loop over the supplied list of histograms for comparison
  for (int index = 0; index < pc->GetNumHistos(); index++) {

    int number = index + 1;
    hd = pc->GetHistoData(number);
    int type = hd->GetType();
    types[index] = type;
    string name = hd->GetName();
    string value = hd->GetValueX();
    cout << name << endl;

    // get the reference and comparison 2d histograms
    TH2F *hnew2d = (TH2F *)pc->GetNewHisto(name);
    TH2F *href2d = (TH2F *)pc->GetRefHisto(name);

    // set this histograms label in the allResults histo
    h2dResults.GetXaxis()->SetBinLabel(number, name.c_str());

    // ignore if histogram is broken
    if (hnew2d->GetEntries() <= 1 || href2d->GetEntries() <= 1) {
      cout << "problem with histogram " << name << endl; continue;
    }

    // create 1d histograms to put projection results into
    string titleP = "h1dResults_passed_" + name;
    string titleF = "h1dResults_failed_" + name;
    TH1F h1dResults_passed(titleP.c_str(),titleP.c_str(),83,-41,42);
    TH1F h1dResults_failed(titleF.c_str(),titleF.c_str(),83,-41,42);

    // create a subdirectory in which to put resultant histograms
    gSystem->mkdir(name.c_str());

    // loop over ieta
    float running_result = 0; int counted_bins = 0; 
    for (int bin = 1; bin <= href2d->GetNbinsX(); bin++) {

      int ieta = bin - 42;
      stringstream bin_label;
      bin_label << ieta;

      // ignore the center bin
      if (bin == 42) continue;

      // unique names should _NOT_ be used (ProjectionY will re-alloc memory
      // each time... But... Root is buggy and won't reliably display)....
      stringstream hname_new, hname_ref;
      hname_new << "tmp_hnew_" << name << "_" << bin;
      hname_ref << "tmp_href_" << name << "_" << bin;

      // get the 1d projection for this ieta bin out of the histogram
      TH1D *hnew = hnew2d->ProjectionY(hname_new.str().c_str(),bin,bin);
      TH1D *href = href2d->ProjectionY(hname_ref.str().c_str(),bin,bin);

      // ignore empty bins
      if (hnew->Integral() == 0 || href->Integral() == 0) { 
        cout << "  ieta = " << ieta << ": empty!" << endl;
        continue;
      } else { empty[index] = false; }

      float ks_score, chi2_score, result;
      
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
      double NewMin = 0;
      double NewMax = 1;
      if (RMS>0) 
	{
	  dNdX = 100. / ( 10 * RMS);
	  NewMin = Mean - 5 * RMS;
	  NewMax = Mean + 5 * RMS;
	}
      
      int rebinning = 1;
      if ((dX * dNdX)>0) 
	rebinning = (int)(double(Nbins) / (dX * dNdX));
          
      // histograms of this type should be rescaled and rebinned
      if (type == 2) {
	href->GetXaxis()->SetRangeUser(NewMin, NewMax);
	hnew->GetXaxis()->SetRangeUser(NewMin, NewMax);
	
	if (rebinning > 1) {
	  href->Rebin(rebinning);
	  hnew->Rebin(rebinning);
	}
	
      } else { // type 1
	
        // this is a bug for reference versions AFTER 140
	/*
        if (abs(ieta) >= 40) {
          for (int i_=1; i_<=(int)href->GetNbinsX()-2;i_=i_+4) {
            float current_ = href->GetBinContent(i_);
            href->SetBinContent(i_ + 2, current_);
            href->SetBinContent(i_, 0);
          }
        }
	*/
        //href->GetXaxis()->SetTitle("i#phi");
        //href->GetYaxis()->SetTitle(value.c_str());

      }

      //href->SetMinimum(0);
      //hnew->SetMinimum(0);

      href->GetXaxis()->SetTitle(value.c_str());
      href->GetYaxis()->SetTitle("Entries");
      href->GetYaxis()->SetTitleOffset(1.5);
      href->SetNormFactor(hnew->GetEntries());
      //      hnew->SetNormFactor(1);

      // ensure that the peaks of both histograms will be shown
      //if (href->GetMaximum() < hnew->GetMaximum())
      //  href->SetAxisRange(0, 1.03 * hnew->GetMaximum(), "Y");

      ks_score = hnew->KolmogorovTest(href);
      chi2_score = hnew->Chi2Test(href );
      // if (KS_TEST) result = ks_score; else result = chi2_score;
      result = (ks_score>chi2_score) ? ks_score : chi2_score;
      running_result += result; counted_bins++;

      if (result > high_score[index]) { high_score[index] = result; }
      if (result < low_score[index]) { low_score[index] = result; }

      href->SetNormFactor(hnew->GetEntries());

      // ensure that the peaks of both histograms will be shown by making a dummy histogram
      float Nentries_ref = href->GetEntries();
      float Nentries_new = hnew->GetEntries();
      float XaxisMin_ref = 0, XaxisMax_ref = 0, YaxisMin_ref = 0, YaxisMax_ref = 0;
      float XaxisMin_new = 0, XaxisMax_new = 0, YaxisMin_new = 0, YaxisMax_new = 0;
      if (Nentries_ref>0) YaxisMax_ref = (href->GetMaximum()+sqrt(href->GetMaximum()))*(Nentries_new/Nentries_ref);
      if (Nentries_new>0) YaxisMax_new = (hnew->GetMaximum()+sqrt(hnew->GetMaximum()));
      
      XaxisMin_ref = href->GetXaxis()->GetXmin()>NewMin  ? href->GetXaxis()->GetXmin() : NewMin;
      XaxisMax_ref = href->GetXaxis()->GetXmax()<=NewMax ? href->GetXaxis()->GetXmax() : NewMax;
      YaxisMax_ref = (YaxisMax_ref>=YaxisMax_new) ? YaxisMax_ref : YaxisMax_new;

      cout << "!!! " << Nentries_ref << " " << Nentries_new << " :  " << YaxisMax_ref << " " << YaxisMax_new << endl;

      if (fabs(XaxisMin_ref - XaxisMax_ref)<1E-6)
	{
	  XaxisMin_ref = 0;
	  XaxisMax_ref = 1;
	}

      TH1F *hdumb = new TH1F("hdumb","", rebinning, XaxisMin_ref, XaxisMax_ref);
      hdumb->SetMinimum(1E-1);
      hdumb->SetMaximum(1.05*YaxisMax_ref);

      // set drawing options on the reference histogram
      href->SetStats(0);
      href->SetLineWidth(2);
      href->SetLineColor(14);
      href->SetMarkerColor(14);
      href->SetFillColor(17);
      //href->SetFillStyle(3001);

      // set drawing options on the new histogram
      hnew->SetStats(0);
      hnew->SetLineWidth(2);
      hnew->SetFillStyle(3001);

      // set drawing options on the dummy histogram
      hdumb->SetStats(0);
      hdumb->GetXaxis()->SetTitle(name.c_str());
      hdumb->GetXaxis()->SetLabelSize(0.5 * hdumb->GetXaxis()->GetTitleSize());
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

      if (result <= threshold) {

        // make this histogram red to denote failure
        hnew->SetFillColor(kRed);
        hnew->SetLineColor(206);
        hnew->SetMarkerColor(206);
        //hnew->SetLineColor(206);
        //hnew->SetMarkerColor(206);

        // mark the sample as being 'not-compatible'
        testFailed[index] = true;
        combinedFailed = true;

        // set the summary bin to failed (only need to set titles for passed h1dResults)
        //h1dResults_passed.GetXaxis()->SetBinLabel(bin, bin_label.str().c_str());
        h1dResults_failed.SetBinContent(bin,result);

        // set the canvas title
        canvas_title.SetTextColor(kRed);

      } else {

        // make this histogram green to denote passing
        hnew->SetFillColor(kGreen);
        hnew->SetLineColor(103);
        hnew->SetMarkerColor(103);
        //hnew->SetLineColor(103);
        //hnew->SetMarkerColor(103);

        // set the summary bin to passed
        //h1dResults_passed.GetXaxis()->SetBinLabel(bin, bin_label.str().c_str());
        h1dResults_passed.SetBinContent(bin,result);

        // set the canvas title
        canvas_title.SetTextColor(kGreen);

      }

      // setup canvas for displaying the compared histograms
      TCanvas histo_c("histo_c","histo_c",785,800);
      histo_c.Draw();

      TPad histo_p("histo_p","histo_p",0.01,0.01,0.99,0.94);
      histo_p.Draw();

      histo_c.cd();
      canvas_title.SetTextSize(0.025);
      canvas_title.Draw();

      histo_p.cd();
  
      hdumb->Draw();
      histo_p.SetLogy(1);
      href->Draw("SAME");
      hnew->Draw("SAME");
      hnew->Draw("E1SAME");

      TLegend l1(0.15,0.01,0.3, 0.08);
      l1.AddEntry(hnew,"New","lF");
      l1.AddEntry(href,"Reference","lF");
      l1.SetFillColor(kNone);
      l1.Draw("SAME");

      // print the result to gif
      stringstream histo_name;
      histo_name << name << "/Compare_ietaBin";
      if (bin < 10) histo_name << "00";
      else if (bin < 100) histo_name << "0";
      histo_name << bin << ".gif";
      histo_c.Print(histo_name.str().c_str(),"gif");

      // report the obtained KS score
      cout << "  ieta = " << ieta << ": result = " << result << endl; 

    } // end loop over ieta bins

    // create ieta summary canvas
    TCanvas ieta_c("ieta_c","ieta_c",1000,500);
    ieta_c.Draw();

    TPad ieta_p("ieta_p","ieta_p",0,0,1,1);
    //ieta_p.SetLeftMargin(0.30);
    //ieta_p.SetBottomMargin(0.15);
    ieta_p.SetLogy(1);
    //ieta_p.SetGrid();
    ieta_p.SetFrameFillColor(10);
    ieta_p.Draw();

    ieta_c.cd();
    TText ieta_title(.01, .01, "");
    ieta_title.Draw("SAME");

    ieta_p.cd();

    // setup the passing test bars
    h1dResults_passed.SetStats(0);
    h1dResults_passed.GetXaxis()->SetLabelSize(0.03);
    h1dResults_passed.GetXaxis()->SetTitle("i#eta Ring");
    //h1dResults_passed.GetXaxis()->SetBit(TAxis::kLabelsVert);
    h1dResults_passed.GetYaxis()->SetTitle("Compatibility");
    h1dResults_passed.GetYaxis()->SetLabelSize(0.03);
    h1dResults_passed.SetBarWidth(0.7);
    h1dResults_passed.SetBarOffset(0.1);
    //h1dResults_passed.SetLineColor(1);
    h1dResults_passed.GetYaxis()->SetRangeUser(1E-7,2);

    // setup the failing test bars
    h1dResults_failed.SetStats(0);
    //h1dResults_failed.GetXaxis()->SetLabelSize(0.04);
    //h1dResults_failed.GetYaxis()->SetTitle("Compatibility");
    h1dResults_failed.SetBarWidth(0.7);
    h1dResults_failed.SetBarOffset(0.1);
    //h1dResults_failed.SetLineColor(1);
    h1dResults_failed.GetYaxis()->SetRangeUser(1E-7,2);

    if (empty[index]) {
      // do nothing
/*  }  else if (type == 1) {
      h1dResults_passed.SetFillColor(18);
      h1dResults_passed.SetLineColor(18);
      h1dResults_passed.Draw("bar");*/
    } else {
      h1dResults_passed.SetFillColor(kGreen);
      h1dResults_passed.SetLineColor(kGreen);
      h1dResults_failed.SetFillColor(kRed);
      h1dResults_failed.SetLineColor(kRed);
      h1dResults_passed.Draw("bar");
      h1dResults_failed.Draw("barSAME");
    }

    // draw the pass/fail threshold line
    TLine l(-41, threshold, 42, threshold);
    l.SetLineColor(kRed);
    l.SetLineWidth(2);
    l.SetLineStyle(2);
    l.Draw("SAME");

    // print the results
    ieta_c.Update();
    string ieta_name = name + "/Results_vs_etaRing.gif";
    ieta_c.Print(ieta_name.c_str(),"gif");
     
    // report the result of the tests, if performed
    cout << "result is: ";
    if (type == 1) cout << "no determination made";
    else if (testFailed[index]) cout << "fail";
    else if (empty[index]) cout << "no data";
    else cout << "pass";
    float mean = counted_bins > 0 ? running_result / counted_bins : 0;
    cout << ", mean KS score = " << mean << endl;



  } // end loop over 2d histograms

  // create summary canvas
  int summary_height = int(780 * float(num_histos) / 22); // 780;
  TCanvas main_c("main_c","main_c",799,summary_height);
  main_c.Draw();

  //TPad main_p("main_p","main_p",0.15,0.05,0.99,0.94);
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

  // this histogram should be empty -- filling only with the TBox objects
  h2dResults.SetStats(0);
  h2dResults.SetBarWidth(0.45);
  h2dResults.SetBarOffset(0.1);
  h2dResults.GetYaxis()->SetRangeUser(1E-7,2);
  h2dResults.GetYaxis()->SetTitle("Compatibility");
  h2dResults.SetBinContent(1,0.5);
  h2dResults.SetLabelSize(0.08*22.0/(1+float(num_histos)));
  h2dResults.GetYaxis()->SetLabelSize(0.04*22.0/(1.0+float(num_histos)));
  h2dResults.GetXaxis()->SetLabelSize(0.04*22.0/(1.0+float(num_histos)));
  h2dResults.Draw("hbar");

  // fill in a display box on the overview compatibility range histogram
  TBox box;
  box.SetLineWidth(2);
  for (int i = 0; i < num_histos; i++) {

    // form a display box based on the range of scores
    float box_min = low_score[i];
    float box_max = high_score[i];

    // ensure that the box does not go below the range of the results histo
    if (box_min < 1E-7) box_min = 1E-7;
    if (box_max < 1E-7) box_max = 1E-7;

    // ensure that something is drawn, even if length = 0 (such as all tests = 1 or 0)
    if (fabs(box_min - box_max) < 1E-7) {

      TLine box_line;
      box_line.SetLineWidth(4);

      float line_pos = (box_max + box_min) / 2;
      if (types[i] == 1)
        box_line.SetLineColor(18);
      else if (testFailed[i])
        box_line.SetLineColor(kRed);
      else box_line.SetLineColor(kGreen);
      
      box_line.DrawLine(line_pos, i + 1.25, line_pos, i + 1.75);
      continue;
    }

    if (empty[i]) {
      box_min = 1E-7; box_max = 1;
      box.SetFillStyle(3005);
      box.SetFillColor(kBlack);
      box.SetLineColor(kBlack);
      box.SetLineStyle(1);
  /*  } else if (types[i] == 1) {
      //box_min = 1E-7; box_max = 1;
      box.SetFillStyle(1001);
      box.SetFillColor(18);
      box.SetLineColor(18);
      box.SetLineStyle(0);*/
    } else if (testFailed[i]) {
      box.SetFillStyle(1001);
      box.SetLineStyle(0);
      box.SetLineColor(kRed);
      box.SetFillColor(kRed);
    } else {
      box.SetFillStyle(1001);
      box.SetLineStyle(0);
      box.SetLineColor(kGreen);
      box.SetFillColor(kGreen);
    }

    // draw the box
    box.DrawBox(box_min, i + 1.25, box_max, i + 1.75);

  }

  // draw the pass/fail threshold line
  TLine l(threshold, 1, threshold, num_histos + 1);
  l.SetLineColor(kRed);
  l.SetLineWidth(2);
  l.SetLineStyle(2);
  l.Draw("SAME");

  // print the results
  //main_c.Update();
  main_c.Print("AllResults-HistoCheck.gif","gif");

  if (combinedFailed) cout << "Final Result: fail" << endl;
  else cout << "Final Result: pass" << endl;

  //delete pc;
  return 0;

}
