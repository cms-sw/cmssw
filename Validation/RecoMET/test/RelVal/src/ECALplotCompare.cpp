#include <iostream>
#include <sstream>
#include "TMath.h"
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

#include "PlotCompareUtility.h"
#include "HistoData.h"
using namespace std;

const bool KS_TEST = false;
const bool CHI2_TEST = true;

int main(int argc, char *argv[]) {

  // make sure command line arguments were supplied
  if (argc != 6) { cerr << "Usage: " << argv[0] << " [reference.root] [new-comparison.root] [root-dir] [new-release] [old-release] \n"; return 1; }

  // create the comparison class
  PlotCompareUtility *pc = new PlotCompareUtility(argv[1],argv[2],argv[3],"");
  HistoData *hd;

  if (pc->GetStatus() != 0) { cout << "Final Result: no_data" << endl; return 0; }

  // add histogram information
  /*
  hd = pc->AddHistoData("hEEpZ_energy_ix_iy"); hd->SetType(3); hd->SetValueX("EE+ Energy (GeV)");  
  hd = pc->AddHistoData("hEEmZ_energy_ix_iy"); hd->SetType(3); hd->SetValueX("EE- Energy (GeV)");
  hd = pc->AddHistoData("hEB_energy_ieta_iphi"); hd->SetType(1); hd->SetValueX("EB Energy (GeV)");
  
  hd = pc->AddHistoData("hEEpZ_Minenergy_ix_iy"); hd->SetType(3); hd->SetValueX("EE+ Min. Energy (GeV)");
  hd = pc->AddHistoData("hEEmZ_Minenergy_ix_iy"); hd->SetType(3); hd->SetValueX("EE- Min. Energy (GeV)");
  hd = pc->AddHistoData("hEB_Minenergy_ieta_iphi"); hd->SetType(1); hd->SetValueX("EB Min. Energy (GeV)");
 
  hd = pc->AddHistoData("hEEpZ_Maxenergy_ix_iy"); hd->SetType(3); hd->SetValueX("EE+ Max. Energy (GeV)");
  hd = pc->AddHistoData("hEEmZ_Maxenergy_ix_iy"); hd->SetType(3); hd->SetValueX("EE- Max. Energy (GeV)");
  hd = pc->AddHistoData("hEB_Maxenergy_ieta_iphi"); hd->SetType(1); hd->SetValueX("EB Max. Energy (GeV)");

  hd = pc->AddHistoData("hEEpZ_Occ_ix_iy"); hd->SetType(3); hd->SetValueX("EE+ Occupancy ");
  hd = pc->AddHistoData("hEEmZ_Occ_ix_iy"); hd->SetType(3); hd->SetValueX("EE- Occupancy ");
  hd = pc->AddHistoData("hEB_Occ_ieta_iphi"); hd->SetType(1); hd->SetValueX("EB Occupancy ");
  */

  hd = pc->AddHistoData("hEEpZ_energyvsir"); hd->SetType(4); hd->SetValueX("EE+ Energy");
  hd = pc->AddHistoData("hEEmZ_energyvsir"); hd->SetType(4); hd->SetValueX("EE- Energy");
  hd = pc->AddHistoData("hEB_energyvsieta"); hd->SetType(2); hd->SetValueX("EB Energy");

  hd = pc->AddHistoData("hEEpZ_Maxenergyvsir"); hd->SetType(4); hd->SetValueX("EE+ Max. Energy");
  hd = pc->AddHistoData("hEEmZ_Maxenergyvsir"); hd->SetType(4); hd->SetValueX("EE- Max. Energy");
  hd = pc->AddHistoData("hEB_Maxenergyvsieta"); hd->SetType(2); hd->SetValueX("EB Max. Energy");
    
  hd = pc->AddHistoData("hEEpZ_Minenergyvsir"); hd->SetType(4); hd->SetValueX("EE+ Min. Energy");
  hd = pc->AddHistoData("hEEmZ_Minenergyvsir"); hd->SetType(4); hd->SetValueX("EE- Min. Energy");
  hd = pc->AddHistoData("hEB_Minenergyvsieta"); hd->SetType(2); hd->SetValueX("EB Min. Energy");
   
  hd = pc->AddHistoData("hEEpZ_SETvsir"); hd->SetType(4); hd->SetValueX("EE+ SET");
  hd = pc->AddHistoData("hEEmZ_SETvsir"); hd->SetType(4); hd->SetValueX("EE- SET");
  hd = pc->AddHistoData("hEB_SETvsieta"); hd->SetType(2); hd->SetValueX("EB SET");
 
  
  hd = pc->AddHistoData("hEEpZ_METvsir"); hd->SetType(4); hd->SetValueX("EE+ MET");  
  hd = pc->AddHistoData("hEEmZ_METvsir"); hd->SetType(4); hd->SetValueX("EE- MET");
  hd = pc->AddHistoData("hEB_METvsieta"); hd->SetType(2); hd->SetValueX("EB MET");
 
  hd = pc->AddHistoData("hEEpZ_METPhivsir"); hd->SetType(4); hd->SetValueX("EE+ METPhi");
  hd = pc->AddHistoData("hEEmZ_METPhivsir"); hd->SetType(4); hd->SetValueX("EE- METPhi");
  hd = pc->AddHistoData("hEB_METPhivsieta"); hd->SetType(2); hd->SetValueX("EB METPhi");
 
  hd = pc->AddHistoData("hEEpZ_MExvsir"); hd->SetType(4); hd->SetValueX("EE+ MEx");
  hd = pc->AddHistoData("hEEmZ_MExvsir"); hd->SetType(4); hd->SetValueX("EE- MEx");
  hd = pc->AddHistoData("hEB_MExvsieta"); hd->SetType(2); hd->SetValueX("EB MEx");
  
  
  hd = pc->AddHistoData("hEEpZ_MEyvsir"); hd->SetType(4); hd->SetValueX("EE+ MEy");
  hd = pc->AddHistoData("hEEmZ_MEyvsir"); hd->SetType(4); hd->SetValueX("EE- MEy");
  hd = pc->AddHistoData("hEB_MEyvsieta"); hd->SetType(2); hd->SetValueX("EB MEy");
  
  
  hd = pc->AddHistoData("hEEpZ_Occvsir"); hd->SetType(4); hd->SetValueX("EE+ Occupancy");
  hd = pc->AddHistoData("hEEmZ_Occvsir"); hd->SetType(4); hd->SetValueX("EE- Occupancy"); 
  hd = pc->AddHistoData("hEB_Occvsieta"); hd->SetType(2); hd->SetValueX("EB Occupancy");

  if (pc->GetStatus() != 0) { cerr << "error encountered, exiting.\n"; return pc->GetStatus(); }

  // create overview results summary histogram
  int num_histos = pc->GetNumHistos();
  TH1F h2dResults("h2dResults","h2dResults", num_histos, 1, num_histos + 1);

  // get the reference and comparison Nevents (not Nentries)	
  int Nevents_ref = ((TH1F *)pc->GetRefHisto("hECAL_Nevents"))->GetEntries();
  int Nevents_new = ((TH1F *)pc->GetNewHisto("hECAL_Nevents"))->GetEntries();
  int Nevents = -1;
  if (Nevents_ref>Nevents_new) Nevents = Nevents_ref;
  else Nevents = Nevents_new;


 // arrays/variables to store test scores and overall pass/fail evaluations
  float low_score[num_histos], high_score[num_histos];
  bool combinedFailed = false;
  bool testFailed[num_histos];
  bool empty[num_histos];
  int types[num_histos];

  for (int index = 0; index < num_histos; index++) {

    // initialize. memset to non-zero is dangerous with float types
    low_score[index] = 1;
    high_score[index] = 0;
    testFailed[index] = false;
    empty[index] = true;
    types[index] = 0;
  }

  float threshold = KS_TEST ? pc->GetKSThreshold() : pc->GetChi2Threshold();


  // loop over the supplied list of histograms for comparison
  for (int index = 0; index < pc->GetNumHistos(); index++) {
    
    int number = index + 1;
    hd = pc->GetHistoData(number);
    int type = hd->GetType();
    types[index] = type;
    
    // ECAL Plot compare can only accommodate types 2 and 4 histos//
    if( type != 2 && type != 4) { cout << "type = " << type  << " cannot be accommodated" << endl; continue;} 
    string name = hd->GetName();
    string value = hd->GetValueX();
    cout << name << endl;

    // get the reference and comparison 2d histograms
    TH2F *hnew_2d = (TH2F *)pc->GetNewHisto(name);
    TH2F *href2d = (TH2F *)pc->GetRefHisto(name);

    // set this histograms label in the allResults histo
    h2dResults.GetXaxis()->SetBinLabel(number, value.c_str());

    // ignore if histogram is broken
    if (hnew_2d->GetEntries() <= 1 || href2d->GetEntries() <= 1) {
      cout << "problem with histogram " << name << endl; continue;
    }

    // create 1d histograms to put projection results into
    string titleP = "h1dResults_passed_" + name;
    string titleF = "h1dResults_failed_" + name;
    string titleN = "h1dResults_noTest_" + name;
    
    Int_t nbins, xmin, xmax = 0;
    
   
    if( type == 2 ){	nbins = 171; xmin = -85; xmax = 86;      }
    if( type == 4 ){	nbins = 50; xmin = 1 ; xmax = 50;      }
    TH1F h1dResults_passed(titleP.c_str(),titleP.c_str(),nbins,xmin,xmax);
    TH1F h1dResults_failed(titleF.c_str(),titleF.c_str(),nbins,xmin,xmax);    
    TH1F h1dResults_noTest(titleN.c_str(),titleN.c_str(),nbins,xmin,xmax);

    // create a subdirectory in which to put resultant histograms
    TString dir=name.c_str();dir.Remove(0,1);
    gSystem->mkdir(dir);
    float running_result_ir = 0; int counted_bins_ir = 0;
    float running_result_ieta = 0; int counted_bins_ieta = 0;
    // loop over ieta
    if( type ==2 ){
    
      for (int bin = 1; bin <= href2d->GetNbinsX(); bin++) {
	
	int ieta = bin - 86;
	stringstream bin_label;
	bin_label << ieta;
	
	// ignore the center bin
	if (bin == 86) continue;
	
	  // unique names should _NOT_ be used (ProjectionY will re-alloc memory
	  // each time... But... Root is buggy and won't reliably display)....
	stringstream hname_new, hname_ref;
	hname_new << "tmp_hnew__" << name << "_" << bin;
	hname_ref << "tmp_href_" << name << "_" << bin;
	
	// get the 1d projection for this ieta bin out of the histogram
	TH1D *hnew_ = hnew_2d->ProjectionY(hname_new.str().c_str(),bin,bin);
	TH1D *href = href2d->ProjectionY(hname_ref.str().c_str(),bin,bin);
	
	// ignore empty bins
	if (hnew_->Integral() == 0 || href->Integral() == 0) { 
	  cout << "  ieta = " << ieta << ": empty!" << endl;
	  h1dResults_noTest.SetBinContent(bin,1);
	    continue;
	} else { empty[index] = false; }
	  
	float ks_score, chi2_score, result;
	  
	// calculate and set range and number of bins
	double h1RMS =  hnew_->GetRMS();
	double h2RMS =  href->GetRMS();
	double RMS = TMath::Max(h1RMS, h2RMS);
	double h1Mean =  hnew_->GetMean();
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
	if ((dX * dNdX)>0) {rebinning = (int)(double(Nbins) / (dX * dNdX));}
	// histograms of this type should be rescaled and rebinned
	href->GetXaxis()->SetRangeUser(NewMin, NewMax);
	hnew_->GetXaxis()->SetRangeUser(NewMin, NewMax);
	cout << "Projected Rebinning is (eta) : " << rebinning<< endl;
	if (rebinning > 1) {
	  href->Rebin(rebinning);
	  hnew_->Rebin(rebinning);
	}
	

	//href->Rebin();
	//hnew_->Rebin();
	
	href->GetXaxis()->SetTitle(value.c_str());
	href->GetYaxis()->SetTitle("Entries");
	href->GetYaxis()->SetTitleOffset(1.5);
//	href->SetNormFactor(hnew_->GetEntries());
	href->SetNormFactor(Nevents_new);
	hnew_->SetNormFactor(Nevents_new);	
	
	ks_score = hnew_->KolmogorovTest(href);
	chi2_score = hnew_->Chi2Test(href );
	result = (ks_score>chi2_score) ? ks_score : chi2_score;
	running_result_ieta += result; counted_bins_ieta++;
	
	if (result > high_score[index]) { high_score[index] = result; }
	if (result < low_score[index]) { low_score[index] = result; }

	href->SetNormFactor(Nevents_new);	
//	href->SetNormFactor(hnew_->GetEntries());
	
	// ensure that the peaks of both histograms will be shown by making a dummy histogram
	float Nentries_ref = href->GetEntries();
	float Nentries_new = hnew_->GetEntries();
	float XaxisMin_ref = 0, XaxisMax_ref = 0, YaxisMin_ref = 0, YaxisMax_ref = 0;
	float XaxisMin_new = 0, XaxisMax_new = 0, YaxisMin_new = 0, YaxisMax_new = 0;
	if (Nentries_ref>0) YaxisMax_ref = (href->GetMaximum()+TMath::Sqrt(href->GetMaximum()))*(Nentries_new/Nentries_ref);
	if (Nentries_new>0) YaxisMax_new = (hnew_->GetMaximum()+TMath::Sqrt(hnew_->GetMaximum()));
	
	XaxisMin_ref = href->GetXaxis()->GetXmin()>NewMin  ? href->GetXaxis()->GetXmin() : NewMin;
	XaxisMax_ref = href->GetXaxis()->GetXmax()<=NewMax ? href->GetXaxis()->GetXmax() : NewMax;
	YaxisMax_ref = (YaxisMax_ref>=YaxisMax_new) ? YaxisMax_ref : YaxisMax_new;
	
	if (TMath::Abs(XaxisMin_ref - XaxisMax_ref)<1E-6)
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
		
	// set drawing options on the new histogram
	hnew_->SetStats(0);
	hnew_->SetLineWidth(2);
	hnew_->SetFillStyle(3001);
	
	// set drawing options on the dummy histogram
	hdumb->SetStats(0);
	TString title_=value.c_str(); title_+=" [i#eta = "; title_+=ieta; title_+="]";
	hdumb->GetXaxis()->SetTitle(title_);
	hdumb->GetXaxis()->SetTitleSize(0.04);
	hdumb->GetXaxis()->SetLabelSize(0.03);
	hdumb->GetYaxis()->SetTitle("Entries");
	hdumb->GetYaxis()->SetTitleOffset(1.5);
	hdumb->GetYaxis()->SetLabelSize(0.03);
	  
	stringstream ss_title;
	ss_title.precision(5);
	if (ks_score>chi2_score)
	  ss_title << "KS Score = " << ks_score;
	else
	  ss_title << "Chi^2 Score = " << chi2_score;
	TText canvas_title(0.15,0.97,ss_title.str().c_str());
	
	if (result <= threshold) {
	  
	  // make this histogram red to denote failure
	  hnew_->SetFillColor(kRed);
	  hnew_->SetLineColor(206);
	  hnew_->SetMarkerColor(206);
	  
	  // mark the sample as being 'not-compatible'
	  testFailed[index] = true;
	  combinedFailed = true;
	    
	  // set the summary bin to failed (only need to set titles for passed h1dResults)
	  h1dResults_failed.SetBinContent(bin,result);
	  
	  // set the canvas title
	  canvas_title.SetTextColor(kRed);
	  
	} else {
	  
	  // make this histogram green to denote passing
	  hnew_->SetFillColor(kGreen);
	  hnew_->SetLineColor(103);
	  hnew_->SetMarkerColor(103);
	  
	  // set the summary bin to passed
	  //h1dResults_passed.GetXaxis()->SetBinLabel(bin, bin_label.str().c_str());
	  h1dResults_passed.SetBinContent(bin,result);
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
	histo_p.SetLogy(1);
	hdumb->Draw();
	href->Draw("SAME");
	hnew_->Draw("SAME");
	hnew_->Draw("E1SAME");
	
    stringstream legend_new;
    stringstream legend_ref;
    legend_new << argv[4] << ": " << Nentries_new << " entries";
    legend_ref << argv[5] << ": " << Nentries_ref << " entries";		
	
    TLegend l1(0.15,0.01,0.33, 0.09);
    l1.SetTextSize(0.022);
    l1.AddEntry(hnew_, legend_new.str().c_str(),"lF");
    l1.AddEntry(href, legend_ref.str().c_str(),"lF");
    l1.SetFillColor(kNone);      
    l1.Draw("SAME");

	
	// print the result to gif
	
	
	stringstream histo_name;
	histo_name << dir << "/Compare_ietaBin";
	if (bin < 10) histo_name << "00";
	else if (bin < 100) histo_name << "0";
	histo_name << bin << ".gif";
	histo_c.Print(histo_name.str().c_str(),"gif");
	
	// report the obtained KS score
	cout << "  ieta = " << ieta << ": result = " << result << endl; 
      
      // Clear objects to avoid memory leaks
	//      hnew_->Clear(); href->Clear(); hdumb->Clear();
      } // end loop over ieta bins
    } // end if (type==2) ...
    
    
    // loop over ir bins
   
    if( type == 4 )
      {

	for (int ir = 1; ir <=href2d->GetNbinsX()  ; ir++) {
	  stringstream bin_label;
	  bin_label << ir;
	  
	  // unique names should _NOT_ be used (ProjectionY will re-alloc memory
	  // each time... But... Root is buggy and won't reliably display)....
	  stringstream hname_new, hname_ref;
	  hname_new << "tmp_hnew__" << name << "_" << ir;
	  hname_ref << "tmp_href_" << name << "_" << ir;
	  
	  // get the 1d projection for this ieta bin out of the histogram
	  TH1D *hnew_ = hnew_2d->ProjectionY(hname_new.str().c_str(),ir,ir);
	  TH1D *href = href2d->ProjectionY(hname_ref.str().c_str(),ir,ir);
	  
	  // ignore empty bins
	  if (hnew_->Integral() == 0 || href->Integral() == 0) { 
	    cout << "  ir = " << ir << ": empty!" << endl;
	    h1dResults_noTest.SetBinContent(ir,1);
	    continue;
	  } else { empty[index] = false; }
	  
	  float ks_score, chi2_score, result;
	  
	  // calculate and set range and number of bins
	  double h1RMS =  hnew_->GetRMS();
	  double h2RMS =  href->GetRMS();
	  double RMS = TMath::Max(h1RMS, h2RMS);
	  double h1Mean =  hnew_->GetMean();
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
	  if ((dX * dNdX)>0)rebinning = (int)(double(Nbins) / (dX * dNdX));
	  
	  href->GetXaxis()->SetRangeUser(NewMin, NewMax);
	  hnew_->GetXaxis()->SetRangeUser(NewMin, NewMax);
	  cout << "Projected Rebinning is : " << rebinning<< endl;
	  if (rebinning > 1) {
	    href->Rebin(rebinning);
	    hnew_->Rebin(rebinning);
	  }
	  
	  href->GetXaxis()->SetTitle(value.c_str());
	  href->GetYaxis()->SetTitle("Entries");
	  href->GetYaxis()->SetTitleOffset(1.5);
	  href->SetNormFactor(hnew_->GetEntries());
//	  href->SetNormFactor(Nevents_new);
//	  hnew_->SetNormFactor(Nevents_new);
	  
	  ks_score = hnew_->KolmogorovTest(href);
	  chi2_score = hnew_->Chi2Test(href );
	  result = (ks_score>chi2_score) ? ks_score : chi2_score;
	  running_result_ir += result; counted_bins_ir++;
	  
	  if (result > high_score[index]) { high_score[index] = result; }
	  if (result < low_score[index]) { low_score[index] = result; }
	  
	  href->SetNormFactor(hnew_->GetEntries());
//	  href->SetNormFactor(Nevents_new);

	  
	  // ensure that the peaks of both histograms will be shown by making a dummy histogram
	  float Nentries_ref = href->GetEntries();
	  float Nentries_new = hnew_->GetEntries();
	  float XaxisMin_ref = 0, XaxisMax_ref = 0, YaxisMin_ref = 0, YaxisMax_ref = 0;
	  float XaxisMin_new = 0, XaxisMax_new = 0, YaxisMin_new = 0, YaxisMax_new = 0;
	  if (Nentries_ref>0) YaxisMax_ref = (href->GetMaximum()+TMath::Sqrt(href->GetMaximum()))*(Nentries_new/Nentries_ref);
	  if (Nentries_new>0) YaxisMax_new = (hnew_->GetMaximum()+TMath::Sqrt(hnew_->GetMaximum()));
	  
	  XaxisMin_ref = href->GetXaxis()->GetXmin()>NewMin  ? href->GetXaxis()->GetXmin() : NewMin;
	  XaxisMax_ref = href->GetXaxis()->GetXmax()<=NewMax ? href->GetXaxis()->GetXmax() : NewMax;
	  YaxisMax_ref = (YaxisMax_ref>=YaxisMax_new) ? YaxisMax_ref : YaxisMax_new;
	  
	  if (TMath::Abs(XaxisMin_ref - XaxisMax_ref)<1E-6)
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
	  
	  // set drawing options on the new histogram
	  hnew_->SetStats(0);
	  hnew_->SetLineWidth(2);
	  hnew_->SetFillStyle(3001);
	  
	  // set drawing options on the dummy histogram
	  hdumb->SetStats(0);
	  TString title_=value.c_str(); title_+=" (GeV)  ir = "; title_+=ir;
	  hdumb->GetXaxis()->SetTitle(title_);
	  hdumb->GetXaxis()->SetTitleSize(0.04);
	  hdumb->GetXaxis()->SetLabelSize(0.03);
	  hdumb->GetYaxis()->SetTitle("Entries");
	  hdumb->GetYaxis()->SetTitleOffset(1.5);
	  hdumb->GetYaxis()->SetLabelSize(0.03);
	  stringstream ss_title;
	  ss_title.precision(5);
	  if (ks_score>chi2_score)  ss_title << "KS Score = " << ks_score;
	  else
	    {
	      ss_title << "Chi^2 Score = " << chi2_score;
	    }
	  TText canvas_title(0.1,0.97,ss_title.str().c_str());
	      
	 
	  if (result <= threshold)
	    {
	      // make this histogram red to denote failure
	      hnew_->SetFillColor(kRed);
	      hnew_->SetLineColor(206);
	      hnew_->SetMarkerColor(206);
	      
	      // mark the sample as being 'not-compatible'
	      testFailed[index] = true;
	      combinedFailed = true;
	      
	      // set the summary bin to failed (only need to set titles for passed h1dResults)
	      h1dResults_failed.SetBinContent(ir,result);
	      
	      // set the canvas title
	      canvas_title.SetTextColor(kRed);
	      
	    } 
	  else 
	    {
	      // make this histogram green to denote passing
	      hnew_->SetFillColor(kGreen);
	      hnew_->SetLineColor(103);
	      hnew_->SetMarkerColor(103);
	      
	      // set the summary bin to passed
	      h1dResults_passed.SetBinContent(ir,result);
	    
	      
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
	  histo_p.SetLogy(1);
	  hdumb->Draw();
	  href->Draw("SAME");
	  hnew_->Draw("SAME");
	  hnew_->Draw("E1SAME");

    stringstream legend_new;
    stringstream legend_ref;
    legend_new << argv[4] << ": " << Nentries_new << " entries";
    legend_ref << argv[5] << ": " << Nentries_ref << " entries";		
	
    TLegend l1(0.15,0.01,0.33, 0.09);
    l1.SetTextSize(0.022);
    l1.AddEntry(hnew_, legend_new.str().c_str(),"lF");
    l1.AddEntry(href, legend_ref.str().c_str(),"lF");
    l1.SetFillColor(kNone);      
    l1.Draw("SAME");

	  
	  // print the result to gif
	  
	  stringstream histo_name;
	  histo_name << dir << "/Compare_ir_";
	  if (ir < 10) histo_name << "00";
	  else if (ir < 100) histo_name << "0";
	  histo_name << ir << ".gif";
	  histo_c.Print(histo_name.str().c_str(),"gif");
	  
	  // report the obtained KS score
	  cout << "  ir = " << ir << ": result = " << result << endl; 

	  // Clear Objects to avoid memory leaks
	  //	  hnew_->Clear(); href->Clear(); hdumb->Clear();
	  
	}// end loop over ir bins
      }
    


    
    // create ieta & ir summary canvas
    
	TCanvas ieta_c("ieta_c","ieta_c",1300,500);
	TPad ieta_p("ieta_p","ieta_p",0,0,1,1);
	if( type == 2 ) {
	 
	  ieta_p.SetLogy(1);
	  ieta_p.SetFrameFillColor(10);
	  ieta_c.cd();ieta_p.Draw();
	  TText ieta_title(.01, .01, "");
	  ieta_c.Draw();ieta_p.cd();ieta_title.Draw("SAME");
	}
	
    // 
	TCanvas ir_c("ir_c","ir_c",1000,500);
	TPad ir_p("ir_p","ir_p",0,0,1,1);
	if(type == 4 ){
	  ir_p.SetLogy(1);
	  ir_p.SetFrameFillColor(10);
	  ir_c.cd(); ir_p.Draw();
	  TText ir_title(.01, .01, "");
	  ir_c.Draw(); ir_p.cd();ir_title.Draw("SAME");
	}


    // setup the passing test bars
    if( type == 2 || type ==4 )
      {
	h1dResults_passed.SetStats(0);

	if( type == 2){
	  h1dResults_passed.GetXaxis()->SetTitle("i#eta Ring");
	  h1dResults_passed.GetXaxis()->SetLabelSize(0.05);
	  h1dResults_passed.SetBarWidth(0.5);
	  h1dResults_failed.SetBarWidth(0.5);
	  h1dResults_failed.SetBarOffset(0.1);
	  h1dResults_noTest.SetBarWidth(0.7);
	}

	if( type == 4) {
	  h1dResults_passed.GetXaxis()->SetTitle("ir Ring"); 
	  h1dResults_passed.GetXaxis()->SetLabelSize(0.05);
	  h1dResults_passed.SetBarWidth(0.7);
	  h1dResults_failed.SetBarWidth(0.7);
	  h1dResults_failed.SetBarOffset(0.1); 
	  h1dResults_noTest.SetBarWidth(0.7);
	}

	h1dResults_passed.GetYaxis()->SetTitle("Compatibility");
	h1dResults_passed.GetYaxis()->SetTitleOffset(0.7);
	h1dResults_passed.GetYaxis()->SetLabelSize(0.03);
	//h1dResults_passed.GetXaxis()->SetLabelSize(0.01);
	
	h1dResults_passed.SetBarOffset(0.1);
	h1dResults_passed.GetYaxis()->SetRangeUser(1E-7,2);
	
	// setup the failing test bars
	h1dResults_failed.SetStats(0);
	h1dResults_failed.GetYaxis()->SetRangeUser(1E-7,2);
	
	h1dResults_noTest.SetStats(0);

	h1dResults_noTest.SetBarOffset(0.0);
	h1dResults_noTest.SetFillColor(kBlack);
	h1dResults_noTest.SetFillStyle(3004);
	h1dResults_noTest.GetYaxis()->SetRangeUser(1E-7,2);
	
	if (empty[index]) {
	  //do nothing ...
	} else {
	  h1dResults_passed.SetFillColor(kGreen);
	  h1dResults_passed.SetLineColor(kGreen);
	  h1dResults_failed.SetFillColor(kRed);
	  h1dResults_failed.SetLineColor(kRed);
	  if(type==2) ieta_p.cd();
	  if(type==4) ir_p.cd();
	  cout << "DRAWING A TYPE " << type << " summary" << endl;
	  h1dResults_passed.Draw("barSAME");
	  h1dResults_failed.Draw("barSAME");
	  h1dResults_noTest.Draw("barSAME");
	}
	    

	// draw the pass/fail threshold line
	TLine l(xmin, threshold, xmax, threshold);
	l.SetLineColor(kRed);
	l.SetLineWidth(2);
	l.SetLineStyle(2);
	l.Draw("SAME");
	
	// print the results
	

	if( type == 2){
	  ieta_c.Update();
	  gSystem->cd(dir);
	  TString ieta_name ="Results_vs_etaRing.gif";
	  // TString ieta_name = dir; dir += "/Results_vs_etaRing.gif";
	  //ieta_c.Print("Results_vs_etaRing.gif","gif");
	  ieta_c.SaveAs("Results_vs_etaRing.gif","gif");
	  gSystem->cd("..");
	}
	if( type == 4){
	  ir_c.Update();
	  TString ir_name = "Results_vs_radius.gif";
	  gSystem->cd(dir);
	  
	  //TString ir_name = dir; dir += "/Results_vs_radius.gif";
	  ir_c.Print(ir_name, "gif");
	  gSystem->cd("..");
	}
	// report the result of the tests, if performed
	cout << "result is: ";
       	if (testFailed[index]) cout << "fail";
	else if (empty[index]) cout << "no data";
	else cout << "pass";
	float mean_ieta = counted_bins_ieta > 0 ? running_result_ieta / (double)counted_bins_ieta : 0;
	cout << ", mean KS score = " << mean_ieta << endl;
	
      }
	
      
    // Clear Objects to avoid mem leaks
    //    h1dResults_passed.Clear(); h1dResults_failed.Clear();h1dResults_noTest.Clear();
    cout << "@@@@@@@@ DONE PRINTING HISTOS AND CLEARING OBJECTS FROM MEMORY " << endl;

  }// end loop over 2d histograms
  
  // create summary canvas
  int summary_height = int(780 * float(num_histos) / 22); // 780;
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
  
  // this histogram should be empty -- filling only with the TBox objects
  h2dResults.SetStats(0);
  h2dResults.SetBarWidth(0.45);
  h2dResults.SetBarOffset(0.1);
  h2dResults.GetYaxis()->SetRangeUser(1E-7,2);
  h2dResults.GetYaxis()->SetTitle("Compatibility");
  h2dResults.SetBinContent(1,0.5);
  h2dResults.SetLabelSize(0.08*22.0/(1+float(num_histos)));
  h2dResults.GetYaxis()->SetLabelSize(0.04*22.0/(1.0+float(num_histos)));
  h2dResults.GetXaxis()->SetLabelSize(0.05*22.0/(1.0+float(num_histos))); 
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
    if (fabs(box_min - box_max) < 0.01) {
      
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
      box.SetLineWidth(1);
      box.SetLineStyle(1);
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
  main_c.Update();


    main_c.Print("AllResults-HistoCheck.gif","gif");
  
  if (combinedFailed) cout << "Final Result: fail" << endl;
  else cout << "Final Result: pass" << endl;
  
  
  
  //delete pc;
  return 0;
  

}

