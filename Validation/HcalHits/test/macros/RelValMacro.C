
#include <fstream>
#include <TFile.h>
#include <TString.h>
#include <TCanvas.h>



using namespace std;

void prn(TString s0, TString s1) {
    std::cout << "\t>> " << s0 << ": " << s1 << std::endl;
}

void prn(TString s0, double d) {
    std::cout << "\t>> " << s0 << ": " << d << std::endl;
}


void ProcessRelVal(TFile &ref_file, TFile &val_file, ifstream &recstr, const int nHist1, const int nHist2, const int nHits2D, const int nProfInd, const int nHistTot, TString ref_vers, TString val_vers, int harvest = 0, bool bRBX = false);

void RelValMacro(TString ref_vers = "218", TString val_vers = "218", TString rfname, TString vfname, TString InputStream = "InputRelVal.txt", int harvest = 0) {

  ifstream RelValStream;

  RelValStream.open(InputStream);

  TFile Ref_File(rfname);
  TFile Val_File(vfname);

/*
    A note about MC histograms: 22 of them are not included in this current implementation.
    Two things must be done to include them again:
    1. Uncomment the appropriate lines below in the service variables detailing how many
       histograms ProcessRelVal will expect from the Input txt files.
    2. Change the aforementioned Input txt files DrawSwitch flags from 0 to 1 (the second 
       number after the histogram name). 
*/


    //Service variables

    //SimHits
  const int SH_nHistTot = 7;
  const int SH_nHist1 = 2;
  const int SH_nHist2 = 0;
  const int SH_nHist2D = 5;
  const int SH_nProfInd = 0;
  
  ProcessRelVal(Ref_File, Val_File, RelValStream, SH_nHist1, SH_nHist2, SH_nHist2D, SH_nProfInd, SH_nHistTot, ref_vers, val_vers, harvest, false, true);
  
  Ref_File.Close();
  Val_File.Close();
  
  return;
}

void ProcessRelVal(TFile &ref_file, TFile &val_file, ifstream &recstr, const int nHist1, const int nHist2, const int nHist2D, const int nProfInd, const int nHistTot, TString ref_vers, TString val_vers, int harvest, bool bRBX, bool bHD = false) {
  
  TString RefHistDir, ValHistDir;
  
  if (bRBX) {
    if (harvest == 11) {
      RefHistDir = "DQMData/Run 1/NoiseRatesV/Run summary/NoiseRatesTask";
      ValHistDir = "DQMData/Run 1/NoiseRatesV/Run summary/NoiseRatesTask";
    } else if (harvest == 10) {
      RefHistDir = "DQMData/NoiseRatesV/NoiseRatesTask";
      ValHistDir = "DQMData/Run 1/NoiseRatesV/Run summary/NoiseRatesTask";
    } else if (harvest == 1) {
      RefHistDir = "DQMData/Run 1/NoiseRatesV/Run summary/NoiseRatesTask";
      ValHistDir = "DQMData/NoiseRatesV/NoiseRatesTask";
    } else {
      RefHistDir = "DQMData/NoiseRatesV/NoiseRatesTask";
      ValHistDir = "DQMData/NoiseRatesV/NoiseRatesTask";
    }
  }
  //AF
  else if (bHD) {
    if (harvest == 11) {
      RefHistDir = "DQMData/Run 2/HcalHitsV/Run summary/SimHitsValidationHcal";
      ValHistDir = "DQMData/Run 2/HcalHitsV/Run summary/SimHitsValidationHcal";
    } else if (harvest == 10) {
      RefHistDir = "DQMData/HcalDigisV/HcalDigiTask";
      ValHistDir = "DQMData/Run 1/HcalDigisV/Run summary/HcalDigiTask";
    } else if (harvest == 1) {
      RefHistDir = "DQMData/Run 1/HcalDigisV/Run summary/HcalDigiTask";
      ValHistDir = "DQMData/HcalDigisV/HcalDigiTask";
    } else {
      RefHistDir = "DQMData/HcalDigisV/HcalDigiTask";
      ValHistDir = "DQMData/HcalDigisV/HcalDigiTask";
    }
  } else {
    if (harvest == 11) {
      RefHistDir = "DQMData/Run 1/HcalRecHitsV/Run summary/HcalRecHitTask";
      ValHistDir = "DQMData/Run 1/HcalRecHitsV/Run summary/HcalRecHitTask";
    } else if (harvest == 10) {
      RefHistDir = "DQMData/HcalRecHitsV/HcalRecHitTask";
      ValHistDir = "DQMData/Run 1/HcalRecHitsV/Run summary/HcalRecHitTask";
    } else if (harvest == 1) {
      RefHistDir = "DQMData/Run 1/HcalRecHitsV/Run summary/HcalRecHitTask";
      ValHistDir = "DQMData/HcalRecHitsV/HcalRecHitTask";
    } else {
      RefHistDir = "DQMData/HcalRecHitsV/HcalRecHitTask";
      ValHistDir = "DQMData/HcalRecHitsV/HcalRecHitTask";
    }
  }
  
  TCanvas* myc = 0;
  TLegend* leg = 0;
  TPaveText* ptchi2 = 0;
  TPaveStats *ptstats_r = 0;
  TPaveStats *ptstats_v = 0;
  
  TH1F * ref_hist1[nHist1];
  TH2F * ref_hist2[nHist2];
  TProfile * ref_prof[nProfInd];
  TH1D * ref_fp[nProfInd];
  TH2F * ref_hist2D[nHist2D];
  
  TH1F * val_hist1[nHist1];
  TH2F * val_hist2[nHist2];
  TProfile * val_prof[nProfInd];
  TH1D * val_fp[nProfInd];
  TH2F * val_hist2D[nHist2D];
  
  int i;
  int DrawSwitch;
  TString StatSwitch, Chi2Switch, LogSwitch, DimSwitch;
  int RefCol, ValCol;
  TString HistName, HistName2;
  char xAxisTitle[200];
  int nRebin;
  float xAxisMin, xAxisMax, yAxisMin, yAxisMax;
  TString OutLabel, ProfileLabel;
  string xTitleCheck;
  
  float hmax = 0;

  int nh1 = 0;
  int nh2 = 0;
  int npr = 0;
  int npi = 0;
  int n2D = 0;
  
  for (i = 0; i < nHistTot; i++) {
    
    //Read in 1/0 switch saying whether this histogram is used
    //Skip it if not used, otherwise get output file label, histogram
    //axis ranges and title
    //ALTERED: Reads in all inputs and then uses 1/0 switch to skip
    //See below
    recstr >> HistName >> DrawSwitch;
    prn("HistName", HistName);
    //        if (DrawSwitch == 0) continue;
      
    recstr >> OutLabel >> nRebin;
    recstr >> xAxisMin >> xAxisMax >> yAxisMin >> yAxisMax;
    recstr >> DimSwitch >> StatSwitch >> Chi2Switch >> LogSwitch;
    recstr >> RefCol >> ValCol;
    recstr.getline(xAxisTitle, 200);
    
    //Make sure extra Profile info is also taken care of
    if (DrawSwitch == 0) {
      if (DimSwitch == "TM") 
	recstr >> ProfileLabel;
      continue;
    }
    
    
    // Nasty trick:
    // recovering colors redefined in rootlogon.C (for "rainbow" Palette)
    Float_t r, g, b;
    Float_t saturation = 1;
    Float_t lightness = 0.5;
    Float_t maxHue = 280;
    Float_t minHue = 0;
        Int_t maxPretty = 50;
        Float_t hue;

        for (int j = 0; j < maxPretty; j++) {
            hue = maxHue - (j + 1)*((maxHue - minHue) / maxPretty);
            TColor::HLStoRGB(hue, lightness, saturation, r, g, b);
            TColor *color = (TColor*) (gROOT->GetListOfColors()->At(j + 51));
            color->SetRGB(r, g, b);
         }
         gStyle->SetPalette(1);


        //Format canvas
        if (DimSwitch == "PRwide") {
            gStyle->SetPadLeftMargin(0.06);
            gStyle->SetPadRightMargin(0.03);
            myc = new TCanvas("myc", "", 1200, 600);
        } else myc = new TCanvas("myc", "", 800, 600);
        myc->SetGrid();

        xTitleCheck = xAxisTitle;
        xTitleCheck = xTitleCheck.substr(1, 7);

        //Format pad
        if (LogSwitch == "Log") myc->SetLogy(1);
        else myc->SetLogy(0);

        //AF
        if (LogSwitch == "Log" && DimSwitch == "2D"){
          myc->SetLogy(0);
          myc->SetLogz(1);
        }

        if (DimSwitch == "1D") {
	  //Get histograms from files
	  ref_file.cd(RefHistDir);
	  ref_hist1[nh1] = (TH1F*) gDirectory->Get(HistName);

            val_file.cd(ValHistDir);
            val_hist1[nh1] = (TH1F*) gDirectory->Get(HistName);

            // HACK to change what is embedded in DQM histos
            ref_hist1[nh1]->GetXaxis()->SetLabelSize(0.04);
            val_hist1[nh1]->GetXaxis()->SetLabelSize(0.04);
            ref_hist1[nh1]->GetYaxis()->SetLabelSize(0.04);
            val_hist1[nh1]->GetYaxis()->SetLabelSize(0.04);
            ref_hist1[nh1]->GetXaxis()->SetTitleSize(0.045);
            val_hist1[nh1]->GetXaxis()->SetTitleSize(0.045);

            ref_hist1[nh1]->GetXaxis()->SetTickLength(-0.015);
            val_hist1[nh1]->GetXaxis()->SetTickLength(-0.015);
            ref_hist1[nh1]->GetYaxis()->SetTickLength(-0.015);
            val_hist1[nh1]->GetYaxis()->SetTickLength(-0.015);

            ref_hist1[nh1]->GetXaxis()->SetLabelOffset(0.012);
            val_hist1[nh1]->GetXaxis()->SetLabelOffset(0.012);
            ref_hist1[nh1]->GetYaxis()->SetLabelOffset(0.012);
            val_hist1[nh1]->GetYaxis()->SetLabelOffset(0.012);

            ref_hist1[nh1]->GetXaxis()->SetTitleOffset(0.8);
            val_hist1[nh1]->GetXaxis()->SetTitleOffset(0.8);


            //Rebin histograms -- has to be done first
            if (nRebin != 1) {
                ref_hist1[nh1]->Rebin(nRebin);
                val_hist1[nh1]->Rebin(nRebin);
            }

            //Set the colors, styles, titles, stat boxes and format axes for the histograms
            ref_hist1[nh1]->SetStats(kTRUE);
            val_hist1[nh1]->SetStats(kTRUE);

            if (StatSwitch != "Stat" && StatSwitch != "Statrv") {
                ref_hist1[nh1]->SetStats(kFALSE);
                val_hist1[nh1]->SetStats(kFALSE);
            }

            //Min/Max Convetion: Default AxisMin = 0. Default AxisMax = -1.
            //xAxis
            if (xAxisMin == 0) xAxisMin = ref_hist1[nh1]->GetXaxis()->GetXmin();
            if (xAxisMax < 0) xAxisMax = ref_hist1[nh1]->GetXaxis()->GetXmax();

            if (xAxisMax > 0 || xAxisMin != 0) {
                ref_hist1[nh1]->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);
                val_hist1[nh1]->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);
            }
            //yAxis
            if (yAxisMin != 0) ref_hist1[nh1]->SetMinimum(yAxisMin);
            if (yAxisMax > 0) ref_hist1[nh1]->SetMaximum(yAxisMax);
            else if (ref_hist1[nh1]->GetMaximum() < val_hist1[nh1]->GetMaximum() &&
                    val_hist1[nh1]->GetMaximum() > 0) {
                if (LogSwitch == "Log") ref_hist1[nh1]->SetMaximum(2 * val_hist1[nh1]->GetMaximum());
                else ref_hist1[nh1]->SetMaximum(1.05 * val_hist1[nh1]->GetMaximum());
            }

            //Title
            if (xTitleCheck != "NoTitle") ref_hist1[nh1]->GetXaxis()->SetTitle(xAxisTitle);

            //Different histo colors and styles
            ref_hist1[nh1]->SetTitle("");
            ref_hist1[nh1]->SetLineColor(RefCol);
            ref_hist1[nh1]->SetLineStyle(1);
            ref_hist1[nh1]->SetMarkerSize(0.02);
            if (StatSwitch != "Stat" && StatSwitch != "Statrv") ref_hist1[nh1]->SetLineWidth(2);

            val_hist1[nh1]->SetTitle("");
            val_hist1[nh1]->SetLineColor(ValCol);
            val_hist1[nh1]->SetLineStyle(2);
            val_hist1[nh1]->SetMarkerSize(0.02);
            if (StatSwitch != "Stat" && StatSwitch != "Statrv") val_hist1[nh1]->SetLineWidth(2);

            //Legend
            TLegend *leg = new TLegend(0.50, 0.91, 0.84, 0.99, "", "brNDC");
            leg->SetBorderSize(2);
            leg->SetFillStyle(1001);
            leg->AddEntry(ref_hist1[nh1], "CMSSW_" + ref_vers, "l");
            leg->AddEntry(val_hist1[nh1], "CMSSW_" + val_vers, "l");

            if (Chi2Switch == "Chi2") {
                //Draw and save histograms
                ref_hist1[nh1]->SetFillColor(40);//42 Originally, now 40 which is lgiht brown
                ref_hist1[nh1]->Draw("hist");
                val_hist1[nh1]->SetLineStyle(1);
                if (StatSwitch == "Statrv") val_hist1[nh1]->Draw("sames e0");
                else val_hist1[nh1]->Draw("same e0");

                //Get p-value from chi2 test
                const float NCHI2MIN = 0.01;

                float pval;
                stringstream mystream;
                char tempbuff[30];

                pval = ref_hist1[nh1]->Chi2Test(val_hist1[nh1]);

                sprintf(tempbuff, "Chi2 p-value: %6.3E%c", pval, '\0');
                mystream << tempbuff;

                ptchi2 = new TPaveText(0.05, 0.92, 0.35, 0.99, "NDC");

                if (pval > NCHI2MIN) ptchi2->SetFillColor(kGreen);
                else ptchi2->SetFillColor(kRed);

                ptchi2->SetTextSize(0.03);
                ptchi2->AddText(mystream.str().c_str());
                ptchi2->Draw();
            } else {
                //Draw and save histograms
                ref_hist1[nh1]->Draw("hist");
                if (StatSwitch == "Statrv") val_hist1[nh1]->Draw("hist sames");
                else val_hist1[nh1]->Draw("hist same");
            }

            //Stat Box where required
            if (StatSwitch == "Stat" || StatSwitch == "Statrv") {
                ptstats_r = new TPaveStats(0.85, 0.86, 0.98, 0.98, "brNDC");
                ptstats_r->SetTextColor(RefCol);
                ref_hist1[nh1]->GetListOfFunctions()->Add(ptstats_r);
                ptstats_r->SetParent(ref_hist1[nh1]->GetListOfFunctions());
                ptstats_v = new TPaveStats(0.85, 0.74, 0.98, 0.86, "brNDC");
                ptstats_v->SetTextColor(ValCol);
                val_hist1[nh1]->GetListOfFunctions()->Add(ptstats_v);
                ptstats_v->SetParent(val_hist1[nh1]->GetListOfFunctions());

                ptstats_r->Draw();
                ptstats_v->Draw();
            }

            leg->Draw();

            myc->SaveAs(OutLabel);
            nh1++;
        }
            //Profiles not associated with histograms
        else if (DimSwitch == "PR" || DimSwitch == "PRwide") {
            //Get profiles from files
            ref_file.cd(RefHistDir);
            ref_prof[npi] = (TProfile*) gDirectory->Get(HistName);

            val_file.cd(ValHistDir);
            val_prof[npi] = (TProfile*) gDirectory->Get(HistName);

            // HACK to change what is embedded in DQM histos
            ref_prof[npi]->GetXaxis()->SetLabelSize(0.04);
            val_prof[npi]->GetXaxis()->SetLabelSize(0.04);
            ref_prof[npi]->GetYaxis()->SetLabelSize(0.04);
            val_prof[npi]->GetYaxis()->SetLabelSize(0.04);
            ref_prof[npi]->GetXaxis()->SetTitleSize(0.045);
            val_prof[npi]->GetXaxis()->SetTitleSize(0.045);

            ref_prof[npi]->GetXaxis()->SetTickLength(-0.015);
            val_prof[npi]->GetXaxis()->SetTickLength(-0.015);
            ref_prof[npi]->GetYaxis()->SetTickLength(-0.015);
            val_prof[npi]->GetYaxis()->SetTickLength(-0.015);

            ref_prof[npi]->GetXaxis()->SetLabelOffset(0.02);
            val_prof[npi]->GetXaxis()->SetLabelOffset(0.02);
            ref_prof[npi]->GetYaxis()->SetLabelOffset(0.02);
            val_prof[npi]->GetYaxis()->SetLabelOffset(0.02);

            ref_prof[npi]->GetXaxis()->SetTitleOffset(1.3);
            val_prof[npi]->GetXaxis()->SetTitleOffset(1.3);


            //Legend
            leg = new TLegend(0.50, 0.91, 0.84, 0.99, "", "brNDC");
            leg->SetBorderSize(2);
            leg->SetFillStyle(1001);

            //Ordinary profiles
            if (DimSwitch == "PR") {
                ref_prof[npi]->SetTitle("");
                ref_prof[npi]->SetErrorOption("");

                val_prof[npi]->SetTitle("");
                val_prof[npi]->SetErrorOption("");

                ref_prof[npi]->GetXaxis()->SetTitle(xAxisTitle);

                if (StatSwitch != "Stat" && StatSwitch != "Statrv") {
                    ref_prof[npi]->SetStats(kFALSE);
                    val_prof[npi]->SetStats(kFALSE);
                }

                ref_prof[npi]->SetLineColor(41);
                ref_prof[npi]->SetLineStyle(1);
                ref_prof[npi]->SetLineWidth(1);
                ref_prof[npi]->SetMarkerColor(41);
                ref_prof[npi]->SetMarkerStyle(21);
                ref_prof[npi]->SetMarkerSize(0.8);

                val_prof[npi]->SetLineColor(43);
                val_prof[npi]->SetLineStyle(1);
                val_prof[npi]->SetLineWidth(1);
                val_prof[npi]->SetMarkerColor(43);
                val_prof[npi]->SetMarkerStyle(22);
                val_prof[npi]->SetMarkerSize(1.0);

                if (ref_prof[npi]->GetMaximum() < val_prof[npi]->GetMaximum() &&
                        val_prof[npi]->GetMaximum() > 0) {
                    if (LogSwitch == "Log") ref_prof[npi]->SetMaximum(2 * val_prof[npi]->GetMaximum());
                    else ref_prof[npi]->SetMaximum(1.05 * val_prof[npi]->GetMaximum());
                }

                ref_prof[npi]->Draw("hist pl");
                val_prof[npi]->Draw("hist pl same");

                leg->AddEntry(ref_prof[npi], "CMSSW_" + ref_vers, "pl");
                leg->AddEntry(val_prof[npi], "CMSSW_" + val_vers, "pl");
            }//Wide profiles
            else if (DimSwitch == "PRwide") {
                TString temp = HistName + "_px_v";
                ref_fp[npi] = ref_prof[npi]->ProjectionX();
                val_fp[npi] = val_prof[npi]->ProjectionX(temp.Data());

                ref_fp[npi]->SetTitle("");
                val_fp[npi]->SetTitle("");

                ref_fp[npi]->GetXaxis()->SetTitle(xAxisTitle);

                if (StatSwitch != "Stat" && StatSwitch != "Statrv") {
                    ref_fp[npi]->SetStats(kFALSE);
                    val_fp[npi]->SetStats(kFALSE);
                }

                int nbins = ref_fp[npi]->GetNbinsX();
                for (int j = 1; j < nbins; j++) {
                    ref_fp[npi]->SetBinError(j, 0.);
                    val_fp[npi]->SetBinError(j, 0.);
                }
                ref_fp[npi]->SetLineWidth(0);
                ref_fp[npi]->SetLineColor(0); // 5 yellow
                ref_fp[npi]->SetLineStyle(1);
                ref_fp[npi]->SetMarkerColor(2);
                ref_fp[npi]->SetMarkerStyle(20);
                ref_fp[npi]->SetMarkerSize(0.5);

                val_fp[npi]->SetLineWidth(0);
                val_fp[npi]->SetLineColor(0); // 45 blue
                val_fp[npi]->SetLineStyle(2);
                val_fp[npi]->SetMarkerColor(4);
                val_fp[npi]->SetMarkerStyle(22);
                val_fp[npi]->SetMarkerSize(0.5);

                if (ref_fp[npi]->GetMaximum() < val_fp[npi]->GetMaximum() &&
                        val_fp[npi]->GetMaximum() > 0) {
                    if (LogSwitch == "Log") ref_fp[npi]->SetMaximum(2 * val_fp[npi]->GetMaximum());
                    else ref_fp[npi]->SetMaximum(1.05 * val_fp[npi]->GetMaximum());
                }

                ref_fp[npi]->Draw("p9");
                val_fp[npi]->Draw("p9same");

                leg->AddEntry(ref_fp[npi], "CMSSW_" + ref_vers, "lp");
                leg->AddEntry(val_fp[npi], "CMSSW_" + val_vers, "lp");

            }

            leg->Draw("");

            myc->SaveAs(OutLabel);

            npi++;
        }//Timing Histograms (special: read two lines at once)
        else if (DimSwitch == "TM") {

            recstr >> HistName2;

            ref_file.cd(RefHistDir);

            ref_hist2[nh2] = (TH2F*) gDirectory->Get(HistName);
            ref_prof[npi] = (TProfile*) gDirectory->Get(HistName2);

            ref_hist2[nh2]->SetMarkerStyle(21);
            ref_prof[npi] ->SetMarkerStyle(21);
            ref_hist2[nh2]->SetMarkerSize(0.02);
            ref_prof[npi] ->SetMarkerSize(0.02);

            val_file.cd(ValHistDir);

            val_hist2[nh2] = (TH2F*) gDirectory->Get(HistName);
            val_prof[npi] = (TProfile*) gDirectory->Get(HistName2);

            val_hist2[nh2]->SetMarkerStyle(21);
            val_prof[npi] ->SetMarkerStyle(21);
            val_hist2[nh2]->SetMarkerSize(0.02);
            val_prof[npi] ->SetMarkerSize(0.02);

            // HACK to change what is embedded in DQM histos
            ref_hist2[nh2]->GetXaxis()->SetLabelSize(0.04);
            val_hist2[nh2]->GetXaxis()->SetLabelSize(0.04);
            ref_hist2[nh2]->GetYaxis()->SetLabelSize(0.04);
            val_hist2[nh2]->GetYaxis()->SetLabelSize(0.04);
            ref_hist2[nh2]->GetXaxis()->SetTitleSize(0.045);
            val_hist2[nh2]->GetXaxis()->SetTitleSize(0.045);

            ref_hist2[nh2]->GetXaxis()->SetTickLength(-0.015);
            val_hist2[nh2]->GetXaxis()->SetTickLength(-0.015);
            ref_hist2[nh2]->GetYaxis()->SetTickLength(-0.015);
            val_hist2[nh2]->GetYaxis()->SetTickLength(-0.015);

            ref_hist2[nh2]->GetXaxis()->SetLabelOffset(0.02);
            val_hist2[nh2]->GetXaxis()->SetLabelOffset(0.02);
            ref_hist2[nh2]->GetYaxis()->SetLabelOffset(0.02);
            val_hist2[nh2]->GetYaxis()->SetLabelOffset(0.02);

            ref_hist2[nh2]->GetXaxis()->SetTitleOffset(1.3);
            val_hist2[nh2]->GetXaxis()->SetTitleOffset(1.3);


            //Min/Max Convetion: Default AxisMin = 0. Default AxisMax = -1.
            //xAxis
            if (xAxisMin == 0) xAxisMin = ref_hist2[nh2]->GetXaxis()->GetXmin();
            if (xAxisMax < 0) xAxisMax = ref_hist2[nh2]->GetXaxis()->GetXmax();

            if (xAxisMax > 0 || xAxisMin != 0) {
                ref_hist2[nh2]->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);
                val_hist2[nh2]->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);
            }
            //yAxis
            if (yAxisMin != 0) ref_hist2[nh2]->SetMinimum(yAxisMin);
            if (yAxisMax > 0) ref_hist2[nh2]->SetMaximum(yAxisMax);
            else if (ref_hist2[nh2]->GetMaximum() < val_hist2[nh2]->GetMaximum() &&
                    val_hist2[nh2]->GetMaximum() > 0) {
                if (LogSwitch == "Log") ref_hist2[nh2]->SetMaximum(2 * val_hist2[nh2]->GetMaximum());
                else ref_hist2[nh2]->SetMaximum(1.05 * val_hist2[nh2]->GetMaximum());
            }

            //AF
            if (yAxisMax > 0 || yAxisMin != 0) {
                ref_hist2[nh2]->GetYaxis()->SetRangeUser(yAxisMin, yAxisMax);
                val_hist2[nh2]->GetYaxis()->SetRangeUser(yAxisMin, yAxisMax);
            }

            //Legend
            leg = new TLegend(0.50, 0.91, 0.84, 0.99, "", "brNDC");
            leg->SetBorderSize(2);
            leg->SetFillStyle(1001);

            ref_hist2[nh2]->GetXaxis()->SetTitle(xAxisTitle);
            ref_hist2[nh2]->SetStats(kFALSE);

	    ref_hist2[nh2]->SetTitle("");
            val_hist2[nh2]->SetTitle("");

            ref_hist2[nh2]->SetMarkerColor(RefCol); // rose
            ref_hist2[nh2]->Draw();
            ref_prof[npi]->SetLineColor(41);
            ref_prof[npi]->Draw("same");

            val_hist2[nh2]->SetMarkerColor(ValCol);
            val_hist2[nh2]->Draw("same");
            val_prof[npi]->SetLineColor(45);
            val_prof[npi]->Draw("same");

            leg->AddEntry(ref_prof[npi], "CMSSW_" + ref_vers, "pl");
            leg->AddEntry(val_prof[npi], "CMSSW_" + val_vers, "pl");

            leg->Draw("");

            myc->SaveAs(OutLabel);

            npi++;
            nh2++;
            i++;
        } else if (DimSwitch == "2D") {

            myc->SetGrid(0, 0);

            //Get histograms from files
            ref_file.cd(RefHistDir);
            ref_hist2D[n2D] = (TH2F*) gDirectory->Get(HistName);

            val_file.cd(ValHistDir);
            val_hist2D[n2D] = (TH2F*) gDirectory->Get(HistName);

            ref_hist2D[n2D]->SetStats(kFALSE);
            val_hist2D[n2D]->SetStats(kFALSE);

            // HACK to change what is embedded in DQM histos
            ref_hist2D[n2D]->GetXaxis()->SetLabelSize(0.04);
            val_hist2D[n2D]->GetXaxis()->SetLabelSize(0.04);
            ref_hist2D[n2D]->GetYaxis()->SetLabelSize(0.04);
            val_hist2D[n2D]->GetYaxis()->SetLabelSize(0.04);
            ref_hist2D[n2D]->GetXaxis()->SetTitleSize(0.045);
            val_hist2D[n2D]->GetXaxis()->SetTitleSize(0.045);

            ref_hist2D[n2D]->GetXaxis()->SetTickLength(-0.015);
            val_hist2D[n2D]->GetXaxis()->SetTickLength(-0.015);
            ref_hist2D[n2D]->GetYaxis()->SetTickLength(-0.015);
            val_hist2D[n2D]->GetYaxis()->SetTickLength(-0.015);

            ref_hist2D[n2D]->GetXaxis()->SetLabelOffset(0.02);
            val_hist2D[n2D]->GetXaxis()->SetLabelOffset(0.02);
            ref_hist2D[n2D]->GetYaxis()->SetLabelOffset(0.02);
            val_hist2D[n2D]->GetYaxis()->SetLabelOffset(0.02);

            ref_hist2D[n2D]->GetXaxis()->SetTitleOffset(1.0);
            val_hist2D[n2D]->GetXaxis()->SetTitleOffset(1.0);

	    ref_hist2D[n2D]->SetTitle("");
	    val_hist2D[n2D]->SetTitle("");

            // special zoom on HB/HE depth1
            if (n2D == 1) {
                ref_hist2D[n2D]->GetXaxis()->SetRangeUser(-29., 28.);
                val_hist2D[n2D]->GetXaxis()->SetRangeUser(-29., 28.);
            }

            //AF
            //Min/Max Convetion: Default AxisMin = 0. Default AxisMax = -1.
            //xAxis
            if (xAxisMax > 0 || xAxisMin != 0) {
                ref_hist2D[n2D]->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);
                val_hist2D[n2D]->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);
            }
            //yAxis
            if (yAxisMax > 0 || yAxisMin != 0) {
                ref_hist2D[n2D]->GetYaxis()->SetRangeUser(yAxisMin, yAxisMax);
                val_hist2D[n2D]->GetYaxis()->SetRangeUser(yAxisMin, yAxisMax);
            }

            TLegend *leg1 = new TLegend(0.50, 0.91, 0.84, 0.99, "", "brNDC");
            leg1->SetBorderSize(2);
            leg1->SetFillStyle(1001);
            leg1->AddEntry(ref_hist2D[n2D], "CMSSW_" + ref_vers, "l");

            if (xTitleCheck != "NoTitle") ref_hist2D[n2D]->GetXaxis()->SetTitle(xAxisTitle);
            ref_hist2D[n2D]->Draw("colz");
            leg1->Draw();
            myc->SaveAs("ref_" + OutLabel);


            TLegend *leg2 = new TLegend(0.50, 0.91, 0.84, 0.99, "", "brNDC");
            leg2->SetBorderSize(2);
            leg2->SetFillStyle(1001);
            leg2->AddEntry(val_hist2D[n2D], "CMSSW_" + val_vers, "l");

            if (xTitleCheck != "NoTitle") val_hist2D[n2D]->GetXaxis()->SetTitle(xAxisTitle);
            val_hist2D[n2D]->Draw("colz");
            leg2->Draw();
            myc->SaveAs("val_" + OutLabel);

            n2D++;
        }


        if (myc) delete myc;
        if (leg) delete leg;
        if (ptchi2) delete ptchi2;
        if (ptstats_r) delete ptstats_r;
        if (ptstats_v) delete ptstats_v;
    }
    return;
}
