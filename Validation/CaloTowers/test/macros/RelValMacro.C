#include "TFile.h"
#include "TString.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TPaveText.h"
#include "TPaveStats.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLegend.h"
#include "TKey.h"
#include "TClass.h"

#include <iostream>
#include <map>
#include <string>
#include <cstring>
#include <cstdio>

#include <fstream>

#include "rootlogon.h"

#include <Python.h>
#include <boost/python.hpp>
#include <vector>

template<class T1, class T2>
void prn(T1 s1, T2 s2) 
{
    std::cout << "\t>> " << s1 << ": " << s2 << std::endl;
}

void RelValMacro(std::string seriesOfTubes);
void ProcessRelVal(TFile *ref_file, TFile *val_file, std::string ref_vers, std::string val_vers, std::string histName, std::string outLabel, int nRebin, double xAxisMin, double xAxisMax, double yAxisMin, double yAxisMax,
                   std::string dimSwitch, std::string statSwitch, std::string chi2Switch, std::string logSwitch, std::string ratioFlag, int refCol, int valCol, std::string xAxisTitle, std::string normFlag, std::string histName2 = "");
template<class T>
void setObjProps(T obj);

class DirectoryFinder
{
private:
    std::map<std::string, TDirectory*> ptdMap;
    TDirectory* findDirectory( TDirectory *target, std::string& s, int dig = 2);
public:
    TDirectory* operator()(TDirectory *target, std::string& s);
} dfRef, dfVal;


void RelValMacro(std::string seriesOfTubes)
{
    //Split the string passed from the python driver
    std::stringstream ss(seriesOfTubes);
    std::string item;
    std::vector<std::string> props;
    while (getline(ss, item, '|')) {
        props.push_back(item);
    }
    std::string ref_vers = props[0];
    std::string val_vers = props[1];
    std::string rfname = props[2];
    std::string vfname = props[3];
    std::string histName = props[4];
    std::string ofileName = props[5];
    int nRebin = std::stoi(props[6]);
    double xAxisMin = std::stod(props[7]);
    double xAxisMax = std::stod(props[8]);
    double yAxisMin = std::stod(props[9]);
    double yAxisMax = std::stod(props[10]);
    std::string dimFlag = props[11];
    std::string statFlag = props[12];
    std::string chi2Flag = props[13];
    std::string logFlag = props[14];
    std::string ratioFlag = props[15];
    int refCol = std::stoi(props[16]);
    int valCol = std::stoi(props[17]);
    std::string xAxisTitle = props[18];
    std::string histName2 = props[19];
    std::string normFlag = props[20];

    if(strcmp(histName.c_str(),"HcalDigiTask/HcalDigiTask_signal_amplitude_HE") == 0) {

      std::cout<<"=================="<<std::endl;
      std::cout<<xAxisMin<< "  "<<xAxisMax <<std::endl;
    }
    //Warning!!! This rootlogon hacks the root color pallate.  This should probably be rewritten.
    setColors();
    
	TFile* Ref_File = new TFile(rfname.c_str());
	TFile* Val_File = new TFile(vfname.c_str());
    
        
	if(Ref_File && Val_File)
	{
        
        if(histName2 == "none") histName2 = "";
        
        //Make plot
        ProcessRelVal(Ref_File, Val_File, ref_vers, val_vers, histName, ofileName, nRebin, xAxisMin, xAxisMax, yAxisMin, yAxisMax, dimFlag, statFlag, chi2Flag, logFlag, ratioFlag, refCol, valCol, xAxisTitle, histName2, normFlag);
	}
	else
	{
	    if(!Ref_File) std::cout << "Input root file \"" << rfname << "\" not found!!!" << std::endl;
	    if(!Val_File) std::cout << "Input root file \"" << vfname << "\" not found!!!" << std::endl;
	}


//    ProcessSubDetCT(Ref_File, Val_File, RelValStream, CT_nHist1, CT_nHist2, CT_nProf, CT_nHistTot, ref_vers, val_vers, harvest);

    return;
}

void ProcessRelVal(TFile *ref_file, TFile *val_file, std::string ref_vers, std::string val_vers, std::string histName, std::string outLabel, int nRebin, double xAxisMin, double xAxisMax, double yAxisMin, double yAxisMax,
                   std::string dimSwitch, std::string statSwitch, std::string chi2Switch, std::string logSwitch, std::string ratioFlag, int refCol, int valCol, std::string xAxisTitle, std::string histName2, std::string normFlag)
{
    std::string NormHist = "HcalRecHitTask/N_HB";

    //split directory off histName 
    int slashLoc = histName.rfind("/");
    std::string histDir = histName.substr(0, slashLoc);
    if(slashLoc < histName.size() - 1) histName = histName.substr(slashLoc + 1, histName.size());

    int slashLocN = NormHist.rfind("/");
    std::string histDirN = NormHist.substr(0, slashLocN);
    if(slashLocN < NormHist.size() - 1) NormHist = NormHist.substr(slashLocN + 1, NormHist.size());

    std::cout << "Processing \"" << histDir << "/" << histName << "\"" << std::endl;

    //Get objects from TFiles
    TDirectory *refTD = dfRef(ref_file, histDir);
    TObject *refObj = 0;

    TDirectory *refTDN = dfRef(ref_file, histDirN);
    TObject *refObjN = 0;
    
    if(refTD) 
    {
        refObj = refTD->Get(histName.c_str());
        if(refObj) refObj = refObj->Clone();
    }
    else 
    {
	std::cout << "Cannot find directory \"" << histDir << "\" in file \"" << ref_file->GetName() << "\"" << std::endl;
	return;
    }
    if(!refObj)
    {
	std::cout << "Cannot find histogram \"" << histDir << "/" << histName << "\" in file \"" << ref_file->GetName() << "\"" << std::endl;
	return;
    }

    if(refTDN) 
    {
        refObjN = refTDN->Get(NormHist.c_str());
        if(refObjN) refObjN = refObjN->Clone();
    }
    else 
    {
	std::cout << "Cannot find directory \"" << histDirN << "\" in file \"" << ref_file->GetName() << "\"" << std::endl;
    }
    if(!refObjN)
    {
	std::cout << "Cannot find histogram \"" << histDirN << "/" << NormHist << "\" in file \"" << ref_file->GetName() << "\"" << std::endl;
    }

    TDirectory *valTD = dfVal(val_file, histDir);
    TObject *valObj = 0;
    TDirectory *valTDN = dfVal(val_file, histDirN);
    TObject *valObjN = 0;
    if(valTD) 
    {
        valObj = valTD->Get(histName.c_str());
        if(valObj) valObj = valObj->Clone();
    }
    else
    {
	std::cout << "Cannot find directory \"" << histDir << "\" in file \"" << val_file->GetName() << "\"" << std::endl;
	return;
    }
    if(!valObj)
    {
	std::cout << "Cannot find histogram \"" << histDir << "/" << histName << "\" in file \"" << val_file->GetName() << "\"" << std::endl;
	return;
    }

    if(valTDN) 
    {
        valObjN = valTDN->Get(NormHist.c_str());
        if(valObjN) valObjN = valObjN->Clone();
    }
    else
    {
	std::cout << "Cannot find directory \"" << histDirN << "\" in file \"" << val_file->GetName() << "\"" << std::endl;
    }
    if(!valObjN)
    {
	std::cout << "Cannot find histogram \"" << histDirN << "/" << NormHist << "\" in file \"" << val_file->GetName() << "\"" << std::endl;
    }

    //Try to continue processing even if N_HB is missing
    //We only care if the ratio flag is set
    //If we can't find any way to normalize the plots, unset the ratioflag
    if(std::stoi(ratioFlag) == 1){
    if(!refTDN && !valTDN)
       {
           std::cout << "Cannot find directory \"" << histDirN << "\" in either file \"" << std::endl;
           ratioFlag = "0";
       }

       if(!refObjN && !valObjN)
       {
           std::cout << "Cannot find histogram \"" << histDirN << "/" << NormHist << "\" in either file \"" << std::endl;
           ratioFlag = "0";
       }
       else if(!valObjN)
       {
           valObjN = refObjN->Clone();
           std::cout << "Using histogram \"" << NormHist << "from file \"" << ref_file->GetName() << std::endl;
       }
       else if(!refObjN)
       {
           refObjN = valObjN->Clone();
           std::cout << "Using histogram \"" << NormHist << "from file \"" << val_file->GetName() << std::endl;
       }
    }// Make sure we can normalize ratio plots

    std::cout << "Loaded \"" << histDir << "/" << histName << "\"" << std::endl;

    //Format canvas
    TCanvas *myc = 0;
    if (dimSwitch.compare("PRwide") == 0) {
        gStyle->SetPadLeftMargin(0.06);
        gStyle->SetPadRightMargin(0.03);
        myc = new TCanvas("myc", "", 1200, 600);
    } else myc = new TCanvas("myc", "", 800, 600);
//    gStyle->SetOptStat(0);    
    myc->SetGrid();
   
    TPad *pad1, *pad2;

// Ratio Flag

    float nRef =1, nVal = 1;
   
    std::cout << "Ratio Flag: " << std::stoi(ratioFlag) << std::endl;

    if(std::stoi(ratioFlag) == 1) {
   
        std::cout << "Histogram will include ratio" << std::endl;

	TH1* refN_HB = (TH1*)refObjN;
	TH1* valN_HB = (TH1*)valObjN;

	nRef = refN_HB->Integral();
	nVal = valN_HB->Integral();

        // Divide canvas into two pads
        //    myc->Divide(1,2,0,0);
        pad1 = new TPad("pad1","pad1", 0.0, 0.3, 1.0, 1.0, 0);
        pad1->SetBottomMargin(1); // Upper and lower plots are joined (0) or separate (1)
        pad1->SetGridx();         // Vertical grid
	pad1->SetFillColor(kCyan-10); //spandey
	pad2 = new TPad("pad2","pad2", 0.0, 0.03, 1.0, 0.3, 0);  //spandey updated pad size
        pad2->SetTopMargin(0);
        pad2->SetBottomMargin(0.2);
        pad2->SetGridx(); // vertical grid
        pad2->SetGridy(); // horizontal grid
	pad2->SetFillColor(kCyan-10); //spandey

        pad1->Draw();
        pad2->Draw();
    
    //    float pad2width = pad2->GetWw();
    //    float pad2height = pad2->GetWh() * pad2->GetAbsHNCD();
    //    float x2pixels = 10;
    //    float y2pixels = 10;
    //    float x2size = x2pixels / pad2width;
    //    float y2size = y2pixels / pad2height;
    
        //Format pads
        //    myc->cd(1);
        //    pad1->cd();
        if(logSwitch.compare("Log") == 0 && dimSwitch.compare("2D") == 0)
        {
            pad1->SetLogy(0);
            pad1->SetLogz(1);
        }
        else if(logSwitch.compare("Log") == 0)
        {
            pad1->SetLogy(1);
        }
//        pad2->cd();
        pad2->SetGridy();
        
//        pad1->cd();
        
    }
        
        
    std::string xTitleCheck = xAxisTitle;
    xTitleCheck = xTitleCheck.substr(1, 7);
    
    if (dimSwitch.compare("1D") == 0) 
    {
        //Get histograms from objects
        TH1* ref_hist1 = (TH1*)refObj;
        TH1* val_hist1 = (TH1*)valObj;

        // change what is embedded in DQM histos
        setObjProps(ref_hist1);
        setObjProps(val_hist1);

        //Rebin histograms -- has to be done first
        if (nRebin != 1) {
            ref_hist1->Rebin(nRebin);
            val_hist1->Rebin(nRebin);
        }

        TH1* ratio_hist1;
        
        // Ratio Flag
        if(std::stoi(ratioFlag) == 1){
	    //Let's normalize the val plot to have the same number of events as the ref plot
	    //But only if normFlag isn't tripped
	    if(normFlag.compare("Norm") == 0)
	    val_hist1->Scale(nRef/nVal);
        
            //Create Copies (Clones) to use in Ratio Plot
            TH1* ref_hist1_clone = (TH1*)ref_hist1->Clone("ref_hist1_clone");
            TH1* val_hist1_clone = (TH1*)val_hist1->Clone("val_hist1_clone");
            
            //Prepare clones for correct uncertainties
            ref_hist1_clone->Sumw2();
            val_hist1_clone->Sumw2();
            
            // Normalize (scale = n_ref/n_val)
            //float n_ref = ref_hist1_clone->Integral();
            //float n_val = val_hist1_clone->Integral();
            //float scale = n_ref/n_val;
            //val_hist1_clone->Scale(scale);
            
            //Create ratio histogram (val - ref)/ref
            ratio_hist1 = (TH1*)val_hist1_clone;
            ratio_hist1->Sumw2();
            ratio_hist1->Add(ref_hist1_clone,-1.);
            ratio_hist1->Divide(ref_hist1_clone);
            
//            //Format Ratio Plot
//            float pad2width = pad2->GetWw();
//            float pad2height = pad2->GetWh() * pad2->GetAbsHNDC();
//            float x2pixels = 100;
//            float y2pixels = 15;
//            float x2size = x2pixels / pad2width;
//            float y2size = y2pixels / pad2height;
//    
//            TAxis* x2axis = ratio_hist1->GetXaxis();
//            TAxis* y2axis = ratio_hist1->GetYaxis();
//    
//            x2axis->SetTitleOffset(2);
//            x2axis->SetTitleSize(0.15);
//            x2axis->SetLabelSize(x2size);
//            
//            y2axis->SetTitleOffset(0.3);
//            y2axis->SetTitleSize(0.12);
//            y2axis->SetRangeUser(0,2.5);
//            y2axis->SetLabelSize(y2size);

// Sanitizing axis inputs    
            //Min/Max Convetion: Default AxisMin = 0. Default AxisMax = -1.
            //xAxis
            if (xAxisMin == 0) xAxisMin = ref_hist1->GetXaxis()->GetXmin();
            if (xAxisMax < 0) xAxisMax = ref_hist1->GetXaxis()->GetXmax();
	
	    //Sanitize xAxis inputs
            if (xAxisMin < ref_hist1->GetXaxis()->GetXmin()) xAxisMin = ref_hist1->GetXaxis()->GetXmin();
	    if (xAxisMax > ref_hist1->GetXaxis()->GetXmax()) xAxisMax = ref_hist1->GetXaxis()->GetXmax();

            ratio_hist1->SetTitle("");
            ratio_hist1->SetLineStyle(1);
            ratio_hist1->SetMarkerStyle(1);
            ratio_hist1->SetMarkerSize(0.02);
            
            //Format Ratio Plot
	    //lets get schwifty
            float pad2width = pad2->GetWw();
            float pad2height = pad2->GetWh() * pad2->GetAbsHNDC() ;
            float x2pixels = 100;
            float y2pixels = 15;
            float x2size = x2pixels / pad2width;
            float y2size = y2pixels / pad2height;
            
            TAxis* x2axis = ratio_hist1->GetXaxis();
            TAxis* y2axis = ratio_hist1->GetYaxis();
            

	    x2axis->SetTitleOffset(1.0); // Important for seeing x-axis title!
	    x2axis->SetTitleSize(0.1); //spandey
            x2axis->SetLabelSize(x2size*0.64);
            x2axis->SetRangeUser(xAxisMin, xAxisMax);
            
            y2axis->SetTitle("(val - ref)/ref");
            y2axis->SetTitleOffset(0.3);
            y2axis->SetTitleSize(0.12);
            //        y2axis->SetRangeUser(0,2.5);
            y2axis->SetLabelSize(y2size);
            y2axis->SetNdivisions(4);
            
            ratio_hist1->SetStats(kFALSE);

        }

        //Set the colors, styles, titles, stat boxes and format axes for the histograms
        ref_hist1->SetStats(kTRUE);
        val_hist1->SetStats(kTRUE);
        
        if (statSwitch.compare("Stat") != 0 && statSwitch.compare("Statrv") != 0) {
            ref_hist1->SetStats(kFALSE);
            val_hist1->SetStats(kFALSE);
        }
        
        
        //Min/Max Convetion: Default AxisMin = 0. Default AxisMax = -1.
        //xAxis
        if (xAxisMin == 0) xAxisMin = ref_hist1->GetXaxis()->GetXmin();
        if (xAxisMax < 0) xAxisMax = ref_hist1->GetXaxis()->GetXmax();
	
	//Sanitize xAxis inputs
	if (xAxisMin < ref_hist1->GetXaxis()->GetXmin()) xAxisMin = ref_hist1->GetXaxis()->GetXmin();
	if (xAxisMax > ref_hist1->GetXaxis()->GetXmax()) xAxisMax = ref_hist1->GetXaxis()->GetXmax();
	
        if (xAxisMax > 0 || xAxisMin != 0) {
            ref_hist1->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);
            val_hist1->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);

        }
        //yAxis
        if (yAxisMin != 0) ref_hist1->SetMinimum(yAxisMin);
        if (yAxisMax > 0) ref_hist1->SetMaximum(yAxisMax);
        else if (ref_hist1->GetMaximum() < val_hist1->GetMaximum() &&
                val_hist1->GetMaximum() > 0) {
            if (logSwitch.compare("Log") == 0) ref_hist1->SetMaximum(2 * val_hist1->GetMaximum());
            else ref_hist1->SetMaximum(1.05 * val_hist1->GetMaximum());
        }

        //Title
//        if (xTitleCheck != "NoTitle") ref_hist1->GetXaxis()->SetTitle(xAxisTitle.c_str());
        ref_hist1->GetXaxis()->SetTitle("");
        if (xTitleCheck != "NoTitle" && std::stoi(ratioFlag) == 1) ratio_hist1->GetXaxis()->SetTitle(xAxisTitle.c_str());
        if (xTitleCheck != "NoTitle" && std::stoi(ratioFlag) != 1) ref_hist1->GetXaxis()->SetTitle(xAxisTitle.c_str());

        //Different histo colors and styles
        ref_hist1->SetTitle("");
        ref_hist1->SetLineColor(refCol);
        ref_hist1->SetLineStyle(1);
        ref_hist1->SetMarkerSize(0.02);

        val_hist1->SetTitle("");
        val_hist1->SetLineColor(valCol);
        val_hist1->SetLineStyle(2);
        val_hist1->SetMarkerSize(0.02);
        if(statSwitch.compare("Stat") != 0 && statSwitch.compare("Statrv") != 0)
        {
            ref_hist1->SetLineWidth(2);
            val_hist1->SetLineWidth(2);
        }

        //Legend
        TLegend *leg = new TLegend(0.50, 0.91, 0.84, 0.99, "", "brNDC");
        leg->SetBorderSize(2);
        leg->SetFillStyle(1001);
        leg->AddEntry(ref_hist1, ("CMSSW_" + ref_vers).c_str(), "l");
        leg->AddEntry(val_hist1, ("CMSSW_" + val_vers).c_str(), "l");

        //It's time to draw (#yolo)!
        if (chi2Switch.compare("Chi2") == 0) {
            
            // Title Time
            
            //Draw and save histograms
            if(std::stoi(ratioFlag) == 1){
                pad1->cd();
            }
            ref_hist1->SetFillColor(40);//42 Originally, now 40 which is light brown
            ref_hist1->Draw("hist");
            val_hist1->SetLineStyle(1);
            if (statSwitch.compare("Statrv") == 0) val_hist1->Draw("sames e0");
            else val_hist1->Draw("same e0");
            
            // Ratio Flag
            if(std::stoi(ratioFlag) == 1){
                //Draw ratio
                pad2->cd();
	        //pad1->cd();
                ratio_hist1->Draw();
                pad1->cd();
		//pad2->cd();
            }

            //Get p-value from chi2 test
            const float NCHI2MIN = 0.01;

            float pval;
            char tempbuff[30];

            pval = ref_hist1->Chi2Test(val_hist1);

            sprintf(tempbuff, "Chi2 p-value: %6.3E", pval);

            TPaveText* ptchi2 = new TPaveText(0.05, 0.92, 0.35, 0.99, "NDC");

            if (pval > NCHI2MIN) ptchi2->SetFillColor(kGreen);
            else ptchi2->SetFillColor(kRed);

            ptchi2->SetTextSize(0.03);
            ptchi2->AddText(tempbuff);
            ptchi2->Draw();
        } else {
            
            // Title Time
            
            
            //Draw and save histograms
            if(std::stoi(ratioFlag) == 1){
                pad1->cd();
            }
            ref_hist1->Draw("hist");
            if (statSwitch.compare("Statrv") == 0) val_hist1->Draw("hist sames");
            else val_hist1->Draw("hist same");

            
            // Ratio Flag
            if(std::stoi(ratioFlag) == 1){
                //Draw ratio
                pad2->cd();
                ratio_hist1->Draw();
                pad1->cd();
            }
        }

        //Stat Box where required
        if (statSwitch.compare("Stat") == 0 || statSwitch.compare("Statrv") == 0) {
            TPaveStats* ptstats_r = new TPaveStats(0.85, 0.86, 0.98, 0.98, "brNDC");
            ptstats_r->SetTextColor(refCol);
            ref_hist1->GetListOfFunctions()->Add(ptstats_r);
            ptstats_r->SetParent(ref_hist1->GetListOfFunctions());
            TPaveStats* ptstats_v = new TPaveStats(0.85, 0.74, 0.98, 0.86, "brNDC");
            ptstats_v->SetTextColor(valCol);
            val_hist1->GetListOfFunctions()->Add(ptstats_v);
            ptstats_v->SetParent(val_hist1->GetListOfFunctions());

            ptstats_r->Draw();
            ptstats_v->Draw();
        }

        leg->Draw();

        myc->SaveAs(outLabel.c_str());
    }
    //Profiles not associated with histograms
    else if (dimSwitch.compare("PR") == 0 || dimSwitch.compare("PRwide") == 0) 
    {
        //Get profiles from objects
        TProfile* ref_prof = (TProfile*)refObj;
        TProfile* val_prof = (TProfile*)valObj;

        // HACK to change what is embedded in DQM histos
        setObjProps(ref_prof);
        setObjProps(val_prof);

        //Legend
        TLegend* leg = new TLegend(0.50, 0.91, 0.84, 0.99, "", "brNDC");
        leg->SetBorderSize(2);
        leg->SetFillStyle(1001);

        //Ordinary profiles
        if(dimSwitch.compare("PR") == 0) 
        {
            ref_prof->SetTitle("");
            ref_prof->SetErrorOption("");

            val_prof->SetTitle("");
            val_prof->SetErrorOption("");

            ref_prof->GetXaxis()->SetTitle(xAxisTitle.c_str());

            if (statSwitch.compare("Stat") != 0 && statSwitch.compare("Statrv") != 0) {
                ref_prof->SetStats(kFALSE);
                val_prof->SetStats(kFALSE);
            }

            ref_prof->SetLineColor(41);
            ref_prof->SetLineStyle(1);
            ref_prof->SetLineWidth(1);
            ref_prof->SetMarkerColor(41);
            ref_prof->SetMarkerStyle(21);
            ref_prof->SetMarkerSize(0.8);

            val_prof->SetLineColor(43);
            val_prof->SetLineStyle(1);
            val_prof->SetLineWidth(1);
            val_prof->SetMarkerColor(43);
            val_prof->SetMarkerStyle(22);
            val_prof->SetMarkerSize(1.0);

            if (ref_prof->GetMaximum() < val_prof->GetMaximum() &&
                    val_prof->GetMaximum() > 0) {
                if (logSwitch.compare("Log") == 0) ref_prof->SetMaximum(2 * val_prof->GetMaximum());
                else ref_prof->SetMaximum(1.05 * val_prof->GetMaximum());
            }

            ref_prof->Draw("hist pl");
            val_prof->Draw("hist pl same");

            leg->AddEntry(ref_prof, ("CMSSW_" + ref_vers).c_str(), "pl");
            leg->AddEntry(val_prof, ("CMSSW_" + val_vers).c_str(), "pl");
        }//Wide profiles
        else if(dimSwitch.compare("PRwide") == 0)
        {
            char temp[128];
            sprintf(temp, "%s_px_v", ref_prof->GetName());
            TH1* ref_fp = ref_prof->ProjectionX();
            TH1* val_fp = val_prof->ProjectionX(temp);

            ref_fp->SetTitle("");
            val_fp->SetTitle("");

            ref_fp->GetXaxis()->SetTitle(xAxisTitle.c_str());

            if(statSwitch.compare("Stat") != 0 && statSwitch.compare("Statrv") != 0) 
            {
                ref_fp->SetStats(kFALSE);
                val_fp->SetStats(kFALSE);
            }

            int nbins = ref_fp->GetNbinsX();
            for (int j = 1; j < nbins; j++) {
                ref_fp->SetBinError(j, 0.);
                val_fp->SetBinError(j, 0.);
            }
            ref_fp->SetLineWidth(0);
            ref_fp->SetLineColor(0); // 5 yellow
            ref_fp->SetLineStyle(1);
            ref_fp->SetMarkerColor(2);
            ref_fp->SetMarkerStyle(20);
            ref_fp->SetMarkerSize(0.5);

            val_fp->SetLineWidth(0);
            val_fp->SetLineColor(0); // 45 blue
            val_fp->SetLineStyle(2);
            val_fp->SetMarkerColor(4);
            val_fp->SetMarkerStyle(22);
            val_fp->SetMarkerSize(0.5);

            if (ref_fp->GetMaximum() < val_fp->GetMaximum() &&
                    val_fp->GetMaximum() > 0) {
                if (logSwitch.compare("Log") == 0) ref_fp->SetMaximum(2 * val_fp->GetMaximum());
                else ref_fp->SetMaximum(1.05 * val_fp->GetMaximum());
            }

            ref_fp->Draw("p9");
            val_fp->Draw("p9same");

            leg->AddEntry(ref_fp, ("CMSSW_" + ref_vers).c_str(), "lp");
            leg->AddEntry(val_fp, ("CMSSW_" + val_vers).c_str(), "lp");

        }

        leg->Draw("");

        myc->SaveAs(outLabel.c_str());
    }//Timing Histograms (special: read two lines at once)
    else if (dimSwitch.compare("TM") == 0) 
    {
        //split directory off histName 
        int slashLoc2 = histName2.rfind("/");
        std::string histDir2 = histName2.substr(0, slashLoc2);
        if(slashLoc2 < histName2.size() - 1) histName2 = histName2.substr(slashLoc2 + 1, histName2.size());

        //Get objects from TFiles
        TDirectory *refTD2 = dfRef(ref_file, histDir2);
        TObject *refObj2 = refTD->Get(histName2.c_str())->Clone();
        TDirectory *valTD2 = dfVal(val_file, histDir2);
        TObject *valObj2 = valTD->Get(histName2.c_str())->Clone();

        TH2* ref_hist2 = (TH2*)refObj;
        TProfile* ref_prof = (TProfile*)refObj2;

        ref_hist2->SetMarkerStyle(21);
        ref_prof ->SetMarkerStyle(21);
        ref_hist2->SetMarkerSize(0.02);
        ref_prof ->SetMarkerSize(0.02);

        TH2* val_hist2 = (TH2F*)valObj;
        TProfile* val_prof = (TProfile*)valObj2;

        val_hist2->SetMarkerStyle(21);
        val_prof ->SetMarkerStyle(21);
        val_hist2->SetMarkerSize(0.02);
        val_prof ->SetMarkerSize(0.02);

        // HACK to change what is embedded in DQM histos
        setObjProps(ref_hist2);
        setObjProps(val_hist2);

        //Min/Max Convention: Default AxisMin = 0. Default AxisMax = -1.
        //xAxis
        if (xAxisMin == 0) xAxisMin = ref_hist2->GetXaxis()->GetXmin();
        if (xAxisMax < 0) xAxisMax = ref_hist2->GetXaxis()->GetXmax();

        if (xAxisMax > 0 || xAxisMin != 0) {
            ref_hist2->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);
            val_hist2->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);
        }
        //yAxis
        if (yAxisMin != 0) ref_hist2->SetMinimum(yAxisMin);
        if (yAxisMax > 0) ref_hist2->SetMaximum(yAxisMax);
        else if (ref_hist2->GetMaximum() < val_hist2->GetMaximum() &&
                val_hist2->GetMaximum() > 0) {
            if (logSwitch == "Log") ref_hist2->SetMaximum(2 * val_hist2->GetMaximum());
            else ref_hist2->SetMaximum(1.05 * val_hist2->GetMaximum());
        }

        //AF
        if (yAxisMax > 0 || yAxisMin != 0) {
            ref_hist2->GetYaxis()->SetRangeUser(yAxisMin, yAxisMax);
            val_hist2->GetYaxis()->SetRangeUser(yAxisMin, yAxisMax);
        }

        //Legend
        TLegend* leg = new TLegend(0.50, 0.91, 0.84, 0.99, "", "brNDC");
        leg->SetBorderSize(2);
        leg->SetFillStyle(1001);

        ref_hist2->GetXaxis()->SetTitle(xAxisTitle.c_str());
        ref_hist2->SetStats(kFALSE);

        ref_hist2->SetTitle("");
        val_hist2->SetTitle("");

        ref_hist2->SetMarkerColor(refCol); // rose
        ref_hist2->Draw();
        ref_prof->SetLineColor(41);
        ref_prof->Draw("same");

        val_hist2->SetMarkerColor(valCol);
        val_hist2->Draw("same");
        val_prof->SetLineColor(45);
        val_prof->Draw("same");

        leg->AddEntry(ref_prof, ("CMSSW_" + ref_vers).c_str(), "pl");
	leg->AddEntry(val_prof, ("CMSSW_" + val_vers).c_str(), "pl");

        leg->Draw("");

        myc->SaveAs(outLabel.c_str());

	if(refObj2) delete refObj2;
	if(valObj2) delete valObj2;
    }
    else if(dimSwitch.compare("2D") == 0) 
    {

        myc->SetGrid(0, 0);

        //Get histograms from objects
        TH2* ref_hist2D = (TH2*)refObj;
        TH2* val_hist2D = (TH2*)valObj;

        ref_hist2D->SetStats(kFALSE);
        val_hist2D->SetStats(kFALSE);

        // HACK to change what is embedded in DQM histos
        setObjProps(ref_hist2D);
        setObjProps(val_hist2D);

        ref_hist2D->SetTitle("");
        val_hist2D->SetTitle("");

        // special zoom on HB/HE depth1
        //if (n2D == 1) {
        //    ref_hist2D->GetXaxis()->SetRangeUser(-29., 28.);
        //    val_hist2D->GetXaxis()->SetRangeUser(-29., 28.);
        //}

        //Min/Max Convetion: Default AxisMin = 0. Default AxisMax = -1.
        //xAxis
        if (xAxisMax > 0 || xAxisMin != 0) {
            ref_hist2D->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);
            val_hist2D->GetXaxis()->SetRangeUser(xAxisMin, xAxisMax);
        }
        //yAxis
        if (yAxisMax > 0 || yAxisMin != 0) {
            ref_hist2D->GetYaxis()->SetRangeUser(yAxisMin, yAxisMax);
            val_hist2D->GetYaxis()->SetRangeUser(yAxisMin, yAxisMax);
        }
        //Set bin minimum to 0
        ref_hist2D->SetMinimum(0.0);
        val_hist2D->SetMinimum(0.0);

        TLegend *leg1 = new TLegend(0.50, 0.91, 0.84, 0.99, "", "brNDC");
        leg1->SetBorderSize(2);
        leg1->SetFillStyle(1001);
        leg1->AddEntry(ref_hist2D, ("CMSSW_" + ref_vers).c_str(), "l");

        if (xTitleCheck != "NoTitle") ref_hist2D->GetXaxis()->SetTitle(xAxisTitle.c_str());
        ref_hist2D->Draw("colz");
        leg1->Draw();
        myc->SaveAs(("ref_" + outLabel).c_str());

        TLegend *leg2 = new TLegend(0.50, 0.91, 0.84, 0.99, "", "brNDC");
        leg2->SetBorderSize(2);
        leg2->SetFillStyle(1001);
        leg2->AddEntry(val_hist2D, ("CMSSW_" + val_vers).c_str(), "l");

        if (xTitleCheck != "NoTitle") val_hist2D->GetXaxis()->SetTitle(xAxisTitle.c_str());
        val_hist2D->Draw("colz");
        leg2->Draw();
        myc->SaveAs(("val_" + outLabel).c_str());
    }

    if(myc) delete myc;
    if(refObj) delete refObj;
    if(valObj) delete valObj;

    return;
}

TDirectory* DirectoryFinder::operator()(TDirectory *target, std::string& s)
{
    if(ptdMap.find(s) == ptdMap.end()) return (ptdMap[s] = findDirectory(target, s));
    else                               return ptdMap[s];
}

TDirectory* DirectoryFinder::findDirectory( TDirectory *target, std::string& s, int dig)
{
    TDirectory *retval = 0;

    // loop over all keys in this directory                                                                                                                                                                                                  
    TIter nextkey(target->GetListOfKeys());
    TKey *key, *oldkey=0;
    while((key = (TKey*)nextkey()))
    {

        //std::cout << "Found " << key->ReadObj()->GetName() << std::endl;

	//keep only the highest cycle number for each key                                                                                                                                                                                    
	if (oldkey && !strcmp(oldkey->GetName(),key->GetName())) continue;

	// read object from file                                                                                                                                                                                                             
//	target->cd();
	TObject *obj = key->ReadObj();
//	obj->Print();

	if(obj->IsA()->InheritsFrom(TDirectory::Class()))
	{
	    // it's a subdirectory                                                                                                                                                                                                           
	    //std::cout << "Found subdirectory " << obj->GetName() << std::endl;

	    if(strcmp(s.c_str(), obj->GetName()) == 0) return (TDirectory*)obj;

	    if((retval = findDirectory((TDirectory*)obj, s, dig-1))) break;

	} else if(dig < 1){
            break;
        }

    }

    return retval;
}

template<class T>
void setObjProps(T obj)
{
    obj->GetXaxis()->SetLabelSize(0.04);
    obj->GetYaxis()->SetLabelSize(0.04);
    obj->GetXaxis()->SetTitleSize(0.045);

    obj->GetXaxis()->SetTickLength(-0.015);
    obj->GetYaxis()->SetTickLength(-0.015);

    obj->GetXaxis()->SetLabelOffset(0.02);
    obj->GetYaxis()->SetLabelOffset(0.02);
    
    obj->GetXaxis()->SetTitleOffset(1.3);
}


BOOST_PYTHON_MODULE(RelValMacro)
{
    using namespace boost::python;
    def("RelValMacro", RelValMacro);
}

