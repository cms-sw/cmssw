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

#include "rootlogon.h"

template<class T1, class T2>
void prn(T1 s1, T2 s2) 
{
    std::cout << "\t>> " << s1 << ": " << s2 << std::endl;
}

void RelValMacro(std::string ref_vers, std::string val_vers, std::string rfname, std::string vfname, std::string inputStream = "InputRelVal.txt");
void ProcessRelVal(TFile *ref_file, TFile *val_file, std::string ref_vers, std::string val_vers, std::string histName, std::string outLabel, int nRebin, double xAxisMin, double xAxisMax, double yAxisMin, double yAxisMax,
                   std::string dimSwitch, std::string statSwitch, std::string chi2Switch, std::string logSwitch, int refCol, int valCol, std::string xAxisTitle, std::string histName2 = "");
template<class T>
void setObjProps(T obj);

class DirectoryFinder
{
private:
    std::map<std::string, TDirectory*> ptdMap;
    TDirectory* findDirectory( TDirectory *target, std::string& s);
public:
    TDirectory* operator()(TDirectory *target, std::string& s);
} dfRef, dfVal;

int main(int argn, char **argv)
{
    if(argn == 5)       RelValMacro(argv[1], argv[2], argv[3], argv[4]);
    else if(argn == 6)  RelValMacro(argv[1], argv[2], argv[3], argv[4], argv[5]);
    else
    {
	printf("Usage: ./RelValMacro.exe refVersion valVersion refFileName valFileName [input stream]\n");
    }
}

void RelValMacro(std::string ref_vers, std::string val_vers, std::string rfname, std::string vfname, std::string inputStream) 
{
    //Warning!!! This rootlogon hacks the root color pallate.  This should probably be rewritten.  
    setColors();
    
    //File Read 
    FILE * inputFile = NULL;
    if((inputFile = fopen(inputStream.c_str(), "r")))
    {
        char buff[4096];
        char *c;
	
	char histName[128], histName2[128] = "", ofileName[128], xAxisTitle[128];
	double xAxisMin, xAxisMax, yAxisMin, yAxisMax;
	char dimFlag[32], statFlag[32], chi2Flag[32], logFlag[32];
	int nRebin, draw, refCol, valCol;

	TFile* Ref_File = new TFile(rfname.c_str());
	TFile* Val_File = new TFile(vfname.c_str());

	if(Ref_File && Val_File)
	{
	    while(!feof(inputFile) && (c = fgets(buff, 4096, inputFile)) != NULL)
	    {            
		//The following lines allow for comments.  The first comment character (#) will be replaced with a end of string.
		char* k = strchr(buff, '#');
		if(k) *k = '\0';
		//Parse the line
		if(sscanf(buff, "%s %d %s %d %lf %lf %lf %lf %s %s %s %s %d %d %[^\n]", histName, &draw, ofileName, &nRebin, &xAxisMin, &xAxisMax, &yAxisMin, &yAxisMax, dimFlag, statFlag, chi2Flag, logFlag, &refCol, &valCol, xAxisTitle) == 15)
		{
		    //Skip is set not to draw
		    if(!draw) continue;
                
		    //Ugly hack for the timing plots, this should be fixed 
		    if(strcmp(dimFlag, "TM") == 0)
		    {
			fgets(buff, 4096, inputFile);
			sscanf(buff, "%s", histName2);
		    }
                
		    //Make plot
		    ProcessRelVal(Ref_File, Val_File, ref_vers, val_vers, histName, ofileName, nRebin, xAxisMin, xAxisMax, yAxisMin, yAxisMax, dimFlag, statFlag, chi2Flag, logFlag, refCol, valCol, xAxisTitle, histName2);
		}
	    }
	}
	else
	{
	    if(!Ref_File) std::cout << "Input root file \"" << rfname << "\" not found!!!" << std::endl;
	    if(!Val_File) std::cout << "Input root file \"" << vfname << "\" not found!!!" << std::endl;
	}
        fclose(inputFile);
	
	Ref_File->Close();
	Val_File->Close();
    }
    else
    {
        std::cout << "Input file \"" << inputStream << "\" not found!!!" << std::endl;
    }

//    ProcessSubDetCT(Ref_File, Val_File, RelValStream, CT_nHist1, CT_nHist2, CT_nProf, CT_nHistTot, ref_vers, val_vers, harvest);

    return;
}

void ProcessRelVal(TFile *ref_file, TFile *val_file, std::string ref_vers, std::string val_vers, std::string histName, std::string outLabel, int nRebin, double xAxisMin, double xAxisMax, double yAxisMin, double yAxisMax,
                   std::string dimSwitch, std::string statSwitch, std::string chi2Switch, std::string logSwitch, int refCol, int valCol, std::string xAxisTitle, std::string histName2)
{
    //split directory off histName 
    int slashLoc = histName.rfind("/");
    std::string histDir = histName.substr(0, slashLoc);
    if(slashLoc < histName.size() - 1) histName = histName.substr(slashLoc + 1, histName.size());

    //Get objects from TFiles
    TDirectory *refTD = dfRef(ref_file, histDir);
    TObject *refObj = 0;
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
	std::cout << "Cannot find histogram \"" << histDir << "\\" << histName << "\" in file \"" << ref_file->GetName() << "\"" << std::endl;
	return;
    }

    TDirectory *valTD = dfVal(val_file, histDir);
    TObject *valObj = 0;
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
    if(!refObj)
    {
	std::cout << "Cannot find histogram \"" << histDir << "\\" << histName << "\" in file \"" << val_file->GetName() << "\"" << std::endl;
	return;
    }

    //Format canvas
    TCanvas *myc = 0;
    if (dimSwitch.compare("PRwide") == 0) {
        gStyle->SetPadLeftMargin(0.06);
        gStyle->SetPadRightMargin(0.03);
        myc = new TCanvas("myc", "", 1200, 600);
    } else myc = new TCanvas("myc", "", 800, 600);
    myc->SetGrid();

    std::string xTitleCheck = xAxisTitle;
    xTitleCheck = xTitleCheck.substr(1, 7);

    //Format pad
    if(logSwitch.compare("Log") == 0 && dimSwitch.compare("2D") == 0)
    {
        myc->SetLogy(0);
        myc->SetLogz(1);
    }
    else if(logSwitch.compare("Log") == 0)
    {
        myc->SetLogy(1);
    }
    
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
        if (xTitleCheck != "NoTitle") ref_hist1->GetXaxis()->SetTitle(xAxisTitle.c_str());

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

        if (chi2Switch.compare("Chi2") == 0) {
            //Draw and save histograms
            ref_hist1->SetFillColor(40);//42 Originally, now 40 which is lgiht brown
            ref_hist1->Draw("hist");
            val_hist1->SetLineStyle(1);
            if (statSwitch.compare("Statrv") == 0) val_hist1->Draw("sames e0");
            else val_hist1->Draw("same e0");

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
            //Draw and save histograms
            ref_hist1->Draw("hist");
            if (statSwitch.compare("Statrv") == 0) val_hist1->Draw("hist sames");
            else val_hist1->Draw("hist same");
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

TDirectory* DirectoryFinder::findDirectory( TDirectory *target, std::string& s)
{
    TDirectory *retval = 0;

    // loop over all keys in this directory                                                                                                                                                                                                  
    TIter nextkey(target->GetListOfKeys());
    TKey *key, *oldkey=0;
    while((key = (TKey*)nextkey()))
    {
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

	    if((retval = findDirectory((TDirectory*)obj, s))) break;

	}
	else
	{
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
