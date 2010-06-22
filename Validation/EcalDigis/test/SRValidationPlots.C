#include <sstream>
#include <TH1>
#include <TH1F>

#include "ecalGrid.C"

using namespace std;

void fixcolz(){
   //to leave room for colz axis
   gPad->SetMargin(0.1,0.15,0.1,0.1);
 }

/** Macro to plot histograms produce by the EcalSelectiveReadoutValidation package
 * @param inputfile path to the input ROOT file containing the histograms
 * @param output file format: see TH1::SaveAs. e.g. .gif, .ps
 * @param mode switch specifying part of the ECAL to include in the map drawing
 * Format is: 1000*eep+100*ebp+10*ebm+eem, where eep is 1 if EE+ endcap must be
 * included, 0 otherwise, etc.
 */
void SRValidationPlots(TString inputfile = "srvalid_hists.root",
                       const char* ext = ".gif",
                       int mode = 1111){
  Plot p;
  p.run(inputfile, ext, mode);
}

struct Plot{

  bool sim = false;
  bool rec = false;
  bool barrel; //set in ctor from mode
  bool endcap;//set in ctor from mode
  bool autoLog  = false;
  
  static TH1* h1Dummy;
  
  int mode = 1111;
  
  int digit(int num, int idigit){
    for(int i = 0; i< idigit; ++i) num /= 10;
    return num  % 10;
  }

  void run(TString inputfile = "EcalSelectiveReadoutValidationHistos.root",
           const char* ext = ".gif",
           int mode_ = 1111){

    barrel = digit(mode, 1) || digit(mode, 2);

    endcap = digit(mode, 0) || digit(mode, 3);

    mode = mode_;
    gROOT ->Reset();

    gStyle->SetOptStat(111100);
    gStyle->SetStatFontSize(.025);
  
    gSystem->Exec("mkdir plots");

    TH1* h1Dummy = new TH1F("dummy","dummy", 100, 0, 1);
  
    TCanvas* c = new TCanvas();
  
    char*  rfilename = inputfile;

    delete gROOT->GetListOfFiles()->FindObject(rfilename);

    TFile * rfile = new TFile(rfilename);
    if(!rfile->cd("/DQMData/EcalDigisV/EcalDigiTask")){
      if(!rfile->cd("/DQMData/EcalDigiTask")){
        cout << "DQM plot directory not found in file " << rfilename << endl;
        if(!rfile->cd("/DQMData")){
          return;
        }
      }
    }
  
    plotAndSave("h2ChOcc", ext, "colz", false, false, 3);//("hChOcc", ext, "colz");
    plotAndSave("hEbEMean", ext);
    if(sim) plotAndSave("hEbNoZsRecVsSimE", ext, "colz");
    plotAndSave("hEbNoise", ext, "", true);
    if(rec) plotAndSave("hEbRecE", ext, "", true);
    if(rec) plotAndSave("hEbRecEHitXtal", ext);
    if(rec && sim) plotAndSave("hEbRecVsSimE", ext, "colz");
    if(sim) plotAndSave("hEbSimE", ext);
    plotAndSave("hEeEMean", ext);
    if(rec && sim) plotAndSave("hEeNoZsRecVsSimE", ext, "colz");
    if(sim) plotAndSave("hEeNoise", ext);
    if(rec) plotAndSave("hEeRecE", ext, "colz");
    if(rec) plotAndSave("hEeRecEHitXtal", ext);
    if(sim) plotAndSave("hEeRecVsSimE", ext, "colz");
    if(sim) plotAndSave("hEeSimE", ext);
    plotAndSave("hTp", ext, "", true);
    plotAndSave("h2TpVsEtSum", ext, "colz", false, false);
    //    plotAndSave("hTtFlag", ext, "", true);
    plotAndSave("h2TtfVsEtSum", ext, "colz", false, true);
    plotAndSave("h2TtfVsTp", ext, "colz", false, true);

    const char* dccVolHists[] = { "hDccVolFromData", "hDccVol", "hDccLiVol", "hDccHiVol" };

    for(unsigned i = 0; i < sizeof(dccVolHists)/sizeof(dccVolHists[0]); ++i){
      setAxisRange(hist(dccVolHists[i]), .5, 9.5, 27.5, 44.5, 54.5);
    }
    plotAndSave("hDccVolFromData", ext); //("vol");
    plotAndSave("hDccVol", ext);
    plotAndSave("hDccLiVol", ext);
    plotAndSave("hDccHiVol", ext);
    

    plotAndSave("hVol", ext);   //("volB");
    plotAndSave("hVolBHI", ext);//("volBHI");
    plotAndSave("hVolBLI", ext); //("volBLI");
    plotAndSave("hVolE", ext);//("volE");
    plotAndSave("hVolEHI", ext);//("volEHI");
    plotAndSave("hVolELI", ext);//("volELI");
    plotAndSave("hVolHI", ext);//("volHI");
    plotAndSave("hVolLI", ext);//("volLI");
  
    plotAndSave("h2Zs1Ru", ext, "colz", false, false, 1);

    plotAndSave("h2FRORu", ext, "colz", false, true, 1);
    plotAndSave("h2HiTtf", ext, "colz", false, true, 2);
    plotAndSave("h2LiTtf", ext, "colz", false, true, 2);
    plotAndSave("h2MiTtf", ext, "colz", false, true, 2);
    plotAndSave("h2ForcedTtf", ext, "colz", false, false, 2);
    plotAndSave("h2Tp", ext, "colz", false, true, 2);
    plotAndSave("hTtf", ext, "", true);


    plotAndSave("hCompleteZSMap", ext, "colz", false, true, 1);
    hist("hCompleteZsCnt")->GetXaxis()->SetRangeUser(0,20);
    plotAndSave("hCompleteZsCnt", ext);
    hist("hDroppedFROCnt")->GetXaxis()->SetRangeUser(0,10);
    plotAndSave("hDroppedFROCnt", ext);
    plotAndSave("hDroppedFROMap", ext, "colz", false, true, 1);
    plotAndSave("hDroppedFRORateMap", ext, "colz", false, true, 1);
    plotAndSave("hEbEMean", ext);
    plotAndSave("hEbFROCnt", ext);
    plotAndSave("hEbZsErrCnt", ext);
    plotAndSave("hEbZsErrType1Cnt", ext);
    plotAndSave("hEeFROCnt", ext, "", true);
    plotAndSave("hEeZsErrCnt", ext);
    plotAndSave("hEeZsErrType1Cnt", ext);
    plotAndSave("hFROCnt", ext);
    plotAndSave("hIncompleteFROCnt", ext);
    plotAndSave("hIncompleteFROMap", ext, "colz", false, true, 1);
    plotAndSave("hSRAlgoErrorMap", ext, "colz", false, false, 1);
    hist("hZsErrCnt")->GetXaxis()->SetRangeUser(0.,200.);
    plotAndSave("hZsErrCnt", ext);
    hist("hZsErrType1Cnt")->GetXaxis()->SetRangeUser(0.,20.);
    plotAndSave("hZsErrType1Cnt", ext);
    plotAndSave("zsEbHiFIRemu", ext);
    plotAndSave("zsEbLiFIRemu", ext);
    plotAndSave("zsEeHiFIRemu", ext);
    plotAndSave("zsEeLiFIRemu", ext);
  }


  TH1* hist(const char* name){
    TH1* h = 0;
    gDirectory->GetObject(name, h);
    if(h==0){
      h = h1Dummy;
      cout << "Histo " << name << " was not found!" << endl;
    }
    return h;
  }

  bool plotAndSave(const char* name, const char* fileExt = ".gif",
                   const char* opt = "", bool logy = false,
                   bool logz = false, int map=0){

    if(!barrel 
       && (strncmp(name, "hEb", 3) == 0 || strncmp(name, "zsEb", 4) ==0)) return false;

    if(!endcap 
       && (strncmp(name, "hEe", 3) == 0 || strncmp(name, "zsEe", 4) ==0)) return false;


    int optStat = gStyle->GetOptStat();
    double optFontSize = gStyle->GetStatFontSize();

    if(strcmp(opt, "colz")==0){
      fixcolz();
      gStyle->SetOptStat(110000);
      gStyle->SetStatFontSize(0.02);                       
    }
    
  
    TH1* h = 0;
    gDirectory->GetObject(name, h);
    if(h==0){
      cout << "Histogram " << name << " not found!" << endl;
      return false;
    }

    if(autoLog){
      int useLog;
      if((h->GetMaximum() > 100*h->GetMinimum(0))){
//          && (h->GetNbinsY() > 1
//              || (h->GetNBinsX() > 1
//                  && h->GetRMS() < h->GetXaxis()->GetBinWidth(1)))){
        useLog = 1;
      } else{
        useLog = 0;
      }
      if(h->GetNbinsY()>1){//2D histo
        gPad->SetLogy(0);
        gPad->SetLogz(useLog);
      } else{//assumes 1D histo
        gPad->SetLogy(useLog);
      }
    } else{
      gPad->SetLogy(logy?1:0);
      gPad->SetLogz(logz?1:0);
    }

    h->Draw(opt);

    if(map==1){
      setAxisRange(h, -40, -20.5, 0, 20.5, 40, 20);
      ruGrid(mode);
    }
    if(map==2){
      setAxisRange(h, -28.5, -17.5, 0, 17.5, 28.5, 72);
      tccGrid(mode);
    }
    if(map==3){
      setAxisRange(h, -200, -100, 0, 100, 200, 100);
      xtalGrid(mode);
    }
  
    stringstream s;
    s << "plots/" << name << fileExt;
    gPad->SaveAs(s.str().c_str());

    gStyle->SetOptStat(optStat);
    gStyle->SetStatFontSize(optFontSize);
  }

  void setAxisRange(TH1* h,
                    double xmin,
                    double eemEbm,
                    double ebmEbp,
                    double eepEem,
                    double xmax,
                    double yEeMax = -1.){
    double lb = xmin;
    double ub = xmax;
    if(!(mode & 1000)) lb = eemEbm;
    if(!(mode & 1100)) lb = ebmEbp;
    if(!(mode & 1110)) lb = eepEem;
    if(!(mode & 1))    ub = eepEem;
    if(!(mode & 11))   ub = ebmEbp;
    if(!(mode & 111))  ub = eemEbm;

    //to fix
    if(mode==110){ lb = eemEbm; }
    
    if(ub!=lb){
      h->GetXaxis()->SetRangeUser(lb, ub);
    }
    if(!(mode & 110) && yEeMax > 0 && h->GetDimension()>1){
      h->GetYaxis()->SetRangeUser(h->GetYaxis()->GetXmin(),
                                  yEeMax);
    }
  }
};

