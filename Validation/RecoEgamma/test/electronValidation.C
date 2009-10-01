{

TString DBS_SAMPLE = gSystem->Getenv("DBS_SAMPLE") ;

TString val_ref_file_name = gSystem->Getenv("VAL_REF_FILE") ;
TString val_new_file_name = gSystem->Getenv("VAL_NEW_FILE") ;
TString val_ref_release = gSystem->Getenv("VAL_REF_RELEASE") ;
TString val_new_release = gSystem->Getenv("VAL_NEW_RELEASE") ;
TString val_analyzer = gSystem->Getenv("VAL_ANALYZER") ;

TString val_web = gSystem->Getenv("VAL_WEB") ;
TString val_web_sub_dir = gSystem->Getenv("VAL_WEB_SUB_DIR") ;
TString val_web_url = gSystem->Getenv("VAL_WEB_URL") ;

std::string val_web_path = val_web+"/"+val_new_release+"/Electrons/vs"+val_ref_release+"/"+val_web_sub_dir ;
std::string histos_path = val_web_path+"/histos.txt" ;
std::string index_path = val_web_path+"/index.html" ;

// style:
TStyle *eleStyle = new TStyle("eleStyle","Style for electron validation");
eleStyle->SetCanvasBorderMode(0);
eleStyle->SetCanvasColor(kWhite);
eleStyle->SetCanvasDefH(600);
eleStyle->SetCanvasDefW(800);
eleStyle->SetCanvasDefX(0);
eleStyle->SetCanvasDefY(0);
eleStyle->SetPadBorderMode(0);
eleStyle->SetPadColor(kWhite);
eleStyle->SetPadGridX(false);
eleStyle->SetPadGridY(false);
eleStyle->SetGridColor(0);
eleStyle->SetGridStyle(3);
eleStyle->SetGridWidth(1);
eleStyle->SetOptStat(1);
eleStyle->SetPadTickX(1);
eleStyle->SetPadTickY(1);
eleStyle->SetHistLineColor(1);
eleStyle->SetHistLineStyle(0);
eleStyle->SetHistLineWidth(2);
eleStyle->SetEndErrorSize(2);
eleStyle->SetErrorX(0.);
eleStyle->SetOptStat(1);
eleStyle->SetTitleColor(1, "XYZ");
eleStyle->SetTitleFont(42, "XYZ");
eleStyle->SetTitleXOffset(1.0);
eleStyle->SetTitleYOffset(1.0);
eleStyle->SetLabelOffset(0.005, "XYZ");
eleStyle->SetTitleSize(0.05, "XYZ");
eleStyle->SetTitleFont(22,"X");
eleStyle->SetTitleFont(22,"Y");
eleStyle->SetHistLineWidth(2);
eleStyle->SetPadBottomMargin(0.13);
eleStyle->SetPadLeftMargin(0.15);
eleStyle->SetMarkerStyle(21);
eleStyle->SetMarkerSize(0.8);

eleStyle->cd();

gROOT->ForceStyle();

TString internal_path("DQMData/EgammaV/") ;
internal_path += val_analyzer+"/" ;

TString val_ref_file_url ;
TString file_ref_dir = internal_path ;
TFile * file_ref = 0 ;
if ( val_ref_file_name != "" )
 {
  file_ref = TFile::Open(val_ref_file_name) ;
  if (file_ref!=0)
   {
    std::cout<<"open "<<val_ref_file_name<<std::endl ;
    if (val_ref_file_name.BeginsWith(val_web)==kTRUE)
     {
      val_ref_file_url = val_ref_file_name ;
      val_ref_file_url.Remove(0,val_web.Length()) ;
      val_ref_file_url.Prepend(val_web_url) ;
     }
    if (file_ref->cd(internal_path)!=kTRUE)
     {
      std::cerr<<"Failed move to: "<<internal_path<<std::endl ;
      file_ref_dir = "" ;
     }
    else
     {
      std::cerr<<"cd "<<internal_path<<std::endl ;
      file_ref->cd() ;
     }
   }
  else
   { std::cerr<<"Failed to open: "<<val_ref_file_name<<std::endl ; }
 }

TString val_new_file_url ;
TString file_new_dir = internal_path  ;
TFile * file_new = 0 ;
if ( val_new_file_name != "" )
 {
  file_new = TFile::Open(val_new_file_name) ;
  if (file_new!=0)
   {
    std::cout<<"open "<<val_new_file_name<<std::endl ;
    if (val_new_file_name.BeginsWith(val_web)==kTRUE)
     {
      val_new_file_url = val_new_file_name ;
      val_new_file_url.Remove(0,val_web.Length()) ;
      val_new_file_url.Prepend(val_web_url) ;
     }
    if (file_new->cd(internal_path)!=kTRUE)
     {
      std::cerr<<"Failed move to: "<<internal_path<<std::endl ;
      file_new_dir = "" ;
     }
    else
     {
      std::cerr<<"cd "<<internal_path<<std::endl ;
      file_new->cd() ;
     }
   }
  else
   { std::cerr<<"Failed to open: "<<val_new_file_name<<std::endl ; }
 }

TCanvas * canvas ;
TH1 * histo_ref, * histo_new ;

std::ofstream web_page(index_path.c_str()) ;

web_page
  <<"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n"
  <<"<html>\n"
  <<"<head>\n"
  <<"<meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\" />\n"
  <<"<title>"<<val_new_release<<" vs "<<val_ref_release<<" / "<<DBS_SAMPLE<<"</title>\n"
  <<"</head>\n"
  <<"<h1><a href=\"../\">"<<val_new_release<<" vs "<<val_ref_release<<"</a> / "<<DBS_SAMPLE<<"</h1>\n" ;

web_page<<"<p>" ;
if (file_ref==0)
 {
  web_page
	 <<"In all plots below"
	 <<", there was no "<<val_ref_release<<" histograms to compare with"
     <<", and the <a href=\""<<val_new_file_url<<"\">"<<val_new_release<<" histograms</a> are in red"
	 <<"." ;
 }
else
 {
  web_page
	 <<"In all plots below"
     <<", the <a href=\""<<val_new_file_url<<"\">"<<val_new_release<<" histograms</a> are in red"
	 <<", and the <a href=\""<<val_ref_file_url<<"\">"<<val_ref_release<<" histograms</a> are in blue"
	 <<"." ;
 }
web_page
  <<" They were made using analyzer "
  <<"<a href=\"http://cmslxr.fnal.gov/lxr/source/Validation/RecoEgamma/interface/"<<val_analyzer<<".h\">"
  <<"Validation/RecoEgamma/interface/"<<val_analyzer<<".h"
  <<"</a> and configuration "
  <<"<a href=\"http://cmslxr.fnal.gov/lxr/source/Validation/RecoEgamma/test/"<<val_analyzer<<"_cfg.py\">"
  <<"Validation/RecoEgamma/test/"<<val_analyzer<<"_cfg.py"
  <<"</a>, with dataset "<<DBS_SAMPLE<<" as input." ;
web_page
  <<" Some more details"
  <<": <a href=\"electronValidation.C\">script</a> used to make the plots"
  <<", <a href=\"histos.txt\">specification</a> of histograms"
  <<", <a href=\"gifs/\">images</a> of histograms"
  <<"." ;
web_page<<"</p>\n" ;

std::string histo_name, gif_name, gif_path, canvas_name ;
TString short_histo_name ;
int scaled, log, err ;
int divide;
std::string num, denom, cat ;
int eol ; // end of line
int eoc ; // enf of category

std::ifstream histo_file1(histos_path.c_str()) ;
web_page
  <<"<br><table border=\"1\" cellpadding=\"5\" width=\"100%\">"
  <<"<tr valign=\"top\"><td width=\"20%\">\n" ;
int cat_num = 0 ;

cat = "" ;
do
 {
  std::getline(histo_file1,cat) ;
 } while (cat.empty()) ;

web_page<<"<b>"<<cat<<"</b><br><br>" ;

while (histo_file1>>histo_name>>scaled>>log>>err>>divide>>num>>denom>>eol>>eoc)
 {
  short_histo_name = histo_name ;
  short_histo_name.Remove(0,2) ;
  web_page<<"<a href=\"#"<<short_histo_name<<"\">"<<short_histo_name<<"</a><br>\n" ;
  if (eoc)
   {
    cat_num++ ;
    if ((cat_num%5)==0)
     { web_page<<"<br></td></tr>\n<tr valign=\"top\"><td width=\"20%\">" ; }
    else
     { web_page<<"<br></td><td width=\"20%\">\n" ; }
    cat = "" ;
    do
     {
      std::getline(histo_file1,cat) ;
     } while (cat.empty()) ;
    web_page<<"<b>"<<cat<<"</b><br><br>" ;
   }
 }
web_page<<"<br></td></tr></table>\n" ;
histo_file1.close() ;

web_page<<"<br><br><table cellpadding=\"5\"><tr valign=\"top\"><td>\n" ;
std::ifstream histo_file2(histos_path.c_str()) ;

cat = "" ;
do
 {
  std::getline(histo_file2,cat) ;
 } while (cat.empty()) ;

while (histo_file2>>histo_name>>scaled>>log>>err>>divide>>num>>denom>>eol>>eoc)
 {
  gif_name = "gifs/"+histo_name+".gif" ;
  gif_path = val_web_path+"/"+gif_name ;
  canvas_name = std::string("c")+histo_name ;
  canvas = new TCanvas(canvas_name.c_str()) ;
  canvas->SetFillColor(10) ;
  short_histo_name = histo_name ;
  short_histo_name.Remove(0,2) ;

  web_page<<"<a id=\""<<short_histo_name<<"\" name=\""<<short_histo_name<<"\"></a>" ;

  if ( file_ref != 0 )
   {
    histo_ref = (TH1 *)file_ref->Get(file_ref_dir+histo_name.c_str()) ;
    if (histo_ref!=0)
     {
      histo_ref->SetLineColor(4) ;
      histo_ref->SetLineWidth(3) ;
      if (divide==0)
       { histo_ref->Draw("hist") ; }
      else
       {
        // special for efficiencies
        TH1F *h_num = (TH1F *)file_ref->Get(file_ref_dir+num.c_str()) ;
        TH1F *h_res = (TH1F*)h_num->Clone("res");
        h_res->Reset();
        TH1F *h_denom = (TH1F *)file_ref->Get(file_ref_dir+denom.c_str()) ;
        std::cout << "DIVIDING "<< num.c_str() << " by " << denom.c_str() << std::endl;
        h_res->Divide(h_num,h_denom,1,1,"b");
        h_res->GetXaxis()->SetTitle(h_num->GetXaxis()->GetTitle());
        h_res->GetYaxis()->SetTitle(h_num->GetYaxis()->GetTitle());
        h_res->SetLineColor(4) ;
        h_res->SetLineWidth(3) ;
        h_res ->Draw("hist") ;
       }
     }
    else
     {
      web_page<<"No <b>"<<histo_name<<"</b> for "<<val_ref_release<<".<br>" ;
     }
   }

  gErrorIgnoreLevel = kWarning ;

  histo_new = (TH1 *)file_new->Get(file_new_dir+histo_name.c_str()) ;
  if (histo_new!=0)
   {
    if (log==1) canvas->SetLogy(1);
    histo_new->SetLineColor(2) ;
    histo_new->SetMarkerColor(2) ;
    histo_new->SetLineWidth(3) ;
    if ((scaled==1)&&(file_ref!=0)&&(histo_ref!=0)&&(histo_new->GetEntries()!=0))
     { if (histo_ref!=0) histo_new->Scale(histo_ref->GetEntries()/histo_new->GetEntries()) ; }
    if (divide==0)
     {
      if (err==1)
       {
        if (histo_ref!=0) histo_new->Draw("same E1 P") ;
        else histo_new->Draw("E1 P") ;
       }
      else
       {
        if (histo_ref!=0) histo_new->Draw("same hist") ;
        else histo_new->Draw("hist") ;
       }
     }
	  else
     {
      // special for efficiencies
      TH1F *h_num = (TH1 *)file_new->Get(file_new_dir+num.c_str()) ;
      TH1F *h_res = (TH1F*)h_num->Clone("res");
      TH1F *h_denom = (TH1 *)file_new->Get(file_new_dir+denom.c_str()) ;
      h_res->Divide(h_num,h_denom,1,1,"b");
      h_res->GetXaxis()->SetTitle(h_num->GetXaxis()->GetTitle());
      h_res->GetYaxis()->SetTitle(h_num->GetYaxis()->GetTitle());
      h_res->SetLineColor(2) ;
      h_res->SetMarkerColor(2) ;
      h_res->SetLineWidth(3) ;
      if (err==1) h_res ->Draw("same E1 P") ;
      else  h_res ->Draw("same hist") ;
     }
    std::cout<<histo_name
      <<" has "<<histo_new->GetEffectiveEntries()<<" entries"
      <<" of mean value "<<histo_new->GetMean()
      <<std::endl ;
    canvas->SaveAs(gif_path.c_str()) ;
	  web_page<<"<img class=\"image\" width=\"500\" src=\""<<gif_name<<"\"><br>" ;
   }
  else if ((file_ref!=0)&&(histo_ref!=0))
   {
    std::cout<<histo_name<<" NOT FOUND"<<std::endl ;
    web_page<<"<br>(no such histo for "<<val_new_release<<")" ;
    canvas->SaveAs(gif_path.c_str()) ;
	  web_page<<"<img class=\"image\" width=\"500\" src=\""<<gif_name<<"\"><br>" ;
   }
  else
   {
    web_page<<"No <b>"<<histo_name<<"</b> for "<<val_new_release<<".<br>" ;
   }
  if (eol)
   { web_page<<"</td></tr>\n<tr valign=\"top\"><td>" ; }
  else
   { web_page<<"</td><td>" ; }
  if (eoc)
   {
	cat = "" ;
    do
     {
      std::getline(histo_file2,cat) ;
     } while (cat.empty()) ;
   }
 }
histo_file2.close() ;
web_page<<"</td></tr></table>\n" ;

// cumulated efficiencies

web_page<<"\n</html>"<<std::endl ;
web_page.close() ;

}
