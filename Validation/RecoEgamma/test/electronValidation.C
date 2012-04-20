#include <TObjArray.h>

TH1F * DivideHistos
 ( TFile * f,
   TH1 * h_proto,
   const TString & num_name,
   const TString & denom_name )
 {
  TH1F * h_num = (TH1F *)f->Get(num_name) ;
  TH1F * h_denom = (TH1F *)f->Get(denom_name) ;
//  std::cout
//   <<"DIVIDING "<<num_name<<" ("<<h_num->GetEntries()<<" entries)"
//   <<" by "<<denom_name<<" ("<<h_denom->GetEntries()<<" entries)"
//   <<std::endl ;
  TH1F * h_res = (TH1F*)h_proto->Clone();
  h_res->Divide(h_num,h_denom,1,1,"b") ;
//  h_res->GetXaxis()->SetTitle(h_proto->GetXaxis()->GetTitle()) ;
//  h_res->GetYaxis()->SetTitle(h_proto->GetYaxis()->GetTitle()) ;
////  h_res->SetName(h_proto->GetName()) ;
//  h_res->SetTitle(h_proto->GetTitle()) ;
  return h_res ;
 }

void Join
 ( const TObjArray * tokens, TString & common )
 {
  tokens->Compress() ;
  if (tokens->GetEntries()==0)
   { common = "" ; return ; }
  else
   { common = ((TObjString *)(tokens->At(0)))->GetString() ; }
  if (tokens->GetEntries()==1)
   { return ; }
  TObjString * token_obj ;
  Int_t token, max_token = tokens->GetEntries() ;
  for ( token=1 ; token<max_token ; ++token )
   {
    common += "_" ;
    token_obj = (TObjString *)(tokens->At(token)) ;
    common += token_obj->GetString() ;
   }
 }

void CompareHistoNames
 ( const TString & h1_name,
   const TString & h2_name,
   TString &  common,
   TString &  h1_specific,
   TString &  h2_specific )
 {
  TObjArray * h1_tokens = h1_name.Tokenize("_") ;
  TObjArray * h2_tokens = h2_name.Tokenize("_") ;
  TObjArray * common_tokens = new TObjArray ;
  Int_t h1_nb_tokens = h1_tokens->GetEntries() ;
  Int_t h2_nb_tokens = h2_tokens->GetEntries() ;
  Int_t token, max_token = (h1_nb_tokens<h2_nb_tokens?h1_nb_tokens:h2_nb_tokens) ;
  for ( token=0 ; token<max_token ; ++token )
   {
    if (h1_tokens->At(token)->IsEqual(h2_tokens->At(token))==kFALSE) break ;
    common_tokens->Add(h1_tokens->At(token)) ;
    h1_tokens->RemoveAt(token) ;
    h2_tokens->RemoveAt(token) ;
   }
  Join(common_tokens,common) ;
  Join(h1_tokens,h1_specific) ;
  Join(h2_tokens,h2_specific) ;
 }

void RenderHisto( TObject * obj, TCanvas * canvas )
 {
  assert(obj->InheritsFrom("TH1")) ;
  TH1 * histo = dynamic_cast<TH1*>(obj) ;
  assert(histo) ;

  TString histo_option = ((TH1 *)obj)->GetOption() ;
  if ((histo_option.Contains("ELE_LOGY")==kTRUE)&&(histo->GetMaximum()>0))
   { canvas->SetLogy(1) ; }

  int histo_name_flag = 1 ; // use 0 to switch off
  if ( obj->InheritsFrom("TH2") )
   {
    gStyle->SetPalette(1) ;
    gStyle->SetOptStat(110+histo_name_flag) ;
   }
  else if ( obj->InheritsFrom("TProfile") )
   { gStyle->SetOptStat(110+histo_name_flag) ; }
  else // TH1
   { gStyle->SetOptStat(111110+histo_name_flag) ; }
 }


int electronValidation()
 {
  TString DBS_SAMPLE = gSystem->Getenv("DBS_SAMPLE") ;
  TString DBS_COND = gSystem->Getenv("DBS_COND") ;

  TString val_ref_file_name = gSystem->Getenv("VAL_REF_FILE") ;
  TString val_new_file_name = gSystem->Getenv("VAL_NEW_FILE") ;
  TString val_ref_release = gSystem->Getenv("VAL_REF_RELEASE") ;
  TString val_new_release = gSystem->Getenv("VAL_NEW_RELEASE") ;
  TString val_analyzer = gSystem->Getenv("VAL_ANALYZER") ;
  TString val_configuration = gSystem->Getenv("VAL_CONFIGURATION") ;
  TString val_comment = gSystem->Getenv("VAL_COMMENT") ;

  TString val_web = gSystem->Getenv("VAL_WEB") ;
  TString val_web_sub_dir = gSystem->Getenv("VAL_WEB_SUB_DIR") ;
  TString val_web_url = gSystem->Getenv("VAL_WEB_URL") ;

  std::string val_web_path = val_web+"/"+val_new_release+"/Electrons/vs"+val_ref_release+"/"+val_web_sub_dir ;
  std::string val_web_url_path = val_web_url+"/"+val_new_release+"/Electrons/vs"+val_ref_release+"/"+val_web_sub_dir ;
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

  TString internal_path("DQMData/Run 1/EgammaV/Run summary/") ;
  TString old_internal_path("DQMData/EgammaV/") ;

  TString val_ref_file_url ;
  TString file_ref_dir ;
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
      if (file_ref->cd(internal_path)==kTRUE)
       {
        std::cerr<<"cd "<<internal_path<<std::endl ;
        file_ref_dir = internal_path ;
        file_ref->cd() ;
       }
      else if (file_ref->cd(old_internal_path)==kTRUE)
       {
        std::cerr<<"cd "<<old_internal_path<<std::endl ;
        file_ref_dir = old_internal_path ;
        file_ref->cd() ;
       }
      else
       {
        std::cerr<<"Failed move to: "<<internal_path<<" and "<<old_internal_path<<std::endl ;
        file_ref_dir = "" ;
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
      if (file_new->cd(internal_path)==kTRUE)
       {
        std::cerr<<"cd "<<internal_path<<std::endl ;
        file_new_dir = internal_path ;
        file_new->cd() ;
       }
      else if (file_new->cd(old_internal_path)==kTRUE)
       {
        std::cerr<<"cd "<<old_internal_path<<std::endl ;
        file_new_dir = old_internal_path ;
        file_new->cd() ;
       }
      else
       {
        std::cerr<<"Failed move to: "<<internal_path<<" and "<<old_internal_path<<std::endl ;
        file_new_dir = "" ;
       }
     }
    else
     { std::cerr<<"Failed to open: "<<val_new_file_name<<std::endl ; }
   }

  TCanvas * canvas ;
  TH1 * histo_ref, * histo_new ;
  TPaveStats * st_ref, * st_new ;

  std::ofstream web_page(index_path.c_str()) ;

  web_page
    <<"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n"
    <<"<html>\n"
    <<"<head>\n"
    <<"<meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\" />\n"
    <<"<title>"<<val_new_release<<" vs "<<val_ref_release<<" / "<<DBS_SAMPLE<<" / "<<DBS_COND<<"</title>\n"
    <<"</head>\n"
    <<"<h1><a href=\"../\"><img border=0 width=\"22\" height=\"22\" src=\"../../../../img/up.gif\" alt=\"Up\"/></a>&nbsp;"<<val_new_release<<" vs "<<val_ref_release<<" / "<<DBS_SAMPLE<<" / "<<DBS_COND<<"</h1>\n" ;

  web_page<<"<p>"<<val_comment ;
  if (file_ref==0)
   {
    web_page
     <<" In all plots below"
     <<", there was no "<<val_ref_release<<" histograms to compare with"
       <<", and the <a href=\""<<val_new_file_url<<"\">"<<val_new_release<<" histograms</a> are in red"
     <<"." ;
   }
  else
   {
    web_page
     <<" In all plots below"
       <<", the <a href=\""<<val_new_file_url<<"\">"<<val_new_release<<" histograms</a> are in red"
     <<", and the <a href=\""<<val_ref_file_url<<"\">"<<val_ref_release<<" histograms</a> are in blue"
     <<"." ;
   }
  web_page
    <<" They were made using analyzer "
    <<"<a href=\"http://cmslxr.fnal.gov/lxr/source/Validation/RecoEgamma/plugins/"<<val_analyzer<<".h\">"
    <<"Validation/RecoEgamma/plugins/"<<val_analyzer<<".h"
    <<"</a> and configuration "
    <<"<a href=\"http://cmslxr.fnal.gov/lxr/source/Validation/RecoEgamma/test/"<<val_configuration<<"\">"
    <<"Validation/RecoEgamma/test/"<<val_configuration
    <<"</a>, with dataset "<<DBS_SAMPLE<<" as input." ;
  web_page
    <<" Some more details"
    <<": <a href=\"electronValidation.C\">script</a> used to make the plots"
    <<", <a href=\"histos.txt\">specification</a> of histograms"
    <<", <a href=\"gifs/\">images</a> of histograms"
    <<"." ;
  web_page<<"</p>\n" ;

  // canvas_name std::string => TString
  TString canvas_name, histo_name, histo_full_path, gif_name, gif_path ;
  TString short_histo_name ;
  TString first_short_histo_name, first_histo_name ;
  TString dl_short_histo_name, dl_histo_name ;
  TString num_ref, denom_ref, num_full, denom_full ;
  Int_t n_ele_charge ;
  int scaled, err ;
  int divide;
  std::string cat, line, histo_path, num, denom ;
  int eol ; // end of line
  int eoc ; // enf of category
  double rescale_factor = 0. ;

  std::ifstream histo_file1(histos_path.c_str()) ;

  web_page
    <<"<br><table border=\"1\" cellpadding=\"5\" width=\"100%\">"
    <<"<tr valign=\"top\"><td width=\"20%\">\n" ;
  int cat_num = 0 ;

  cat = "" ;
  do
   {
    std::getline(histo_file1,cat) ;
   } while (cat.find_first_not_of(" \t")==std::string::npos) ;

  web_page<<"<b>"<<cat<<"</b><br><br>" ;

  while (std::getline(histo_file1,line))
   {
    if (line.empty()) continue ;
    std::size_t first = line.find_first_not_of(" \t") ;
    if (first==std::string::npos) continue ;
    if (line[first]=='#') continue ;

    std::istringstream linestream(line) ;
    divide = 0 ; num = denom = "" ;
    linestream >> histo_path >> scaled >> err >> eol >> eoc >> divide >> num >> denom ;

    histo_name = histo_path ;
    Ssiz_t pos = histo_name.Last('/') ;
    if (pos!=kNPOS) histo_name.Remove(0,pos+1) ;
    short_histo_name = histo_name ;
    short_histo_name.Remove(0,2) ;
    if (short_histo_name.BeginsWith("ele_"))
     { short_histo_name.Remove(0,4) ; }

    if (first_histo_name.IsNull())
     {
      first_short_histo_name = short_histo_name ;
      first_histo_name = histo_name ;
      dl_short_histo_name = short_histo_name ;
      dl_histo_name = histo_name ;
     }
    else
     {
      TString common ;
      TString first_specific ;
      TString second_specific ;
      CompareHistoNames(first_short_histo_name,short_histo_name,common,first_specific,second_specific) ;
      if (!dl_histo_name.IsNull())
       {
        if (first_specific.IsNull())
         { web_page<<"<a href=\"#"<<first_histo_name<<"\">"<<first_short_histo_name<<"</a>" ; }
        else
         { web_page<<common<<"&nbsp;|&nbsp;<a href=\"#"<<first_histo_name<<"\">"<<first_specific<<"</a>" ; }
        dl_short_histo_name = dl_histo_name = "" ;
       }
      web_page<<"&nbsp;|&nbsp;<a href=\"#"<<histo_name<<"\">"<<second_specific<<"</a>" ;
     }

    if ((eol)||(eoc))
     {
      if (!dl_histo_name.IsNull())
       { web_page<<"<a href=\"#"<<dl_histo_name<<"\">"<<dl_short_histo_name<<"</a>" ; }
      web_page<<"<br>\n" ;
      first_histo_name = first_short_histo_name = "" ;
     }

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
       } while (cat.find_first_not_of(" \t")==std::string::npos) ;
      web_page<<"<b>"<<cat<<"</b><br><br>" ;
     }
   }
  web_page<<"<br></td></tr></table>\n" ;
  histo_file1.close() ;

  web_page<<"<br><br><table cellpadding=\"5\"><tr valign=\"top\"><td><a href=\""<<val_web_url_path<<"\"><img width=\"18\" height=\"18\" border=\"0\" align=\"middle\" src=\"../../../../img/up.gif\" alt=\"Top\"/></a></td><td>\n" ;
  std::ifstream histo_file2(histos_path.c_str()) ;

  n_ele_charge = 0 ;
  cat = "" ;
  do
   {
    std::getline(histo_file2,cat) ;
   } while (cat.empty()) ;

  while (std::getline(histo_file2,line))
   {
    if (line.empty()) continue ;
    std::size_t first = line.find_first_not_of(" \t") ;
    if (first==std::string::npos) continue ;
    if (line[first]=='#') continue ;

    std::istrstream linestream(line) ;
    divide = 0 ; num = denom = "" ;
    linestream >> histo_path >> scaled >> err >> eol >> eoc >> divide >> num >> denom ;

    histo_name = histo_path.c_str() ;
    histo_ref = 0 ;
    histo_new = 0 ;
    st_ref = 0 ;
    st_new = 0 ;

    Ssiz_t pos = histo_name.Last('/') ;
    if (pos!=kNPOS) histo_name.Remove(0,pos+1) ;
    short_histo_name = histo_name ;
    short_histo_name.Remove(0,2) ;

    gif_name = "gifs/" ; gif_name += histo_name ; gif_name += ".gif" ;
    gif_path = val_web_path ; gif_path += "/" ; gif_path += gif_name ;
    canvas_name = "c" ; canvas_name += histo_name ;
    canvas = new TCanvas(canvas_name) ;
    canvas->SetFillColor(10) ;

    web_page<<"<a id=\""<<histo_name<<"\" name=\""<<short_histo_name<<"\"></a>" ;

    // search histo_ref
    if ( file_ref != 0 )
     {
      if (file_ref_dir.IsNull())
       { histo_full_path = histo_name ; }
      else
       { histo_full_path = file_ref_dir ; histo_full_path += histo_path.c_str() ; }
      histo_ref = (TH1 *)file_ref->Get(histo_full_path) ;
      if (histo_ref!=0)
       {
        // renaming those histograms avoid very strange bugs because they
        // have the same names as the ones loaded from the new file
        histo_ref->SetName(TString(histo_ref->GetName())+"_ref") ;
       }
      else
       {
        web_page<<"No <b>"<<histo_path<<"</b> for "<<val_ref_release<<".<br>" ;
       }
     }

    // search histo_new
    histo_full_path = file_new_dir ; histo_full_path += histo_path.c_str() ;
    histo_new = (TH1 *)file_new->Get(histo_full_path) ;

    // special treatments
    if ((scaled==1)&&(histo_new!=0)&&(histo_ref!=0)&&(histo_ref->GetEntries()!=0))
     {
      Int_t new_entries = histo_new->GetEntries() ;
      if (new_entries==0) { new_entries = n_ele_charge ; }
      if (new_entries==0)
       { std::cerr<<"DO NOT KNOW HOW TO RESCALE "<<histo_name<<std::endl ; }
      else
       {
        // we want to reuse the rescale factor of the first histogram
        // for all the subsequent histograms.
        if (rescale_factor==0.) { rescale_factor = new_entries/histo_ref->GetEntries() ; }
        histo_ref->Scale(rescale_factor) ;
       }
     }
    if ((histo_new!=0)&&(histo_ref!=0)&&(histo_ref->GetMaximum()>histo_new->GetMaximum()))
     { histo_new->SetMaximum(histo_ref->GetMaximum()*1.1) ; }

    if (histo_new==0)
     {
      web_page<<"No <b>"<<histo_path<<"</b> for "<<val_new_release<<".<br>" ;
     }
    else
     {
      // catch n_ele_charge
      if (histo_name=="h_ele_charge")
       { n_ele_charge = histo_new->GetEntries() ; }

      // draw histo_new
      TString newDrawOptions(err==1?"E1 P":"hist") ;
      gErrorIgnoreLevel = kWarning ;
      if (divide!=0)
       {
        num_full = file_new_dir ; num_full += num.c_str() ;
        denom_full = file_new_dir ; denom_full += denom.c_str() ;
        histo_new = DivideHistos(file_new,histo_new,num_full,denom_full) ;
       }
      histo_new->SetLineColor(kRed) ;
      histo_new->SetMarkerColor(2) ;
      histo_new->SetLineWidth(3) ;
      RenderHisto(histo_new,canvas) ;
      histo_new->Draw(newDrawOptions) ;
      canvas->Update() ;
      st_new = (TPaveStats*)histo_new->FindObject("stats");
      st_new->SetTextColor(kRed) ;

      // draw histo_ref
      if (histo_ref!=0)
       {
        if (divide!=0)
         {
          num_ref = num ;
          denom_ref = denom ;
          if (file_ref_dir.IsNull())
           {
            pos = num_ref.Last('/') ;
            if (pos!=kNPOS) num_ref.Remove(0,pos+1) ;
            pos = denom_ref.Last('/') ;
            if (pos!=kNPOS) denom_ref.Remove(0,pos+1) ;
           }
          histo_ref = DivideHistos(file_ref,histo_ref,file_ref_dir+num_ref,file_ref_dir+denom_ref) ;
         }
        RenderHisto(histo_ref,canvas) ;
        histo_ref->SetLineColor(kBlue) ;
        histo_ref->SetLineWidth(3) ;
        histo_ref->Draw("sames hist") ;
        canvas->Update() ;
        st_ref = (TPaveStats*)histo_ref->FindObject("stats");
        st_ref->SetTextColor(kBlue) ;
        Double_t y1 = st_ref->GetY1NDC() ;
        Double_t y2 = st_ref->GetY2NDC() ;
        st_ref->SetY1NDC(2*y1-y2) ;
        st_ref->SetY2NDC(y1) ;
       }

      // Redraws
      newDrawOptions = "sames " ;
      newDrawOptions += (err==1?"E1 P":"hist") ;
      histo_new->Draw(newDrawOptions) ;
      if (st_ref!=0) st_ref->Draw() ;
      if (st_new!=0) st_new->Draw() ;

      // eventual log scale
      //if ( (log==1) && ( (histo_new->GetEntries()>0) || ( (histo_ref!=0) && (histo_ref->GetEntries()!=0) ) ) )
      // { canvas->SetLogy(1) ; }

      std::cout<<histo_name
        <<" has "<<histo_new->GetEffectiveEntries()<<" entries"
        <<" of mean value "<<histo_new->GetMean()
        <<std::endl ;
      canvas->SaveAs(gif_path.Data()) ;
      web_page<<"<a href=\""<<gif_name<<"\"><img border=\"0\" class=\"image\" width=\"440\" src=\""<<gif_name<<"\"></a><br>" ;
     }

//    else if ((file_ref!=0)&&(histo_ref!=0))
//     {
//      std::cout<<histo_path<<" NOT FOUND"<<std::endl ;
//      web_page<<"<br>(no such histo)" ;
//      canvas->SaveAs(gif_path.Data()) ;
//      web_page<<"<a href=\""<<gif_name<<"\"><img border=\"0\" class=\"image\" width=\"440\" src=\""<<gif_name<<"\"></a><br>" ;
//     }

    if (eol)
     { web_page<<"</td></tr>\n<tr valign=\"top\"><td><a href=\""<<val_web_url_path<<"\"><img width=\"18\" height=\"18\" border=\"0\" align=\"middle\" src=\"../../../../img/up.gif\" alt=\"Top\"/></a></td><td>" ; }
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

  web_page<<"\n</html>"<<std::endl ;
  web_page.close() ;

 }
