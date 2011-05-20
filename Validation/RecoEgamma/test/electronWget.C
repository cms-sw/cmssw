
int electronWget()
 {
  TString TEST_HARVESTED_FILE = gSystem->Getenv("TEST_HARVESTED_FILE") ;
  TString TEST_HISTOS_FILE = gSystem->Getenv("TEST_HISTOS_FILE") ;
  TString VAL_ANALYZER = gSystem->Getenv("VAL_ANALYZER") ;

  TString input_path("DQMData/Run 1/EgammaV/Run summary/") ;
  input_path += VAL_ANALYZER ;
  TFile * input_file = TFile::Open(TEST_HARVESTED_FILE) ;
  TList * input_keys ;
  if (input_file!=0)
   {
    std::cout<<"open "<<TEST_HARVESTED_FILE<<std::endl ;
    if (input_file->cd(input_path)==kTRUE)
     {
      std::cerr<<"cd "<<input_path<<std::endl ;
      input_keys = gDirectory->GetListOfKeys() ;
     }
    else
     {
      std::cerr<<"Failed move to: "<<input_path<<std::endl ;
      return 2 ;
     }
   }
  else
   {
    std::cerr<<"Failed to open: "<<val_ref_file_name<<std::endl ;
    return 1 ;
   }

  TFile * output_file = TFile::Open(TEST_HISTOS_FILE,"RECREATE") ; ;
  TString output_path("DQMData/EgammaV/") ;
  output_path += VAL_ANALYZER ;
  if (output_file!=0)
   {
    std::cout<<"open "<<TEST_HISTOS_FILE<<std::endl ;
    if (output_file->mkdir("DQMData")!=0)
     { std::cerr<<"cd "<<"DQMData"<<std::endl ; output_file->cd("DQMData") ; }
    else
     { std::cerr<<"Failed move to: "<<"DQMData"<<std::endl ; return 4 ; }
    if (gDirectory->mkdir("EgammaV")!=0)
     { std::cerr<<"cd EgammaV"<<std::endl ; gDirectory->cd("EgammaV") ; }
    else
     { std::cerr<<"Failed move to: "<<"EgammaV"<<std::endl ; return 5 ; }
    if (gDirectory->mkdir(VAL_ANALYZER)!=0)
     { std::cerr<<"cd "<<VAL_ANALYZER<<std::endl ; gDirectory->cd(VAL_ANALYZER) ; }
    else
     { std::cerr<<"Failed move to: "<<VAL_ANALYZER<<std::endl ; return 6 ; }
   }
  else
   {
    std::cerr<<"Failed to create: "<<TEST_HISTOS_FILE<<std::endl ;
    return 3 ;
   }

  TObject * obj ;
  TH1 * histo ;
  TKey * key ;
  TIter nextKey(input_keys) ;
  while (key = (TKey *)nextKey())
   {
    obj = key->ReadObj() ;
    if (obj->IsA()->InheritsFrom("TH1"))
     {
      histo = (TH1 *)obj ;
      std::cout
        <<"Histo "<<histo->GetName()
        <<" has "<<histo->GetEntries()<<" entries"
        <<" (~"<<histo->GetEffectiveEntries()<<")"
        <<" of mean value "<<histo->GetMean()
        <<std::endl ;
      histo->Clone() ;
     }
    else
     { std::cout<<"What is "<<obj->GetName()<<" ?"<<std::endl ; }
   }


  input_file->Close() ;
  output_file->Write() ;
  output_file->Close() ;
  return 0 ;

 }
