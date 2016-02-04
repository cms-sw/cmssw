
#include "TObject.h"
#include "TH1.h"
#include "TList.h"
#include "TDirectory.h"
#include <fstream>
#include <vector>
#include "TString.h"
#include "<cstring>"
#include "<string>"
#include <iomanip>

void Revert( TString oldfilename , TString newfilename )
{
  cout << oldfilename << "  " << newfilename << endl;
 
  char abs_path[500] = getenv("PWD");
  TString pwd(abs_path);
  cout << pwd << endl;
  
  TString f_ =pwd; f_+="/"; f_+=oldfilename;
  //  TFile *newfile = new TFile("METTester_data_ttbar.root","RECREATE"); 
  // TFile *File = new TFile("METTester_data_ttbarold.root");
  TFile *newfile = new TFile(newfilename,"RECREATE"); 
  TFile *File = new TFile(oldfilename);
  newfile->cd("");
  newfile->mkdir("DQMData");
  newfile->cd("DQMData");
  TDirectory *dir_DQMData(gDirectory);

  dir_DQMData->mkdir("METTask");
  newfile->cd("DQMData/METTask");
  TDirectory *dir_METTask(gDirectory);

  
  TString metdirname = "DQMData/RecoMETV/METTask/MET/met";

  File->cd(metdirname);
  
  TDirectory *dir(gDirectory);
  TIter nextkey(dir->GetListOfKeys());  
  TKey *key;
  cout << "ENTERING WHILE LOOP" << endl;
  
  while( (key = (TKey*)nextkey() ) )
    {
      TObject *obj = key->ReadObj();
      if ( obj->IsA()->InheritsFrom("TH1")  ) 
	{	   
	  TString histo_name = obj->GetName();
	  TH1 *histo = (TH1*)obj;
	  newfile->cd("DQMData/METTask");
	  histo->Write(histo_name);
	  File->cd(metdirname);
	}
    }

  TString genmetdirname = "DQMData/RecoMETV/METTask/MET/genMet";
  File->cd(genmetdirname);
  
  TIter genmet_it(gDirectory->GetListOfKeys());
  
  while( ( key = (TKey*)genmet_it() ) ) 
    {
      TObject *obj = key->ReadObj() ;
      if( obj->IsA()->InheritsFrom("TH1") ) 
	{
	  TString histo_name = obj->GetName();
	  TH1 *histo = (TH1*)obj;
	  newfile->cd("DQMData/METTask");
	  histo->Write(histo_name);
	  File->cd(genmetdirname);
	}
    }
  
  TString calodirname = "DQMData/RecoMETV/METTask/CaloTowers/SchemeB/";
  dir_METTask->mkdir("CT");
  dir_METTask->cd("CT");
  TDirectory *dir_CT(gDirectory);
  dir_CT->mkdir("data");
  dir_CT->mkdir("geometry");
  File->cd(calodirname);
  TIter calo_it(gDirectory->GetListOfKeys() );
  while( ( key = (TKey*)calo_it() ) ) 
    {
      TObject *obj = key->ReadObj() ;
      if( obj->IsA()->InheritsFrom("TH1") ) 
	{
	  TString histo_name = obj->GetName();
	  TH1 *histo = (TH1*)obj;
	  cout << histo->GetName() << endl;
	  newfile->cd("DQMData/METTask/CT/data");
	  histo->Write(histo_name);
	  File->cd(calodirname);
	}
    }
  TString calogeomdirname = "DQMData/RecoMETV/METTask/CaloTowers/geometry/";
  File->cd(calogeomdirname);
  TIter calogeo_it(gDirectory->GetListOfKeys());
  while( ( key = (TKey*)calogeo_it() ) ) 
    {
      TObject *obj = key->ReadObj() ;
      if( obj->IsA()->InheritsFrom("TH1") ) 
	{
	  TString histo_name = obj->GetName();
	  TH1 *histo = (TH1*)obj;
	  cout << histo->GetName() << endl;
	  newfile->cd("DQMData/METTask/CT/geometry");
	  histo->Write(histo_name);
	  File->cd(calogeomdirname);
	}
    }


  TString HCALdirname = "DQMData/RecoMETV/METTask/RecHits/HCAL/data";
  dir_METTask->mkdir("HCAL");
  dir_METTask->cd("HCAL");
  TDirectory *dir_HCAL(gDirectory);
  dir_HCAL->mkdir("data");
  dir_HCAL->mkdir("geometry");
  File->cd(HCALdirname);
  TIter HCAL_it(gDirectory->GetListOfKeys() );
  while( ( key = (TKey*)HCAL_it() ) ) 
    {
      TObject *obj = key->ReadObj() ;
      if( obj->IsA()->InheritsFrom("TH1") ) 
	{
	  TString histo_name = obj->GetName();
	  TH1 *histo = (TH1*)obj;
	  cout << histo->GetName()  << endl;
	  newfile->cd("DQMData/METTask/HCAL/data");
	  histo->Write(histo_name);
	  File->cd(HCALdirname);
	}
    }
  
  TString HCALgeodirname = "DQMData/RecoMETV/METTask/RecHits/HCAL/geometry";
  File->cd(HCALgeodirname);
  TIter HCALgeo_it(gDirectory->GetListOfKeys() );
  while( ( key = (TKey*)HCALgeo_it() ) ) 
    {
      TObject *obj = key->ReadObj() ;
      if( obj->IsA()->InheritsFrom("TH1") ) 
	{
	  TString histo_name = obj->GetName();
	  TH1 *histo = (TH1*)obj;
	  cout << histo->GetName()  << endl;
	  newfile->cd("DQMData/METTask/HCAL/geometry");
	  histo->Write(histo_name);
	  File->cd(HCALgeodirname);
	}
    }



  TString ECALdirname = "DQMData/RecoMETV/METTask/RecHits/ECAL/data";
  dir_METTask->mkdir("ECAL");
  dir_METTask->cd("ECAL");
  TDirectory *dir_ECAL(gDirectory);
  dir_ECAL->mkdir("data");
  dir_ECAL->mkdir("geometry");
  File->cd(ECALdirname);
  TIter ECAL_it(gDirectory->GetListOfKeys() );
  while( ( key = (TKey*)ECAL_it() ) )
    {
      TObject *obj = key->ReadObj() ;
      if( obj->IsA()->InheritsFrom("TH1") )
        {
          TString histo_name = obj->GetName();
          TH1 *histo = (TH1*)obj;
          cout << histo->GetName()  << endl;
          newfile->cd("DQMData/METTask/ECAL/data");
          histo->Write(histo_name);
          File->cd(ECALdirname);
        }
    }

  TString ECALgeodirname = "DQMData/RecoMETV/METTask/RecHits/ECAL/geometry";
  File->cd(ECALgeodirname);
  TIter ECALgeo_it(gDirectory->GetListOfKeys() );
  while( ( key = (TKey*)ECALgeo_it() ) )
    {
      TObject *obj = key->ReadObj() ;
      if( obj->IsA()->InheritsFrom("TH1") )
        {
          TString histo_name = obj->GetName();
          TH1 *histo = (TH1*)obj;
	  cout << histo->GetName() << endl;
          newfile->cd("DQMData/METTask/ECAL/geometry");
          histo->Write(histo_name);
          File->cd(ECALgeodirname);
        }
    }


  newfile->Close();


}


