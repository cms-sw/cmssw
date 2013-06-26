#include <iostream.h>
#include "TFile.h"

class HistoCompare_Tracks {

 public:

  HistoCompare_Tracks() 
  { 
    std::cout << "Initializing HistoCompare_Tracks... " << std::endl; 
    name = "none";
  } ;

  HistoCompare_Tracks(char* thisname = "none") : mypv(-9999.9)
  { 
    name = thisname;
    std::cout << "Initializing HistoCompare_Tracks... " << std::endl; 
    if ( name != "none" )
      {
	cout << "... creating output file" << endl;
	
	out_file.open( thisname, ios::out);
	if ( out_file.fail() )
	  {
	    cout << "Could not open data file" << endl;
	    exit(1);
	  }
      }
  } ;
  
  ~HistoCompare_Tracks()
  {
    if ( name != "none" )
      {
	cout << "... closing output file" << endl;
	out_file.close();
      }
    
  };

  void PVCompute(TH1 * oldHisto , TH1 * newHisto , TText * te,char * option = "");
//   void PVCompute(TH2 * oldHisto , TH2 * newHisto , TText * te );
//   void PVCompute(TProfile * oldHisto , TProfile * newHisto , TText * te );

  Double_t getPV() { return mypv; };
  void setName(char* s) { name = s; };

 private:
  
  Double_t mypv;

  TH1 * myoldHisto1;
  TH1 * mynewHisto1;

  TH2 * myoldHisto2;
  TH2 * mynewHisto2;

  TProfile * myoldProfile;
  TProfile * mynewProfile;

  TText * myte;
  
  char* name;
  
  fstream out_file;

};

HistoCompare_Tracks::PVCompute(TH1 * oldHisto , TH1 * newHisto , TText * te,char * option)
{

  myoldHisto1 = oldHisto;
  mynewHisto1 = newHisto;
  myte = te;

  //mypv = myoldHisto1->Chi2Test(mynewHisto1,option);
  mypv = myoldHisto1->KolmogorovTest(mynewHisto1);
  std::strstream buf;
  std::string value;
  buf<<"PV="<<mypv<<std::endl;
  buf>>value;
  
  myte->DrawTextNDC(0.6,0.7, value.c_str());

  std::cout << "[OVAL] " << myoldHisto1->GetName() << " PV = " << mypv << std::endl;
  
  if ( name != "none" )
    {
      if ( mypv < 0.01 )
	out_file << myoldHisto1->GetName() << "     pv = " << mypv << "      comparison fails !!!" << endl; 
      else
	out_file << myoldHisto1->GetName() << "     pv = " << mypv << endl;
    }
  
  return;
}

