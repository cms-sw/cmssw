#include "../interface/HistoMgr.h"
#include "TFile.h"
#include <iostream>


HistoMgr* HistoMgr::theInstance = 0;

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HistoMgr* HistoMgr::getInstance()
{
  if( theInstance == 0 ){
    theInstance = new HistoMgr;
  }
  return theInstance;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HistoMgr::HistoMgr()
{
}
 

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HistoMgr::~HistoMgr()
{ 
  mih1::const_iterator ite1;

  for( ite1 = theHistos1.begin(); ite1 != theHistos1.end(); ite1++ ){ 
    delete (*ite1).second;
  }

  mih2::const_iterator ite2;

  for( ite2 = theHistos2.begin(); ite2 != theHistos2.end(); ite2++ ){
    delete (*ite2).second;
  }
 
}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void HistoMgr::save( const std::string& name )
{

  cout << "=== saving user histos ===" << endl;
  TFile fiche(name.c_str(),"recreate");
  // write out the histos
  mih1::const_iterator ite1;

  for( ite1 = theHistos1.begin(); ite1 != theHistos1.end(); ite1++ ){
    ((*ite1).second)->Write();
  }

  mih2::const_iterator ite2;

  for( ite2 = theHistos2.begin(); ite2 != theHistos2.end(); ite2++ ){
    ((*ite2).second)->Write();
  }

  mihP::const_iterator iteP;

  for( iteP = theHistosP.begin(); iteP != theHistosP.end(); iteP++ ){
    ((*iteP).second)->Write();
  }

  cout << "=== user histos saved ===" << endl;

}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

// void HistoMgr::save(TFile *theFile )
void HistoMgr::save(int nHist )
{
//  theFile->cd();
//  gDirectory->pwd();
//  theFile->ls();
 
  cout << "=== saving user histos ===" << endl;
  // write out the histos
  mih1::const_iterator ite1;

  for( ite1 = theHistos1.begin(); ite1 != theHistos1.end(); ite1++ ){
   if ( (*ite1).first >= nHist && (*ite1).first < (nHist+1000) )
     ((*ite1).second)->Write();
  }
 
  mih2::const_iterator ite2;

  for( ite2 = theHistos2.begin(); ite2 != theHistos2.end(); ite2++ ){
    if ( (*ite2).first >= nHist && (*ite2).first < (nHist+1000) )
    ((*ite2).second)->Write();
  }

  mihP::const_iterator iteP;

  for( iteP = theHistosP.begin(); iteP != theHistosP.end(); iteP++ ){
    if ( (*iteP).first >= nHist && (*iteP).first < (nHist+1000) )
    ((*iteP).second)->Write();
  }

  cout << "=== user histos saved ===" << endl;

}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bool HistoMgr::addHisto1( int ih, TH1F* his )
{
  theHistos1[ih] = his;
  //cout << " addHisto1 " << his->GetName() << " = " << ih << endl;

  return true;
}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bool HistoMgr::addHisto2( int ih, TH2F* his )
{
  theHistos2[ih] = his;
//  cout << " addHisto2 " << sih->name() << " = " << ih << endl;

  return true;
}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bool HistoMgr::addHistoP( int ih, TProfile* his )
{
  theHistosP[ih] = his;
//  cout << " addHistoProfile " << sih->name() << " = " << ih << endl;

  return true;
}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TH1F* HistoMgr::getHisto1( int ih )
{
  TH1F* his = 0;

  mih1::const_iterator ite = theHistos1.find( ih );
  if( ite != theHistos1.end() ) {
    his = (*ite).second;
  } else {
    cerr << "!!!! FATAL ERROR 1d Histogram does not exist " << ih << endl;
    std::exception();
  }
  return his;

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TH2F* HistoMgr::getHisto2( int ih )
{
  TH2F* his = 0;

  mih2::const_iterator ite = theHistos2.find( ih );
  if( ite != theHistos2.end() ) {
    his = (*ite).second;
  } else {
    cerr << "!!!! FATAL ERROR 2d Histogram does not exist " << ih << endl;
    std::exception();
  }
  return his;

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TProfile* HistoMgr::getHistoP( int ih )
{
  TProfile* his = 0;

  mihP::const_iterator ite = theHistosP.find( ih );
  if( ite != theHistosP.end() ) {
    his = (*ite).second;
  } else {
    cerr << "!!!! FATAL ERROR Profile Histogram does not exist " << ih << endl;
    std::exception();
  }
  return his;

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TH1F* HistoMgr::getHisto1(  const std::string& name )
{
  TH1F* his = 0;

  mih1::const_iterator ite;
  for( ite = theHistos1.begin(); ite != theHistos1.end(); ite++ ){
    if( (*ite).second->GetName() == name ) {
      his = (*ite).second;
      break;
    }
  }

  if( ite == theHistos1.end() ) {
    cerr << "!!!! FATAL ERROR 1d Histogram does not exist " << name << endl;
    std::exception();
  }

  return his;

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TH2F* HistoMgr::getHisto2(  const std::string& name )
{
  TH2F* his = 0;

  mih2::const_iterator ite;
  for( ite = theHistos2.begin(); ite != theHistos2.end(); ite++ ){
    if( (*ite).second->GetName() == name ) {
      his = (*ite).second;
      break;
    }
  }

  if( ite == theHistos2.end() ) {
    cerr << "!!!! FATAL ERROR 2d Histogram does not exist " << name << endl;
    std::exception();
  }

  return his;

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TProfile* HistoMgr::getHistoP(  const std::string& name )
{
  TProfile* his = 0;

  mihP::const_iterator ite;
  for( ite = theHistosP.begin(); ite != theHistosP.end(); ite++ ){
    if( (*ite).second->GetName() == name ) {
      his = (*ite).second;
      break;
    }
  }

  if( ite == theHistosP.end() ) {
    cerr << "!!!! FATAL ERROR Profile Histogram does not exist " << name << endl;
    std::exception();
  }

  return his;

}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int HistoMgr::getHistoId1(  const std::string& name )
{
  int hid = -1;

  mih1::const_iterator ite;
  for( ite = theHistos1.begin(); ite != theHistos1.end(); ite++ ){
    if( (*ite).second->GetName() == name ) {
      hid = (*ite).first;
      cout << " getHistoId1 " << hid << " = " << name << endl;
      break;
    }
  }

  if( ite == theHistos1.end() ) {
    cerr << "!!!! FATAL ERROR 1d Histogram does not exist " << name << endl;
    std::exception();
  }

  return hid;

}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int HistoMgr::getHistoId2(  const std::string& name )
{
  int hid = -1;

  mih2::const_iterator ite;
  for( ite = theHistos2.begin(); ite != theHistos2.end(); ite++ ){
    if( (*ite).second->GetName() == name ) {
      hid = (*ite).first;
      break;
    }
  }

  if( ite == theHistos2.end() ) {
    cerr << "!!!! FATAL ERROR 2d Histogram does not exist " << name << endl;
    std::exception();
  }

  return hid;

}




//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// return h1/h2 with recalculated errors
//
TH1F* HistoMgr::divideErr( const int h1, const int h2)
{

  TH1F* his1 = getHisto1(h1);
  TH1F* his2 = getHisto1(h2);

  if( his1 && his2 ) {
    return divideErr( his1, his2 );
  } else {
    return 0;
  }
}


//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TH1F* HistoMgr::divideErr(TH1F* h1, TH1F* h2) 
{

  TH1F* hout = new TH1F(*h1);
  hout->Reset();
//hout->SetName((std::string(h1->GetName()) + std::string("/") + std::string(h2->GetName())).c_str());
  hout->SetName((std::string(h1->GetName()) + std::string("by") + std::string(h2->GetName())).c_str());
  hout->Divide(h1,h2,1.,1.,"B");

  for (int i = 0; i <= hout->GetNbinsX()+1; i++ ) {
    Float_t tot   = h2->GetBinContent(i) ;
    Float_t tot_e = h2->GetBinError(i);
    Float_t eff = hout->GetBinContent(i) ;
    Float_t Err = 0.;
    if (tot > 0) Err = tot_e / tot * sqrt( eff* (1-eff) );
    if (eff == 1.) Err=1.e-3;

    hout->SetBinError(i, Err);
  }
  return hout;
}

