#ifndef HistoMgr_h
#define HistoMgr_h 

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TFile.h"

#include <iostream>
#include "Riostream.h"
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <stdlib.h>

//#include "HTL/Histograms.h" // Transient histograms.
typedef std::map< int, TH1F* > mih1;
typedef std::map< int, TH2F* > mih2;
typedef std::map< int, TProfile* > mihP;

class TFile;
class HistoMgr 
{
 private:
  HistoMgr();
  ~HistoMgr();

public:
  static HistoMgr* getInstance();

  void save( const std::string& name );
//  void save();
//  void save( TFile *theFile); 
  void save(int nHist);
  bool addHisto1( int ih1, TH1F* his );
  bool addHisto2( int ih2, TH2F* his );
  bool addHistoP( int ihP, TProfile* his );

  TH1F* getHisto1( int ih1 );
  TH2F* getHisto2( int ih2 );
  TProfile* getHistoP( int ihP );
  TH1F* getHisto1( const std::string& name );
  TH2F* getHisto2( const std::string& name );
  TProfile* getHistoP( const std::string& name );
  int getHistoId1( const std::string& name );
  int getHistoId2( const std::string& name );

  TH1F* divideErr( const int h1, const int h2);
  TH1F* divideErr(TH1F* h1, TH1F* h2);

 private:
  mih1 theHistos1;
  mih2 theHistos2;
  mihP theHistosP;

  static HistoMgr* theInstance;

};


#endif
