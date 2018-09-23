#ifndef TestHistoMgr_h
#define TestHistoMgr_h 1

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TFile.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <map>
#include <string>
#include <memory>

typedef std::map< int, TH1F* > mih1;
typedef std::map< int, TH2F* > mih2;
typedef std::map< int, TProfile* > mihp1;
typedef std::map< int, TProfile2D* > mihp2;


class TestHistoMgr 
{
public:

  TestHistoMgr();
  ~TestHistoMgr();
  void save( const std::string& name );
  void openSecondFile( const std::string& name );

  void printComparisonResult( int ih );

  bool addHisto1( TH1F* ih );
  bool addHisto2( TH2F* ih );
  bool addHistoProf1( TProfile* ih );
  bool addHistoProf2( TProfile2D* ih );

  TH1F* getHisto1( int ih );
  TH2F* getHisto2( int ih );
  TProfile* getHistoProf1( int ih );
  TProfile2D* getHistoProf2( int ih );

  TH1F* getHisto1FromSecondFile( const char* hnam );

 private:
  mih1 theHistos1;
  mih2 theHistos2;
  mihp1 theHistoProfs1;
  mihp2 theHistoProfs2;

  std::unique_ptr<TFile> theFileRef;

};


#endif
