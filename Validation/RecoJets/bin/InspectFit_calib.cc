#include <memory>
#include <string>
#include <fstream>
#include <iostream>

#include "Validation/RecoJets/interface/RootSystem.h"
#include "Validation/RecoJets/interface/RootHistograms.h"
#include "Validation/RecoJets/interface/RootPostScript.h"
#include "Validation/RecoJets/interface/FitHist.h"
#include "Validation/RecoJets/bin/NiceStyle.cc"

using namespace std;

class TopInspectFit : public FitHist {
 public:
  TopInspectFit():FitHist(false){};
  ~TopInspectFit(){};
  virtual void readConfig( std::string );
};


void TopInspectFit::readConfig( std::string name )
{
  ConfigFile cfg( name, "=", "$" );  
  configBlockIO  ( cfg );
  configBlockHist( cfg );
  configBlockFit ( cfg );
}

int main(int argc, char* argv[])
{
  setNiceStyle();
  gStyle->SetOptStat( 0 );
  gStyle->SetOptFit ( 0 );

  gStyle->SetStatColor(0);
  gStyle->SetStatBorderSize(0);
  gStyle->SetStatX(0.93);
  gStyle->SetStatY(0.93);
  gStyle->SetStatW(0.18);
  gStyle->SetStatH(0.18);

  if( argc<2 ){
    std::cerr << "ERROR:" 
	 << " Missing argument" << std::endl;
    return 1;
  }

  TopInspectFit plots;
  try{
    plots.readConfig( argv[1] );
    plots.loadHistograms();

    //depending on style draw ps/eps/jpg
    if( !strcmp(plots.writeAs().c_str(), "ps") ){
      plots.fitAndDrawPs();
    } else if( !strcmp(plots.writeAs().c_str(), "eps") ){
      plots.fitAndDrawEps();
    } else{
      std::cerr << "ERROR:"
	   << " Unknown file format requested: "
	   << plots.writeAs() << std::endl; 
      return -1;
    }
    plots.fillTargetHistograms();
    plots.writeFitOutput();
  }
  catch(char* str){
    std::cerr << "ERROR: " << str << std::endl;
    return 1;
  }
  catch(...){
    std::cerr << "ERROR: this one is new...";
      return 1;
  }
  std::cout << "works " << "thanx and GoodBye " << std::endl; 
  return 0;
}
