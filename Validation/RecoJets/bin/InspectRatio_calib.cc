#include <memory>
#include <string>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iostream>

#include "Validation/RecoJets/interface/RootSystem.h"
#include "Validation/RecoJets/interface/RootHistograms.h"
#include "Validation/RecoJets/interface/RootPostScript.h"
#include "Validation/RecoJets/interface/ManipHist.h"
#include "Validation/RecoJets/bin/NiceStyle.cc"

using namespace std;

class TopInspectRatio : public ManipHist {
 public:
  TopInspectRatio(){};
  ~TopInspectRatio(){};
  virtual void readConfig( std::string );
  void writeOutput(){CompHist::writeOutput(kFirstOnly);};
};

void TopInspectRatio::readConfig( std::string name )
{
  ConfigFile cfg( name, "=", "$" );  
  configBlockIO    ( cfg );
  configBlockHist  ( cfg );
  configBlockDivide( cfg );
}

int main(int argc, char* argv[])
{
  setNiceStyle();
  gStyle->SetOptStat( 0 );

  if( argc<2 ){
    std::cerr << "ERROR:" 
	 << " Missing argument" << std::endl;
    return 1;
  }

  TopInspectRatio plots;
  try{
    plots.readConfig( argv[1] );
    plots.loadHistograms();
    
    //depending on style draw ps/eps/jpg
    if( !strcmp(plots.writeAs().c_str(), "ps") ){
      plots.divideAndDrawPs();
    } else if( !strcmp(plots.writeAs().c_str(), "eps") ){
      plots.divideAndDrawEps();
    } else{
      std::cerr << "ERROR:"
	   << " Unknown file format requested: "
	   << plots.writeAs() << std::endl; 
      return -1;
    }
    plots.writeOutput();
  }
  catch(char* str){
    std::cerr << "ERROR: " << str << std::endl;
    return 1;
  }
  catch(...){
    std::cerr << "ERROR: this one is new...";
      return 1;
  }
  std::cout << "Thanx and GoodBye " << std::endl;
  return 0;
}
