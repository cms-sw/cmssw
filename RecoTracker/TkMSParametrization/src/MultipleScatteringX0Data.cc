#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringX0Data.h"

#include <iostream>
#include <string>

#include "TH2F.h"
#include "TFile.h"
#include "TKey.h"


MultipleScatteringX0Data::MultipleScatteringX0Data()
//: theData(0)
{
  std::cout << "MultipleScatteringX0Data initialisation,";
  std::string filename = fileName(); 
  std::cout <<" reading :"<<filename.c_str()<<"..."<<std::endl;
  //  pi_aida::Proxy_Store store(filename.c_str(),"XML", pi_aida::READONLY);
//   if (! store.isOpen() ) return;
//   try {
//     //   theData  = new HistoType(store.retrieve<HistoType>("x0data"));
//     std::cout << "..done, data title: "<< theData->title() << std::endl;
//   } 
//   catch (std::exception & e) { 
//     std::cout << endl
//          << "Error: ** MultipleScatteringX0Data** not initialised !!!"<<std::endl; 
//     //   theData = 0;
//   }
//   store.close();
}

MultipleScatteringX0Data::~MultipleScatteringX0Data()
{} //delete theData; }



std::string MultipleScatteringX0Data::fileName()
{
  //MP
  // ????
  //string defName="MultipleScatteringX0Data.aida";
//   string key="TrackerReco:TkMSParametrization:DataFile";
//   envUtil eU("ORCA_DATA_PATH","");
//   string path= eU.getEnv();
//   FileInPath mydata(path,"TrackerData/TkMaterial/"+defName);
//   string defValue = mydata() ? mydata.name() : defName;
//   SimpleConfigurable<string> name(defValue, key);
//   return name.value();
  return "";
} 

int MultipleScatteringX0Data::nBinsEta() const
{
  //  return theData ? theData->xAxis().bins() : 0;
  return 0;
}

float MultipleScatteringX0Data::minEta() const
{
  //  return theData ? theData->xAxis().lowerEdge() : 0.;
  return 0;
}

float MultipleScatteringX0Data::maxEta() const
{
  // return theData ? theData->xAxis().upperEdge() : 0.;
 return 0.;
}

float MultipleScatteringX0Data::sumX0atEta(float eta, float r) const
{return 0.;
}
// //   if (!theData) return 0.;

//   const double epsilon = 1.e-10;
//   eta = fabs(eta);

//   int ieta = theData->coordToIndexX(eta);
//   int irad = theData->coordToIndexY(r);

//   if (irad < theData->yAxis().bins()) { 
//     return theData->binHeight(ieta,irad);
//   } 
//   else {
//     float sumX0 = 0.;
//     for(int ir = theData->yAxis().bins()-1; ir >= 0; ir--) {
//       float val = theData->binHeight(ieta,ir);
//       if (val > sumX0+epsilon) sumX0 = val;
//     }
//     return sumX0;
//   }

