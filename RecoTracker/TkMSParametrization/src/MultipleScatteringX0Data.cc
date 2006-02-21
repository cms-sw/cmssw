#include "RecoTracker/TkMSParametrization/src/MultipleScatteringX0Data.h"

#include <iostream>
#include <string>

// #include "TH2.h"
// #include "TFile.h"
// #include "TKey.h"


MultipleScatteringX0Data::MultipleScatteringX0Data()
// : theData(0)
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

/*
MultipleScatteringX0Data::MultipleScatteringX0Data()
{
  string filename = fileName(); 
  TFile * file = new TFile(filename.c_str(),"READ");
  if (!file || !file->IsOpen()) {
     cout << "** MultipleScatteringX0Data ** problem with data file: "
           << filename <<endl; 
     return;
  } else {
    cout << " ** MultipleScatteringX0Data ** file: "<<filename<<" opened"<<endl;
  }
  TH2 * data = dynamic_cast<TH2F*> (file->GetKey("h100")->ReadObj());
  if (!data)  {
    cout << " ** MultipleScatteringX0Data ** file: "
         << filename 
         <<" <-- no data found!!!"<<endl;
  }
  pi_aida::Histogram2D * theData = 
      new pi_aida::Histogram2D("h100","x0 data (sum_x0 vs |eta|)", 
          data->GetNbinsX(), 
          data->GetXaxis()->GetXmin(), data->GetXaxis()->GetXmax(),
          data->GetNbinsY(), 
          data->GetYaxis()->GetXmin(), data->GetYaxis()->GetXmax());

  for (int ix = 1; ix <= data->GetNbinsX(); ix++) {
    double x = data->GetXaxis()->GetBinCenter(ix);
    for (int iy = 1; iy <= data->GetNbinsY(); iy++) {
      double y = data->GetYaxis()->GetBinCenter(iy);
      double v = data->GetBinContent(ix,iy);
      theData->fill(x,y,v);
    }
  } 
  data->Delete();
  file->Delete();
  pi_aida::Proxy_Store store("MultipleScatteringX0Data.aida","XML",
      pi_aida::RECREATE);
  store.write(theData,"x0data");
  cout << "objects: "<<endl;
  std::vector<std::string> ss = store.listObjectNames();
  std::vector<std::string>::const_iterator is;
  for (is = ss.begin(); is != ss.end(); is++) cout << *is << endl;
  cout << "types: "<<endl;
  ss = store.listObjectTypes();
  for (is = ss.begin(); is != ss.end(); is++) cout << *is << endl;
  store.close();
}
*/


std::string MultipleScatteringX0Data::fileName()
{
  //string defName="MultipleScatteringX0Data.root";
//   string defName="MultipleScatteringX0Data.aida";
//   string key="TrackerReco:TkMSParametrization:DataFile";
//   envUtil eU("ORCA_DATA_PATH","");
//   string path= eU.getEnv();
//   FileInPath mydata(path,"TrackerData/TkMaterial/"+defName);
//   string defValue = mydata() ? mydata.name() : defName;
//   SimpleConfigurable<string> name(defValue, key);
//   return name.value();
  return "pippo";
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

