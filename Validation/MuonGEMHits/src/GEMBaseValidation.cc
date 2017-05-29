#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include <memory>
using namespace std;
GEMBaseValidation::GEMBaseValidation( const edm::ParameterSet& ps)
{
  nBinZR_ = ps.getUntrackedParameter< std::vector<double> >("nBinGlobalZR") ;
  RangeZR_ = ps.getUntrackedParameter< std::vector<double> >("RangeGlobalZR");
  nBinXY_ = ps.getUntrackedParameter< int >("nBinGlobalXY",360) ;

  regionLabel.push_back("-1");
  regionLabel.push_back("1" );


  layerLabel.push_back("1");
  layerLabel.push_back("2");
}

const GEMGeometry* GEMBaseValidation::initGeometry(edm::EventSetup const & iSetup) {
  const GEMGeometry* GEMGeometry_ = nullptr;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("MuonGEMBaseValidation") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return nullptr;
  }

  LogDebug("MuonBaseValidation") << "GEMGeometry_->regions().size() " << GEMGeometry_->regions().size() << "\n";
  LogDebug("MuonBaseValidation") << "GEMGeometry_->stations().size() " << GEMGeometry_->regions().front()->stations().size() << "\n";
  LogDebug("MuonBaseValidation") << "GEMGeometry_->superChambers().size() " << GEMGeometry_->superChambers().size() << "\n";
  LogDebug("MuonBaseValidation") << "GEMGeometry_->chambers().size() " << GEMGeometry_->chambers().size() << "\n";
  LogDebug("MuonBaseValidation") << "GEMGeometry_->etaPartitions().size() " << GEMGeometry_->etaPartitions().size() << "\n"; 

  nregion  = GEMGeometry_->regions().size();
  nstation = GEMGeometry_->regions().front()->stations().size() ;
  nstationForLabel = nstation;
  npart    = GEMGeometry_->chambers().front()->etaPartitions().size();

  return GEMGeometry_;
}

string GEMBaseValidation::getSuffixName(int region, int station, int layer){
  if ( region == -1 ) region =0 ;
  else if ( region >1 ) std::cout<<"Name)Alert! Region must be -1 or 1 : "<<region<<" "<<station<<" "<<layer<<std::endl;
  return string("_r")+regionLabel[region]+"_st"+getStationLabel(station)+"_l"+layerLabel[layer-1];
}
string GEMBaseValidation::getSuffixName(int region, int station){
  if ( region == -1 ) region =0 ;
  else if ( region >1 ) std::cout<<"Name)Alert! Region must be -1 or 1 : "<<region<<" "<<station<<std::endl;
  return string("_r")+regionLabel[region]+"_st"+getStationLabel(station);
}
string GEMBaseValidation::getSuffixName(int region){
  if ( region == -1 ) region =0 ;
  else if ( region >1 ) std::cout<<"Name)Alert! Region must be -1 or 1 : "<<region<<std::endl;
  return string("_r")+regionLabel[region];
}

string GEMBaseValidation::getSuffixTitle(int region, int station, int layer){
  if ( region == -1 ) region =0 ;
  else if ( region >1 ) std::cout<<"Title)Alert! Region must be -1 or 1 : "<<region<<" "<<station<<" "<<layer<<std::endl;
  return string("Region ")+regionLabel[region]+" Station "+getStationLabel(station)+" Layer "+layerLabel[layer-1];
}
string GEMBaseValidation::getSuffixTitle(int region, int station){
  if ( region == -1 ) region =0 ;
  else if ( region >1 ) std::cout<<"Title)Alert! Region must be -1 or 1 : "<<region<<" "<<station<<std::endl;
  return string("Region ")+regionLabel[region]+" Station "+getStationLabel(station);
}
string GEMBaseValidation::getSuffixTitle(int region){
  if ( region == -1 ) region =0 ;
  else if ( region >1 ) std::cout<<"Title)Alert! Region must be -1 or 1 : "<<region<<std::endl;
  return string("Region ")+regionLabel[region];
}

string GEMBaseValidation::getStationLabel(int i) {
  vector<string> stationLabel;
  if ( nstationForLabel == 2) { 
    string stationLabel[] = {"1","2"};
    return stationLabel[i-1];
  }
  else {
    std::cout<<"Something is wrong"<<std::endl;
    return "";
  }
}




GEMBaseValidation::~GEMBaseValidation() {
}

MonitorElement* GEMBaseValidation::getSimpleZR(DQMStore::IBooker& ibooker, TString title, TString histname) {
    std::vector<double> xbins_vector;
    double station1_xmin = RangeZR_[ 0 ];
    double station1_xmax = RangeZR_[ 1 ];
    double station2_xmin = RangeZR_[ 2 ];
    double station2_xmax = RangeZR_[ 3 ];

    for( double i= station1_xmin-1 ; i< station2_xmax+1; i=i+0.25  ) {
      if ( i > station1_xmax+1 && i<station2_xmin-1 ) continue; 
      xbins_vector.push_back(i);
    }
    TH2F* simpleZR_temp = new TH2F(title,histname, xbins_vector.size()-1, (double*)&xbins_vector[0], 50,120,330);
    MonitorElement* simpleZR = ibooker.book2D( histname, simpleZR_temp);
    return simpleZR;
}

MonitorElement* GEMBaseValidation::getDCEta(DQMStore::IBooker& ibooker, const GEMStation* station, TString title, TString histname ) {
  if( station->rings().front()->superChambers().size() == 0 ) {
    LogDebug("MuonBaseValidation")<<"+++ Error! can not get superChambers. Skip "<<getSuffixTitle(station->region(), station->station())<<" on "<<histname<<"\n";
    return nullptr;
  }

  int nXbins = station->rings().front()->nSuperChambers()*2;
  int nYbins = station->rings().front()->superChambers().front()->chambers().front()->nEtaPartitions();

  TH2F* dcEta_temp = new TH2F(title,histname,nXbins, 0, nXbins, nYbins, 1, nYbins+1);
  int idx = 0 ;

  for(unsigned int sCh = 1; sCh <= station->superChambers().size(); sCh++){
    for(unsigned int Ch = 1; Ch <= 2; Ch++){
      idx++;
      TString label = TString::Format("ch%d_la%d", sCh, Ch);
      dcEta_temp->GetXaxis()->SetBinLabel(idx, label.Data());
    }
  }
  MonitorElement* dcEta = ibooker.book2D( histname, dcEta_temp);
  return dcEta;
}  


MonitorElement* GEMBaseValidation::BookHistZR( DQMStore::IBooker& ibooker, const char* name, const char* label, unsigned int region_num, unsigned int station_num, unsigned int layer_num) {
  string hist_name, hist_title;
  if ( layer_num == 0 || layer_num==1 ) {
    hist_name  = name+string("_zr") + getSuffixName(region_num, station_num+1, layer_num+1);
    hist_title = label+string(" occupancy : region")+getSuffixTitle( region_num, station_num+1, layer_num+1)+" ; globalZ[cm] ; globalR[cm]";
  }
  else {
    hist_name  = name+string("_zr") + getSuffixName(region_num, station_num+1);
    hist_title = label+string(" occupancy : region")+getSuffixTitle( region_num, station_num+1)+" ; globalZ[cm] ; globalR[cm]";
  }
  LogDebug("GEMBaseValidation")<<hist_name<<" "<<hist_title<<std::endl;
  int xbin = (int)nBinZR_[station_num]; 
  int ybin = (int)nBinZR_[ nBinZR_.size()/2+station_num];
  double xmin = 0;
  double xmax = 0; 
  double ymin = 0;
  double ymax = 0;
  ymin = RangeZR_[ RangeZR_.size()/2 + station_num*2 + 0]; 
  ymax = RangeZR_[ RangeZR_.size()/2 + station_num*2 + 1]; 
  if ( region_num ==0 ) {
    xmin = -RangeZR_[ station_num*2 + 1];
    xmax = -RangeZR_[ station_num*2 + 0];
  }
  else {
    xmin = RangeZR_[ station_num*2 + 0];
    xmax = RangeZR_[ station_num*2 + 1];
  }
  return ibooker.book2D( hist_name, hist_title, xbin, xmin, xmax, ybin,ymin, ymax);
}

MonitorElement* GEMBaseValidation::BookHistXY( DQMStore::IBooker& ibooker, const char* name, const char* label, unsigned int region_num, unsigned int station_num, unsigned int layer_num) {
  string hist_name, hist_title;
  if ( layer_num == 0 || layer_num==1 ) {
    hist_name  = name+string("_xy") + getSuffixName( region_num, station_num+1, layer_num+1) ;
    hist_title = label+string(" occupancy : ")+getSuffixTitle( region_num, station_num+1, layer_num+1 )+ " ; globalX [cm]; globalY[cm]"; 
  }
  else {
    hist_name  = name+string("_xy") + getSuffixName( region_num, station_num+1);
    hist_title = label+string(" occupancy : region")+getSuffixTitle( region_num, station_num+1) +" ; globalX [cm]; globalY[cm]";
  } 
  return ibooker.book2D( hist_name, hist_title, nBinXY_, -360,360,nBinXY_,-360,360); 
}

