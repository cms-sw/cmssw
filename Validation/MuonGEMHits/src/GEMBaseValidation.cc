#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
GEMBaseValidation::GEMBaseValidation( const edm::ParameterSet& ps)
{
  nBinZR_ = ps.getUntrackedParameter< std::vector<double> >("nBinGlobalZR") ;
  RangeZR_ = ps.getUntrackedParameter< std::vector<double> >("RangeGlobalZR");
  nBinXY_ = ps.getUntrackedParameter< int >("nBinGlobalXY",360) ;

  regionLabel.push_back("-1");
  regionLabel.push_back("1" );

  stationLabel.push_back("1");
  stationLabel.push_back("2s");
  stationLabel.push_back("2");

  layerLabel.push_back("1");
  layerLabel.push_back("2");
}


GEMBaseValidation::~GEMBaseValidation() {
}

TH2F* GEMBaseValidation::getSimpleZR() {
    std::vector<double> xbins_vector;
    for( int i= 550 ; i< 820; i++  ) {
      xbins_vector.push_back(i);
      if ( i > 580 && i<780 )  i = 780; 
    }
    TH2F* simpleZR_templ = new TH2F("","", xbins_vector.size()-1, (double*)&xbins_vector[0], 50,100,330);
    return simpleZR_templ;
}

MonitorElement* GEMBaseValidation::BookHistZR( DQMStore::IBooker& ibooker, const char* name, const char* label, unsigned int region_num, unsigned int station_num, unsigned int layer_num) {
  string hist_name, hist_label;
  if ( layer_num == 0 || layer_num==1 ) {
    hist_name  = name+string("_zr_r") + regionLabel[region_num]+"_st"+stationLabel[station_num]+"_l"+layerLabel[layer_num];
    hist_label = label+string(" occupancy : region")+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num]+" "+" ; globalZ [cm]; globalR[cm]";
  }
  else {
    hist_name  = name+string("_zr_r") + regionLabel[region_num]+"_st"+stationLabel[station_num];
    hist_label = label+string(" occupancy : region")+regionLabel[region_num]+" station "+stationLabel[station_num]+" "+" ; globalZ [cm]; globalR[cm]";
  }
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
  return ibooker.book2D( hist_name, hist_label, xbin, xmin, xmax, ybin,ymin, ymax);
}

MonitorElement* GEMBaseValidation::BookHistXY( DQMStore::IBooker& ibooker, const char* name, const char* label, unsigned int region_num, unsigned int station_num, unsigned int layer_num) {
  string hist_name, hist_label;
  if ( layer_num == 0 || layer_num==1 ) {
    hist_name  = name+string("_xy_r") + regionLabel[region_num]+"_st"+stationLabel[station_num]+"_l"+layerLabel[layer_num];
    hist_label = label+string(" occupancy : region")+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num]+" "+" ; globalX [cm]; globalY[cm]";
  }
  else {
    hist_name  = name+string("_xy_r") + regionLabel[region_num]+"_st"+stationLabel[station_num];
    hist_label = label+string(" occupancy : region")+regionLabel[region_num]+" station "+stationLabel[station_num]+" "+" ; globalX [cm]; globalY[cm]";
  } 
  return ibooker.book2D( hist_name, hist_label, nBinXY_, -360,360,nBinXY_,-360,360); 
}

