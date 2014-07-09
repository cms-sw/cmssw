#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
GEMBaseValidation::GEMBaseValidation(DQMStore* dbe,
                                               edm::EDGetToken & InputTagToken, const edm::ParameterSet& PlotRange)
{
  dbe_ = dbe;
  inputToken_ = InputTagToken;
  plotRange_ = PlotRange;
  nBinZR_ = plotRange_.getUntrackedParameter< std::vector<double> >("nBinGlobalZR") ;
  RangeZR_ = plotRange_.getUntrackedParameter< std::vector<double> >("RangeGlobalZR");
  nBinXY_ = plotRange_.getUntrackedParameter< int >("nBinGlobalXY",360) ;
  GE11PhiBegin_ = plotRange_.getUntrackedParameter< double >("GE11PhiBegin",-5.) ;
  GE11PhiStep_ = plotRange_.getUntrackedParameter< double >("GE11PhiStep",-5.) ;

  regionLabel.push_back("-1");
  regionLabel.push_back("1" );

  stationLabel.push_back("1");
  stationLabel.push_back("2s");
  stationLabel.push_back("2l");

  layerLabel.push_back("1");
  layerLabel.push_back("2");
}


GEMBaseValidation::~GEMBaseValidation() {
}
void GEMBaseValidation::setGeometry(const GEMGeometry* geom)
{ 
    theGEMGeometry = geom;
}

MonitorElement* GEMBaseValidation::BookHistZR( const char* name, const char* label, unsigned int region_num, unsigned int station_num, unsigned int layer_num) {

  string hist_name  = name+string("_zr_r") + regionLabel[region_num]+"_st"+stationLabel[station_num]+"_l"+layerLabel[layer_num];
  string hist_label = label+string(" occupancy : region")+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num]+" "+" ; globalZ [cm]; globalR[cm]";

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
  return dbe_->book2D( hist_name, hist_label, xbin, xmin, xmax, ybin,ymin, ymax);
}

MonitorElement* GEMBaseValidation::BookHistXY( const char* name, const char* label, unsigned int region_num, unsigned int station_num, unsigned int layer_num) {

  string hist_name  = name+string("_xy_r") + regionLabel[region_num]+"_st"+stationLabel[station_num]+"_l"+layerLabel[layer_num];
  string hist_label = label+string(" occupancy : region")+regionLabel[region_num]+" station "+stationLabel[station_num]+" layer "+layerLabel[layer_num]+" "+" ; globalX [cm]; globalY[cm]";
  return dbe_->book2D( hist_name, hist_label, nBinXY_, -360,360,nBinXY_,-360,360); 
}
