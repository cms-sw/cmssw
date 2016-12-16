#include "Validation/MuonME0Validation/interface/ME0BaseValidation.h"
#include "DataFormats/Common/interface/Handle.h"

using namespace std;
ME0BaseValidation::ME0BaseValidation( const edm::ParameterSet& ps)
{
  nBinZR_ = ps.getUntrackedParameter<std::vector<double>>("nBinGlobalZR") ;
  RangeZR_ = ps.getUntrackedParameter< std::vector<double> >("RangeGlobalZR");
  nBinXY_ = ps.getUntrackedParameter<int>("nBinGlobalXY",200) ;

  regionLabel.push_back("-1");
  regionLabel.push_back("1" );

  layerLabel.push_back("1");
  layerLabel.push_back("2");
  layerLabel.push_back("3");
  layerLabel.push_back("4");
  layerLabel.push_back("5");
  layerLabel.push_back("6");

}


ME0BaseValidation::~ME0BaseValidation() {
}

MonitorElement* ME0BaseValidation::BookHistZR( DQMStore::IBooker& ibooker, const char* name, const char* label, unsigned int region_num, unsigned int layer_num) {
  string hist_name, hist_label;
  if ( layer_num == 0 || layer_num==1 || layer_num==2 || layer_num==3 || layer_num==4 || layer_num==5 || layer_num==6 ) {
    hist_name  = name+string("_zr_r") + regionLabel[region_num]+"_l"+layerLabel[layer_num];
    hist_label = label+string(" occupancy : region")+regionLabel[region_num]+" layer "+layerLabel[layer_num]+" "+" ; globalZ [cm]; globalR[cm]";
  }
  else {
    hist_name  = name+string("_zr_r") + regionLabel[region_num];
    hist_label = label+string(" occupancy : region")+regionLabel[region_num]+" ; globalZ [cm]; globalR[cm]";
  }
  int xbin = (int)nBinZR_[0];
  int ybin = (int)nBinZR_[1];
  double xmin = 0;
  double xmax = 0; 
  double ymin = 0;
  double ymax = 0;
  ymin = RangeZR_[ RangeZR_.size()/2 + 0];;
  ymax = RangeZR_[ RangeZR_.size()/2 + 1];;
  if ( region_num ==0 ) {
    xmin = -RangeZR_[1];
    xmax = -RangeZR_[0];
  }
  else {
    xmin = RangeZR_[0];
    xmax = RangeZR_[1];
  }
  return ibooker.book2D( hist_name, hist_label, xbin, xmin, xmax, ybin,ymin, ymax);
}

MonitorElement* ME0BaseValidation::BookHistXY( DQMStore::IBooker& ibooker, const char* name, const char* label, unsigned int region_num, unsigned int layer_num) {
  string hist_name, hist_label;
  if ( layer_num == 0 || layer_num==1 || layer_num==2 || layer_num==3 || layer_num==4 || layer_num==5 || layer_num==6 ) {
    hist_name  = name+string("_xy_r") + regionLabel[region_num]+"_l"+layerLabel[layer_num];
    hist_label = label+string(" occupancy : region")+regionLabel[region_num]+" layer "+layerLabel[layer_num]+" "+" ; globalX [cm]; globalY[cm]";
  }
  else {
    hist_name  = name+string("_xy_r") + regionLabel[region_num];
    hist_label = label+string(" occupancy : region")+regionLabel[region_num]+" ; globalX [cm]; globalY[cm]";
  } 
  return ibooker.book2D( hist_name, hist_label, nBinXY_, -200,200,nBinXY_,-200,200);
}

