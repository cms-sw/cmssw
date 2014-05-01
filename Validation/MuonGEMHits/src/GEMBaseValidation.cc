#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"

using namespace std;
GEMBaseValidation::GEMBaseValidation(DQMStore* dbe,
                                               const edm::InputTag & inputTag, const edm::ParameterSet& PlotRange)
{
  dbe_ = dbe;
  theInputTag = inputTag;
  plotRange_ = PlotRange;
	nBinZR = plotRange_.getUntrackedParameter< std::vector<double> >("nBinGlobalZR") ;
	RangeZR = plotRange_.getUntrackedParameter< std::vector<double> >("RangeGlobalZR");

  regionLabel.push_back("-1");
  regionLabel.push_back("1" );

  stationLabel.push_back("1");
  stationLabel.push_back("2_short");
  stationLabel.push_back("2_long");

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

  int xbin = (int)nBinZR[station_num]; 
  int ybin = (int)nBinZR[ nBinZR.size()/2+station_num];
  double xmin = 0;
	double xmax = 0; 
	double ymin = 0;
	double ymax = 0;
	ymin = RangeZR[ RangeZR.size()/2 + station_num*2 + 0]; 
	ymax = RangeZR[ RangeZR.size()/2 + station_num*2 + 1]; 
	if ( region_num ==0 ) {
		xmin = -RangeZR[ station_num*2 + 1];
		xmax = -RangeZR[ station_num*2 + 0];
  }
  else {
		xmin = RangeZR[ station_num*2 + 0];
		xmax = RangeZR[ station_num*2 + 1];
  }
  return dbe_->book2D( hist_name, hist_label, xbin, xmin, xmax, ybin,ymin, ymax);
}
