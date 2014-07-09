#ifndef GEMBaseValidation_H
#define GEMBaseValidation_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
class GEMBaseValidation
{
public:
  GEMBaseValidation(DQMStore* dbe,
                         edm::EDGetToken& inputToken, const edm::ParameterSet& PlotRange );
  virtual ~GEMBaseValidation();
  void setGeometry(const GEMGeometry* geom);
  virtual void bookHisto(const GEMGeometry* geom) = 0 ;
  virtual void analyze(const edm::Event& e, const edm::EventSetup&) = 0 ;
  MonitorElement* BookHistZR( const char* name, const char* label, unsigned int region_num, unsigned int station_num, unsigned int layer_num =0 ); 
  MonitorElement* BookHistXY( const char* name, const char* label, unsigned int region_num, unsigned int station_num, unsigned int layer_num =0 ); 
protected:

  std::vector< std::string > regionLabel;
  std::vector< std::string > layerLabel;
  std::vector< std::string > stationLabel;

  DQMStore* dbe_;
  edm::EDGetToken inputToken_;
  const GEMGeometry* theGEMGeometry;
  edm::ParameterSet plotRange_;
  std::vector<double> nBinZR_;
  std::vector<double> RangeZR_;
  int nBinXY_;
  double GE11PhiBegin_; 
  double GE11PhiStep_; 

};

#endif
