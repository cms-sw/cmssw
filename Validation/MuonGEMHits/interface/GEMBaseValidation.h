#ifndef GEMBaseValidation_H
#define GEMBaseValidation_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
class GEMBaseValidation : public DQMEDAnalyzer
{
public:
  explicit GEMBaseValidation( const edm::ParameterSet& ps );
  virtual ~GEMBaseValidation();
  virtual void analyze(const edm::Event& e, const edm::EventSetup&) = 0 ;
  MonitorElement* BookHistZR( DQMStore::IBooker &, const char* name, const char* label, unsigned int region_num, unsigned int station_num, unsigned int layer_num =99 ); 
  MonitorElement* BookHistXY( DQMStore::IBooker &, const char* name, const char* label, unsigned int region_num, unsigned int station_num, unsigned int layer_num =99 );
  TH2F* getSimpleZR() ; 
protected:
  std::vector< std::string > regionLabel;
  std::vector< std::string > layerLabel;
  std::vector< std::string > stationLabel;
  std::vector<double> nBinZR_;
  std::vector<double> RangeZR_;
  edm::EDGetToken InputTagToken_;
  int nBinXY_;
  bool detailPlot_;

private :
};

#endif
