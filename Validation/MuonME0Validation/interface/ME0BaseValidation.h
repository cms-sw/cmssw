#ifndef ME0BaseValidation_H
#define ME0BaseValidation_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

class ME0BaseValidation : public DQMEDAnalyzer
{
public:
  explicit ME0BaseValidation( const edm::ParameterSet& ps );
  virtual ~ME0BaseValidation();
  virtual void analyze(const edm::Event& e, const edm::EventSetup&) = 0 ;
  MonitorElement* BookHistZR( DQMStore::IBooker &, const char* name, const char* label, unsigned int region_num, unsigned int layer_num =99 ); 
  MonitorElement* BookHistXY( DQMStore::IBooker &, const char* name, const char* label, unsigned int region_num, unsigned int layer_num =99 ); 
protected:
  std::vector< std::string > regionLabel;
  std::vector< std::string > layerLabel;
  std::vector<double> nBinZR_;
  std::vector<double> RangeZR_;
  edm::EDGetToken InputTagToken_;
  int nBinXY_;

private :
};

#endif
