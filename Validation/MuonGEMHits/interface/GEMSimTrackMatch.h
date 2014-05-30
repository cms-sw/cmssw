#ifndef GEMSimTrackMatch_H
#define GEMSimTrackMatch_H

#include "Validation/MuonGEMHits/interface/GEMTrackMatch.h"


class GEMSimTrackMatch : public GEMTrackMatch 
{
public:
  GEMSimTrackMatch(DQMStore* , std::string , edm::ParameterSet);
  ~GEMSimTrackMatch();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void bookHisto(const GEMGeometry* geom);

 private:

  MonitorElement* track_eta[5][2];

  MonitorElement* track_phi[5][2];

  MonitorElement* gem_lx[5][2];

  MonitorElement* gem_ly[5][2];

};

#endif
