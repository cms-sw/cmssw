#ifndef GEMSimTrackMatch_H
#define GEMSimTrackMatch_H

#include "Validation/MuonGEMHits/interface/GEMTrackMatch.h"


class GEMSimTrackMatch : public GEMTrackMatch 
{
public:
  GEMSimTrackMatch(DQMStore* , edm::EDGetToken&, edm::EDGetToken& , edm::ParameterSet);
  ~GEMSimTrackMatch();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void bookHisto(const GEMGeometry* geom);

 private:

  MonitorElement* track_eta[4];//[4][3][2];

  MonitorElement* track_phi[4];//[5][2];

  MonitorElement* gem_lx[3][2];

  MonitorElement* gem_ly[3][2];

  MonitorElement* sh_eta[4][3];

  MonitorElement* sh_phi[4][3];

	unsigned int nstation;
};

#endif
