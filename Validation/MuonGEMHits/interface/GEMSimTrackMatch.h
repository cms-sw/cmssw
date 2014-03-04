#ifndef GEMSimTrackMatch_H
#define GEMSimTrackMatch_H

#include "Validation/MuonGEMHits/interface/GEMTrackMatch.h"


class GEMSimTrackMatch : public GEMTrackMatch 
{
public:
  GEMSimTrackMatch(DQMStore* , std::string , edm::ParameterSet);
  ~GEMSimTrackMatch();
  void analyze(const edm::Event& e, const edm::EventSetup&);
  void bookHisto();

 private:

  MonitorElement* track_eta;
  MonitorElement* track_eta_l1;
  MonitorElement* track_eta_l2;
  MonitorElement* track_eta_l1or2;
  MonitorElement* track_eta_l1and2;

  MonitorElement* track_phi;
  MonitorElement* track_phi_l1;
  MonitorElement* track_phi_l2;
  MonitorElement* track_phi_l1or2;
  MonitorElement* track_phi_l1and2;

  MonitorElement* gem_lx_even;
  MonitorElement* gem_lx_even_l1;
  MonitorElement* gem_lx_even_l2;
  MonitorElement* gem_lx_even_l1or2;
  MonitorElement* gem_lx_even_l1and2;

  MonitorElement* gem_ly_even;
  MonitorElement* gem_ly_even_l1;
  MonitorElement* gem_ly_even_l2;
  MonitorElement* gem_ly_even_l1or2;
  MonitorElement* gem_ly_even_l1and2;

  MonitorElement* gem_lx_odd;
  MonitorElement* gem_lx_odd_l1;
  MonitorElement* gem_lx_odd_l2;
  MonitorElement* gem_lx_odd_l1or2;
  MonitorElement* gem_lx_odd_l1and2;

  MonitorElement* gem_ly_odd;
  MonitorElement* gem_ly_odd_l1;
  MonitorElement* gem_ly_odd_l2;
  MonitorElement* gem_ly_odd_l1or2;
  MonitorElement* gem_ly_odd_l1and2;

};

#endif
