#include "Validation/MuonGEMHits/interface/GEMSimTrackMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <TMath.h>
#include <TH1F.h>

GEMTrackMatch::GEMTrackMatch(DQMStore* dbe, std::string simInputLabel , edm::ParameterSet cfg)
{
   cfg_= cfg; 
   simInputLabel_= simInputLabel;
   dbe_= dbe;
}


GEMTrackMatch::~GEMTrackMatch() {
}

bool GEMTrackMatch::isSimTrackGood(const SimTrack &t)
{
  // SimTrack selection
  if (t.noVertex())   return false; 
  if (t.noGenpart()) return false;
  if (std::abs(t.type()) != 13) return false; // only interested in direct muon simtracks
  if (t.momentum().pt() < minPt_ ) return false;
  const float eta(std::abs(t.momentum().eta()));
  if (eta > maxEta_ || eta < minEta_ ) return false; // no GEMs could be in such eta
  return true;
}


void GEMTrackMatch::buildLUT()
{
  const int maxChamberId_ = theGEMGeometry->regions()[0]->stations()[0]->superChambers().size();
  edm::LogInfo("GEMTrackMatch")<<"max chamber "<<maxChamberId_<<"\n";
  std::vector<int> pos_ids;
  pos_ids.push_back(GEMDetId(1,1,1,1,maxChamberId_,2).rawId());

  std::vector<int> neg_ids;
  neg_ids.push_back(GEMDetId(-1,1,1,1,maxChamberId_,2).rawId());

  // VK: I would really suggest getting phis from GEMGeometry
  
  std::vector<float> phis;
  phis.push_back(0.);
  for(int i=1; i<maxChamberId_+1; ++i)
  {
    pos_ids.push_back(GEMDetId(1,1,1,1,i,2).rawId());
    neg_ids.push_back(GEMDetId(-1,1,1,1,i,2).rawId());
    phis.push_back(i*10.);
  }
  positiveLUT_ = std::make_pair(phis,pos_ids);
  negativeLUT_ = std::make_pair(phis,neg_ids);
}


void GEMTrackMatch::setGeometry(const GEMGeometry* geom)
{
  theGEMGeometry = geom;
  //Must use roll 2 as first eta partition.
  const auto top_chamber = static_cast<const GEMEtaPartition*>(theGEMGeometry->idToDetUnit(GEMDetId(1,1,1,1,1,2))); 
  const int nEtaPartitions(theGEMGeometry->chamber(GEMDetId(1,1,1,1,1,2))->nEtaPartitions());
  const auto bottom_chamber = static_cast<const GEMEtaPartition*>(theGEMGeometry->idToDetUnit(GEMDetId(1,1,1,1,1,nEtaPartitions)));
  const float top_half_striplength = top_chamber->specs()->specificTopology().stripLength()/2.;
  const float bottom_half_striplength = bottom_chamber->specs()->specificTopology().stripLength()/2.;
  const LocalPoint lp_top(0., top_half_striplength, 0.);
  const LocalPoint lp_bottom(0., -bottom_half_striplength, 0.);
  const GlobalPoint gp_top = top_chamber->toGlobal(lp_top);
  const GlobalPoint gp_bottom = bottom_chamber->toGlobal(lp_bottom);

  radiusCenter_ = (gp_bottom.perp() + gp_top.perp())/2.;
  chamberHeight_ = gp_top.perp() - gp_bottom.perp();

  buildLUT();
}  


std::pair<int,int> GEMTrackMatch::getClosestChambers(int region, float phi)
{
  const int maxChamberId_ = theGEMGeometry->regions()[0]->stations()[0]->superChambers().size();
  auto& phis(positiveLUT_.first);
  auto upper = std::upper_bound(phis.begin(), phis.end(), phi);
  auto& LUT = (region == 1 ? positiveLUT_.second : negativeLUT_.second);
  return std::make_pair(LUT.at(upper - phis.begin()), (LUT.at((upper - phis.begin() + 1)%maxChamberId_)));
}


