#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/TotemRPReco/interface/TotemRPCluster.h"
#include "DataFormats/TotemRPReco/interface/TotemRPRecHit.h"
#include "DataFormats/TotemRPReco/interface/RPDetTrigger.h"
#include "DataFormats/TotemRPReco/interface/RPTimingDetectorHit.h"

#include <vector>

namespace {
  namespace {
	TotemRPRecHit rp_reco_hit;
    edm::DetSet<TotemRPRecHit> ds_rp_reco_hit;
    edm::DetSetVector<TotemRPRecHit> dsv_rp_reco_hit;
    std::vector<edm::DetSet<TotemRPRecHit> > sv_dsw_rp_reco_hit;
    edm::Wrapper<edm::DetSetVector<TotemRPRecHit> > w_dsv_rp_reco_hit;
	std::vector<TotemRPRecHit> sv_rp_reco_hit;
	std::vector<const TotemRPRecHit*> sv_cp_rp_reco_hit;
    
	// TODO: these needed?
    std::pair<__gnu_cxx::__normal_iterator<const TotemRPRecHit*,std::vector<TotemRPRecHit> >,__gnu_cxx::__normal_iterator<const TotemRPRecHit*,std::vector<TotemRPRecHit> > > pni;
    __gnu_cxx::__normal_iterator<const TotemRPRecHit*,std::vector<TotemRPRecHit> > d1;
    
    RPDetTrigger rp_str_tri;
    edm::DetSet<RPDetTrigger> ds_rp_str_tri;
    std::vector<RPDetTrigger> vec_rp_str_tri;
    std::vector<edm::DetSet<RPDetTrigger> > vec_ds_rp_str_tri;
    edm::DetSetVector<RPDetTrigger> dsv_rp_str_tri;
    edm::Wrapper<edm::DetSet<RPDetTrigger> > wds_rp_str_tri;
    edm::Wrapper<edm::DetSetVector<RPDetTrigger> > wdsv_rp_str_tri;

    TotemRPCluster dc;
    edm::DetSet<TotemRPCluster> dsdc;
    std::vector<TotemRPCluster> svdc;
    std::vector<edm::DetSet<TotemRPCluster> > svdsdc;
    edm::DetSetVector<TotemRPCluster> dsvdc;
    edm::Wrapper<edm::DetSetVector<TotemRPCluster> > wdsvdc;

    RPTimingDetectorHit tdh;
    std::vector<RPTimingDetectorHit> vec_tdh;
    edm::Wrapper<std::vector<RPTimingDetectorHit> > ew_vec_tdh;
  }
}
