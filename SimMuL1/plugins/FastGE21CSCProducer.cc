/**\class FastGE21CSCProducer

 Description:

 Producer for quick studies for how GE2/1 would affect LCT stubs in ME2/1.
 It reads in collection of LCTs (after MPC sorting)
 and writes them back into the event with the simulated deltaPhi to GE2/1 stored in ME2/1 stubs.

 Original Author:  "Vadim Khotilovich"
 $Id: $
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
//#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
//#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include "GEMCode/GEMValidation/src/SimTrackMatchManager.h"

#include "TTree.h"
#include "TLinearFitter.h"

#include <iomanip>
#include <memory>
#include <tuple>

using namespace std;
using namespace matching;


// CSC chamber types, according to CSCDetId::iChamberType()
enum {CSC_ALL = 0, CSC_ME1a, CSC_ME1b, CSC_ME12, CSC_ME13,
      CSC_ME21, CSC_ME22, CSC_ME31, CSC_ME32, CSC_ME41, CSC_ME42};


class FastGE21CSCProducer : public edm::EDProducer
{
public:

  explicit FastGE21CSCProducer(const edm::ParameterSet&);

  ~FastGE21CSCProducer() {}
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  
  virtual void beginRun(edm::Run&, edm::EventSetup const&);

  virtual void produce(edm::Event&, const edm::EventSetup&);


  void processStubs4SimTrack(map<unsigned int, vector<CSCCorrelatedLCTDigi> >& stubs, SimTrackMatchManager& match);

  bool isSimTrackGood(const SimTrack &t);

  enum {CSC_ME21 = 5}; // CSC chamber type for ME2/1

  edm::ParameterSet cfg_;
  std::string simInputLabel_;
  edm::InputTag lctInput_;
  std::string productInstanceName_;
  float minPt_;
  int cscType_;
  double zOddGE21_;
  double zEvenGE21_;
  int verbose_;
  bool createNtuple_;

  std::unique_ptr<TLinearFitter> fitterXZ_;
  std::unique_ptr<TLinearFitter> fitterYZ_;

  const CSCGeometry* csc_geo_;

  TTree* tree_;

  void bookNtuple();
};


FastGE21CSCProducer::FastGE21CSCProducer(const edm::ParameterSet& ps)
: cfg_(ps.getParameterSet("simTrackMatching"))
, simInputLabel_(ps.getUntrackedParameter<string>("simInputLabel", "g4SimHits"))
, lctInput_(ps.getUntrackedParameter<edm::InputTag>("lctInput", edm::InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED")))
, productInstanceName_(ps.getUntrackedParameter<string>("productInstanceName", "FastGE21"))
, minPt_(ps.getUntrackedParameter<double>("minPt", 4.5))
, cscType_(ps.getUntrackedParameter<int>("cscType", CSC_ME21 )) // usually want to use it for ME2/1, but keep some generality
, zOddGE21_(ps.getUntrackedParameter<double>("zOddGE21", 780.))
, zEvenGE21_(ps.getUntrackedParameter<double>("zEvenGE21", 775.))
, verbose_(ps.getUntrackedParameter<int>("verbose", 0))
, createNtuple_(ps.getUntrackedParameter<bool>("createNtuple", true))
, fitterXZ_(new TLinearFitter(1, "pol1"))
, fitterYZ_(new TLinearFitter(1, "pol1"))
{
  if (createNtuple_) bookNtuple();

  fitterXZ_->StoreData(1);
  fitterYZ_->StoreData(1);

  produces<CSCCorrelatedLCTDigiCollection>(productInstanceName_);
}


void FastGE21CSCProducer::beginRun(edm::Run &iRun, edm::EventSetup const &iSetup)
{
  //
}


bool FastGE21CSCProducer::isSimTrackGood(const SimTrack &t)
{
  // SimTrack selection
  if (t.noVertex()) return false;
  if (t.noGenpart()) return false;
  if (std::abs(t.type()) != 13) return false; // only interested in direct muon simtracks
  if (t.momentum().pt() < minPt_) return false;
  float eta = std::abs(t.momentum().eta());
  if (eta > 2.4 || eta < 1.55) return false; // no GEMs could be in such eta
  return true;
}


void FastGE21CSCProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  edm::ESHandle<CSCGeometry> csc_g;
  es.get<MuonGeometryRecord>().get(csc_g);
  csc_geo_ = &*csc_g;

  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;

  ev.getByLabel(simInputLabel_, sim_tracks);
  ev.getByLabel(simInputLabel_, sim_vertices);
  const edm::SimVertexContainer & sim_vert = *sim_vertices.product();

  // pick up the stubs from event and store them into a new mutable collection
  edm::Handle<CSCCorrelatedLCTDigiCollection> ev_stubs;
  ev.getByLabel(lctInput_, ev_stubs);

  map<unsigned int, vector<CSCCorrelatedLCTDigi> > mutable_stubs;
  for(auto detIt = ev_stubs->begin() ; detIt != ev_stubs->end(); ++detIt)
  {
    unsigned int d = (*detIt).first.rawId();
    mutable_stubs[d] = vector<CSCCorrelatedLCTDigi>();
    const auto& range = (*detIt).second;
    for (auto stubIt = range.first; stubIt != range.second; ++stubIt)
    {
      mutable_stubs[d].push_back(*stubIt);
    }
  }

  for (auto& t: *sim_tracks.product())
  {
    if (!isSimTrackGood(t)) continue;

    // match hits, digis and LCTs to this SimTrack
    SimTrackMatchManager match(t, sim_vert[t.vertIndex()], cfg_, ev, es);

    processStubs4SimTrack(mutable_stubs, match);
  }

  // pack modified stubs into a CSCCorrelatedLCTDigiCollection and store it in event 
  std::auto_ptr<CSCCorrelatedLCTDigiCollection> new_stubs(new CSCCorrelatedLCTDigiCollection);
  for (auto dstubs: mutable_stubs)
  {
    CSCDetId id(dstubs.first);
    new_stubs->put(make_pair(dstubs.second.begin(), dstubs.second.end()), id);
  }
  ev.put(new_stubs, productInstanceName_);
}


void FastGE21CSCProducer::processStubs4SimTrack(map<unsigned int, vector<CSCCorrelatedLCTDigi> >& stubs, SimTrackMatchManager& match)
{
  const SimHitMatcher& match_sh = match.simhits();
  //const CSCStubMatcher& match_lct = match.cscStubs();
  const SimTrack &t = match_sh.trk();

  //ntupleRowInit();

  // data struct to keep SimHit-modeled stubs info:
  // map<chamberDetId, tuple<set<HS>, set<WG>, dPhi> >
  map<unsigned int, tuple<set<int>, set<int>, double> > model_stubs;

  auto csc_ch_ids = match_sh.chamberIdsCSC(cscType_);
  for(auto d: csc_ch_ids)
  {
    CSCDetId id(d);

    int nlayers = match_sh.nLayersWithHitsInSuperChamber(d);
    if (nlayers < 4) continue;

    bool odd = id.chamber() & 1;

    tuple<set<int>, set<int>, double> model_stub;

    // Symmetric form of line equation: (x - x0)/a = (y - y0)/b = (z - z0)/c
    // Only 4 of 6 parameters are independent, so with no loss of generality
    // we can set c = 1, and, e.g.,  z0 = zkey, where for zkey we would use position of chamber's key layer
    // We'll do two linear fits for x and y dependency on z (sine z's are fixed by detector positions):
    // x(z) = x0 + a *(z - z0) = xz0 + xz1*z, where xz0 = x0 - xz1*z0, xz1 = a
    // y(z) = y0 + b *(z - z0) = yz0 + yz1*z, where yz0 = y0 - yz1*z0, yz1 = b
    // we find xz0, xz1, yz0, yz1 from linear fits

    cout<<" hitXZ ";
    const auto& hits = match_sh.hitsInChamber(d);
    for (auto& h: hits)
    {
      auto hs_set = match_sh.hitStripsInDetId(h.detUnitId(), 1); // use single HS margin
      get<0>(model_stub).insert(hs_set.begin(), hs_set.end());
      auto wg_set = match_sh.hitWiregroupsInDetId(h.detUnitId(), 1); // use single WG margin
      get<1>(model_stub).insert(wg_set.begin(), wg_set.end());

      GlobalPoint gp = csc_geo_->idToDet(h.detUnitId())->surface().toGlobal(h.entryPoint());
      LocalPoint lp = csc_geo_->idToDet(id.chamberId())->surface().toLocal(gp);
      cout<< lp.x() <<" "<<gp.z()<<"  ";

      double z[1] = {gp.z()};
      fitterXZ_->AddPoint(z, gp.x()); // x(z)
      fitterYZ_->AddPoint(z, gp.y()); // x(z)
    }
    cout<<endl;
    fitterXZ_->Eval();
    fitterYZ_->Eval();

    double xz0  = fitterXZ_->GetParameter(0);
    //double xz0e = fitterXZ_->GetParError(0);
    double xz1  = fitterXZ_->GetParameter(1);
    //double xz1e = fitterXZ_->GetParError(1);
    double yz0  = fitterYZ_->GetParameter(0);
    //double yz0e = fitterYZ_->GetParError(0);
    double yz1  = fitterYZ_->GetParameter(1);
    //double yz1e = fitterYZ_->GetParError(1);

    fitterXZ_->ClearPoints();
    fitterYZ_->ClearPoints();

    // chamber trigger key layer's global position:
    CSCDetId key_id(id.endcap(), id.station(), id.ring(), id.chamber(), CSCConstants::KEY_CLCT_LAYER);
    GlobalPoint gp_key = csc_geo_->idToDet(key_id)->surface().toGlobal(LocalPoint(0.,0.,0.));

    // fitted SimHits stub position at key layer
    GlobalPoint gp_sh_key( xz0 + xz1 * gp_key.z(), yz0 + yz1 * gp_key.z(), gp_key.z() );

    // fitted SimHits stub projection to GEM
    double z_gem;
    if (odd) {
      if (id.endcap() == 1) z_gem = zOddGE21_;
      else                  z_gem = -zOddGE21_;
    }
    else {
      if (id.endcap() == 1) z_gem = zEvenGE21_;
      else                  z_gem = -zEvenGE21_;
    }
    GlobalPoint gp_sh_gem( xz0 + xz1 * z_gem, yz0 + yz1 * z_gem, z_gem );

    double dphi = deltaPhi(gp_sh_key.phi(), gp_sh_gem.phi());

    // global mean position of this track's simhits in the chamber
    auto gp_mean = match_sh.simHitsMeanPosition(hits);

    // just a printout so far
    cout<<" gp_sh "<<t.momentum().eta()<<" "<<t.momentum().pt()<<" "<<t.charge()<<" "
        <<odd<<" "<<id.chamber()<<" "<<gp_sh_key<<" "<<gp_mean<<" "<<gp_sh_gem<<"  "<< dphi <<endl;

    get<2>(model_stub) = dphi;
    if ( !get<0>(model_stub).empty() && !get<1>(model_stub).empty() ) model_stubs[d] = model_stub;
    else
    {
      cout<<"Strange: empty HW or WG sets HS="<< get<0>(model_stub).size()<<" WG="<< get<1>(model_stub).size()<<endl;
    }
  }

  // match SimHit-modeled stubs to real LCT stubs and update real stub's dphi
  for (auto &model_stub: model_stubs)
  {
    auto d = model_stub.first;
    auto hs_set = get<0>(model_stub.second);
    auto wg_set = get<1>(model_stub.second);
    auto dphi   = get<2>(model_stub.second);

    CSCDetId id(d);
    int hs_min = *hs_set.begin();
    int hs_max = *hs_set.rbegin();
    int wg_min = *wg_set.begin();
    int wg_max = *wg_set.rbegin();

    auto dstubs = stubs.find(d);
    if (dstubs == stubs.end()) continue;

    for (auto& stub: dstubs->second)
    {
      int wg = 1 + stub.getKeyWG(); // LCT halfstrip and wiregoup numbers start from 0
      int hs = 1 + stub.getStrip();
      if (hs < hs_min || hs > hs_max || wg < wg_min || wg > wg_max) continue;
      //float old_dphi = digiIt->getGEMDPhi();
      stub.setGEMDPhi(dphi);
    }
  }

  /*
  csc_ch_ids = match_lct.chamberIdsLCT(cscType_);
  for(auto d: csc_ch_ids)
  {
    CSCDetId id(d);
    //bool odd = id.chamber() & 1;
    //auto lct = match_lct.lctInChamber(d);
    //auto gp = match_lct.digiPosition(lct);
    //int bx = digi_bx(lct);
    //int hs = digi_channel(lct);
  }
  */

  //tree_->Fill();
}


void FastGE21CSCProducer::bookNtuple()
{
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void FastGE21CSCProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(FastGE21CSCProducer);
