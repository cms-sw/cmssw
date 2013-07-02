/**\class FastGEMCSCProducer

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
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
//#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
//#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "GEMCode/GEMValidation/src/SimTrackMatchManager.h"
#include "GEMCode/SimMuL1/interface/FastGEMCSCBuilder.h"

#include "CLHEP/Random/RandomEngine.h"

#include <iomanip>
#include <memory>
#include <tuple>

using namespace std;
using namespace matching;


// CSC chamber types, according to CSCDetId::iChamberType()
enum {CSC_ALL = 0, CSC_ME1a, CSC_ME1b, CSC_ME12, CSC_ME13,
      CSC_ME21, CSC_ME22, CSC_ME31, CSC_ME32, CSC_ME41, CSC_ME42};


class FastGEMCSCProducer : public edm::EDProducer
{
public:

  explicit FastGEMCSCProducer(const edm::ParameterSet&);

  ~FastGEMCSCProducer() {}
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
private:
  
  virtual void beginRun(edm::Run&, edm::EventSetup const&);

  virtual void produce(edm::Event&, const edm::EventSetup&);

  void processStubs4SimTrack(map<unsigned int, vector<CSCCorrelatedLCTDigi> >& stubs, SimTrackMatchManager& match);

  bool isSimTrackGood(const SimTrack &t);

  edm::ParameterSet cfg_;
  std::string simInputLabel_;
  edm::InputTag lctInput_;
  std::string productInstanceName_;
  float minPt_;
  float minEta_, maxEta_;
  bool usePropagatedDPhi_;
  int verbose_;

  const CSCGeometry* csc_geo_;

  std::unique_ptr<FastGEMCSCBuilder> builder_;
};


FastGEMCSCProducer::FastGEMCSCProducer(const edm::ParameterSet& ps)
: cfg_(ps.getParameterSet("simTrackMatching"))
, simInputLabel_(ps.getUntrackedParameter<string>("simInputLabel", "g4SimHits"))
, lctInput_(ps.getUntrackedParameter<edm::InputTag>("lctInput", edm::InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED")))
, productInstanceName_(ps.getUntrackedParameter<string>("productInstanceName", "FastGEM"))
, minPt_(ps.getUntrackedParameter<double>("minPt", 4.5))
, minEta_(ps.getUntrackedParameter<double>("minEta", 1.55))
, maxEta_(ps.getUntrackedParameter<double>("maxEta", 2.4))
, usePropagatedDPhi_(ps.getUntrackedParameter<bool>("usePropagatedDPhi", true))
, verbose_(ps.getUntrackedParameter<int>("verbose", 0))
{
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable())
  {
   throw cms::Exception("Configuration")
     << "FastGEMCSCProducer::FastGEMCSCProducer() - RandomNumberGeneratorService is not present in configuration file.\n"
     << "Add the service in the configuration file or remove the modules that require it.";
  }
  CLHEP::HepRandomEngine& engine = rng->getEngine();

  builder_.reset(new FastGEMCSCBuilder(ps, engine));

  produces<CSCCorrelatedLCTDigiCollection>(productInstanceName_);
}


void FastGEMCSCProducer::beginRun(edm::Run &iRun, edm::EventSetup const &iSetup)
{
  //
}


bool FastGEMCSCProducer::isSimTrackGood(const SimTrack &t)
{
  // SimTrack selection
  if (t.noVertex()) return false;
  if (t.noGenpart()) return false;
  if (std::abs(t.type()) != 13) return false; // only interested in direct muon simtracks
  if (t.momentum().pt() < minPt_) return false;
  float eta = std::abs(t.momentum().eta());
  if (eta > maxEta_ || eta < minEta_) return false; // no GEMs could be in such eta
  return true;
}


void FastGEMCSCProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  edm::ESHandle<CSCGeometry> csc_g;
  es.get<MuonGeometryRecord>().get(csc_g);
  csc_geo_ = &*csc_g;
  builder_->setCSCGeometry(csc_geo_);

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


void FastGEMCSCProducer::processStubs4SimTrack(map<unsigned int, vector<CSCCorrelatedLCTDigi> >& stubs, SimTrackMatchManager& match)
{
  const SimHitMatcher& match_sh = match.simhits();

  builder_->build(match_sh);

  // match SimHit-modeled stubs to real LCT stubs and update real stub's dphi
  auto model_stubs_ch_ids = builder_->getChamberIds();
  for (auto d: model_stubs_ch_ids)
  {
    // was there any actual LCT in this detid?
    auto dstubs = stubs.find(d);
    if (dstubs == stubs.end()) continue;

    auto &model_stubs = builder_->getStubs(d);
    for (auto &model_stub: model_stubs)
    {
      for (auto& stub: dstubs->second)
      {
        int wg = 1 + stub.getKeyWG(); // LCT halfstrip and wiregoup numbers start from 0
        int hs = 1 + stub.getStrip();
        if ( ! (model_stub.hasHalfStrip(hs) && model_stub.hasWireGroup(wg)) ) continue;
        //float old_dphi = digiIt->getGEMDPhi();
        float dphi = model_stub.dPhiGEMCSCLinear();
        if (usePropagatedDPhi_) dphi = model_stub.dPhiGEMCSCPropagator();
        stub.setGEMDPhi(dphi);
      }
    }
  }
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void FastGEMCSCProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(FastGEMCSCProducer);
