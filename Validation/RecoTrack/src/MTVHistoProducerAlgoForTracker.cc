#include "Validation/RecoTrack/interface/MTVHistoProducerAlgoForTracker.h"
#include "Validation/RecoTrack/interface/trackFromSeedFitFailed.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimTracker/TrackAssociation/interface/TrackingParticleIP.h"


#include "TMath.h"
#include <TF1.h>

using namespace std;

namespace {
  void BinLogX(TH1 *h) {
    TAxis *axis = h->GetXaxis();
    int bins = axis->GetNbins();

    float from = axis->GetXmin();
    float to = axis->GetXmax();
    float width = (to - from) / bins;
    std::vector<float> new_bins(bins+1, 0);

    for (int i = 0; i <= bins; i++) {
      new_bins[i] = TMath::Power(10, from + i * width);
    }
    axis->Set(bins, new_bins.data());
  }

  template<typename T> void fillPlotNoFlow(MonitorElement *me, T val) {
    auto h = me->getTH1();
    const auto xaxis = h->GetXaxis();
    if(val <= xaxis->GetXmin())
      h->AddBinContent(xaxis->GetFirst());
    else if(val >= xaxis->GetXmax())
      h->AddBinContent(xaxis->GetLast());
    else
      h->Fill(val);
  }

  void setBinLabels(MonitorElement *h, const std::vector<std::string>& labels) {
    for(size_t i=0; i<labels.size(); ++i) {
      h->setBinLabel(i+1, labels[i]);
    }
  }

  void setBinLabelsAlgo(MonitorElement *h, int axis=1) {
    for(size_t i=0; i<reco::TrackBase::algoSize; ++i) {
      h->setBinLabel(i+1, reco::TrackBase::algoName(static_cast<reco::TrackBase::TrackAlgorithm>(i)), axis);
    }
  }

  void fillMVAHistos(std::vector<MonitorElement *>& h_mva,
                     std::vector<MonitorElement *>& h_mvacut,
                     std::vector<MonitorElement *>& h_mva_hp,
                     std::vector<MonitorElement *>& h_mvacut_hp,
                     const std::vector<float>& mvas,
                     unsigned int selectsLoose, unsigned int selectsHP) {
    // Fill MVA1 histos with all tracks, MVA2 histos only with tracks
    // not selected by MVA1, etc. 
    for(size_t i=0; i<mvas.size(); ++i) {
      if(i<=selectsLoose) {
        fillPlotNoFlow(h_mva[i], mvas[i]);
        h_mvacut[i]->Fill(mvas[i]);
      }
      if(i>=1 && i<=selectsHP) {
        fillPlotNoFlow(h_mva_hp[i], mvas[i]);
        h_mvacut_hp[i]->Fill(mvas[i]);
      }
    }
  }

  void fillMVAHistos(double xval,
                     std::vector<MonitorElement *>& h_mva,
                     std::vector<MonitorElement *>& h_mva_hp,
                     const std::vector<float>& mvas,
                     unsigned int selectsLoose, unsigned int selectsHP) {
    // Fill MVA1 histos with all tracks, MVA2 histos only with tracks
    // not selected by MVA1, etc.
    for(size_t i=0; i<mvas.size(); ++i) {
      if(i<=selectsLoose) {
        h_mva[i]->Fill(xval, mvas[i]);
      }
      if(i>=1 && i<=selectsHP) {
        h_mva_hp[i]->Fill(xval, mvas[i]);
      }
    }
  }
}

MTVHistoProducerAlgoForTracker::MTVHistoProducerAlgoForTracker(const edm::ParameterSet& pset, const edm::InputTag& beamSpotTag, const bool doSeedPlots, edm::ConsumesCollector & iC):
  doSeedPlots_(doSeedPlots),
  h_ptSIM(nullptr), h_etaSIM(nullptr), h_tracksSIM(nullptr), h_vertposSIM(nullptr), h_bunchxSIM(nullptr)
{
  //parameters for _vs_eta plots
  minEta  = pset.getParameter<double>("minEta");
  maxEta  = pset.getParameter<double>("maxEta");
  nintEta = pset.getParameter<int>("nintEta");
  useFabsEta = pset.getParameter<bool>("useFabsEta");

  //parameters for _vs_pt plots
  minPt  = pset.getParameter<double>("minPt");
  maxPt  = pset.getParameter<double>("maxPt");
  nintPt = pset.getParameter<int>("nintPt");
  useInvPt = pset.getParameter<bool>("useInvPt");
  useLogPt = pset.getUntrackedParameter<bool>("useLogPt",false);

  //parameters for _vs_Hit plots
  minHit  = pset.getParameter<double>("minHit");
  maxHit  = pset.getParameter<double>("maxHit");
  nintHit = pset.getParameter<int>("nintHit");

  //parameters for _vs_Pu plots
  minPu  = pset.getParameter<double>("minPu");
  maxPu  = pset.getParameter<double>("maxPu");
  nintPu = pset.getParameter<int>("nintPu");

  //parameters for _vs_Layer plots
  minLayers  = pset.getParameter<double>("minLayers");
  maxLayers  = pset.getParameter<double>("maxLayers");
  nintLayers = pset.getParameter<int>("nintLayers");

  //parameters for _vs_phi plots
  minPhi  = pset.getParameter<double>("minPhi");
  maxPhi  = pset.getParameter<double>("maxPhi");
  nintPhi = pset.getParameter<int>("nintPhi");

  //parameters for _vs_Dxy plots
  minDxy  = pset.getParameter<double>("minDxy");
  maxDxy  = pset.getParameter<double>("maxDxy");
  nintDxy = pset.getParameter<int>("nintDxy");

  //parameters for _vs_Dz plots
  minDz  = pset.getParameter<double>("minDz");
  maxDz  = pset.getParameter<double>("maxDz");
  nintDz = pset.getParameter<int>("nintDz");

  dxyDzZoom = pset.getParameter<double>("dxyDzZoom");

  //parameters for _vs_ProductionVertexTransvPosition plots
  minVertpos  = pset.getParameter<double>("minVertpos");
  maxVertpos  = pset.getParameter<double>("maxVertpos");
  nintVertpos = pset.getParameter<int>("nintVertpos");
  useLogVertpos = pset.getUntrackedParameter<bool>("useLogVertpos");

  //parameters for _vs_ProductionVertexZPosition plots
  minZpos  = pset.getParameter<double>("minZpos");
  maxZpos  = pset.getParameter<double>("maxZpos");
  nintZpos = pset.getParameter<int>("nintZpos");

  //parameters for _vs_dR plots
  mindr  = pset.getParameter<double>("mindr");
  maxdr  = pset.getParameter<double>("maxdr");
  nintdr = pset.getParameter<int>("nintdr");

  // paramers for _vs_chi2 plots
  minChi2  = pset.getParameter<double>("minChi2");
  maxChi2  = pset.getParameter<double>("maxChi2");
  nintChi2 = pset.getParameter<int>("nintChi2");

  //parameters for dE/dx plots
  minDeDx  = pset.getParameter<double>("minDeDx");
  maxDeDx  = pset.getParameter<double>("maxDeDx");
  nintDeDx = pset.getParameter<int>("nintDeDx");

  //parameters for Pileup plots
  minVertcount  = pset.getParameter<double>("minVertcount");
  maxVertcount  = pset.getParameter<double>("maxVertcount");
  nintVertcount = pset.getParameter<int>("nintVertcount");

  //parameters for number of tracks plots
  minTracks  = pset.getParameter<double>("minTracks");
  maxTracks  = pset.getParameter<double>("maxTracks");
  nintTracks = pset.getParameter<int>("nintTracks");

  //parameters for vs. PV z plots
  minPVz  = pset.getParameter<double>("minPVz");
  maxPVz  = pset.getParameter<double>("maxPVz");
  nintPVz = pset.getParameter<int>("nintPVz");

  //parameters for vs. MVA plots
  minMVA = pset.getParameter<double>("minMVA");
  maxMVA  = pset.getParameter<double>("maxMVA");
  nintMVA = pset.getParameter<int>("nintMVA");

  //parameters for resolution plots
  ptRes_rangeMin = pset.getParameter<double>("ptRes_rangeMin");
  ptRes_rangeMax = pset.getParameter<double>("ptRes_rangeMax");
  ptRes_nbin = pset.getParameter<int>("ptRes_nbin");

  phiRes_rangeMin = pset.getParameter<double>("phiRes_rangeMin");
  phiRes_rangeMax = pset.getParameter<double>("phiRes_rangeMax");
  phiRes_nbin = pset.getParameter<int>("phiRes_nbin");

  cotThetaRes_rangeMin = pset.getParameter<double>("cotThetaRes_rangeMin");
  cotThetaRes_rangeMax = pset.getParameter<double>("cotThetaRes_rangeMax");
  cotThetaRes_nbin = pset.getParameter<int>("cotThetaRes_nbin");

  dxyRes_rangeMin = pset.getParameter<double>("dxyRes_rangeMin");
  dxyRes_rangeMax = pset.getParameter<double>("dxyRes_rangeMax");
  dxyRes_nbin = pset.getParameter<int>("dxyRes_nbin");

  dzRes_rangeMin = pset.getParameter<double>("dzRes_rangeMin");
  dzRes_rangeMax = pset.getParameter<double>("dzRes_rangeMax");
  dzRes_nbin = pset.getParameter<int>("dzRes_nbin");


  maxDzpvCum = pset.getParameter<double>("maxDzpvCumulative");
  nintDzpvCum = pset.getParameter<int>("nintDzpvCumulative");

  maxDzpvsigCum = pset.getParameter<double>("maxDzpvsigCumulative");
  nintDzpvsigCum = pset.getParameter<int>("nintDzpvsigCumulative");

  //--- tracking particle selectors for efficiency measurements
  using namespace edm;
  using namespace reco::modules;
  auto initTPselector = [&](auto& sel, auto& name) {
    sel = std::make_unique<TrackingParticleSelector>(ParameterAdapter<TrackingParticleSelector>::make(pset.getParameter<ParameterSet>(name), iC));
  };
  auto initTrackSelector = [&](auto& sel, auto& name) {
    sel = makeRecoTrackSelectorFromTPSelectorParameters(pset.getParameter<ParameterSet>(name), beamSpotTag, iC);
  };
  auto initGPselector = [&](auto& sel, auto& name) {
    sel = std::make_unique<GenParticleCustomSelector>(ParameterAdapter<GenParticleCustomSelector>::make(pset.getParameter<ParameterSet>(name), iC));
  };

  initTPselector(generalTpSelector,             "generalTpSelector");
  initTPselector(TpSelectorForEfficiencyVsEta,  "TpSelectorForEfficiencyVsEta");
  initTPselector(TpSelectorForEfficiencyVsPhi,  "TpSelectorForEfficiencyVsPhi");
  initTPselector(TpSelectorForEfficiencyVsPt,   "TpSelectorForEfficiencyVsPt");
  initTPselector(TpSelectorForEfficiencyVsVTXR, "TpSelectorForEfficiencyVsVTXR");
  initTPselector(TpSelectorForEfficiencyVsVTXZ, "TpSelectorForEfficiencyVsVTXZ");

  initTrackSelector(trackSelectorVsEta, "TpSelectorForEfficiencyVsEta");
  initTrackSelector(trackSelectorVsPhi, "TpSelectorForEfficiencyVsPhi");
  initTrackSelector(trackSelectorVsPt,  "TpSelectorForEfficiencyVsPt");

  initGPselector(generalGpSelector,             "generalGpSelector");
  initGPselector(GpSelectorForEfficiencyVsEta,  "GpSelectorForEfficiencyVsEta");
  initGPselector(GpSelectorForEfficiencyVsPhi,  "GpSelectorForEfficiencyVsPhi");
  initGPselector(GpSelectorForEfficiencyVsPt,   "GpSelectorForEfficiencyVsPt");
  initGPselector(GpSelectorForEfficiencyVsVTXR, "GpSelectorForEfficiencyVsVTXR");
  initGPselector(GpSelectorForEfficiencyVsVTXZ, "GpSelectorForEfficiencyVsVTXZ");

  // SeedingLayerSets
  // If enabled, use last bin to denote other or unknown cases
  seedingLayerSetNames = pset.getParameter<std::vector<std::string> >("seedingLayerSets");
  std::vector<std::pair<SeedingLayerSetId, std::string> > stripPairSets;
  if(!seedingLayerSetNames.empty()) {
    std::vector<std::vector<std::string>> layerSets = SeedingLayerSetsBuilder::layerNamesInSets(seedingLayerSetNames);
    for(size_t i=0; i<layerSets.size(); ++i) {
      const auto& layerSet = layerSets[i];
      if(layerSet.size() > std::tuple_size<SeedingLayerSetId>::value) {
        throw cms::Exception("Configuration") << "Got seedingLayerSet " << seedingLayerSetNames[i] << " with " << layerSet.size() << " elements, but I have a hard-coded maximum of " << std::tuple_size<SeedingLayerSetId>::value << ". Please increase the maximum in MTVHistoProducerAlgoForTracker.h";
      }
      SeedingLayerSetId setId;
      for(size_t j=0; j<layerSet.size(); ++j) {
        // It is a bit ugly to assume here that 'M' prefix stands for
        // strip mono hits, as in the SeedingLayerSetsBuilder code any
        // prefixes are arbitrary and their meaning is defined fully
        // in the configuration. But, this is the easiest way.
        bool isStripMono = !layerSet[j].empty() && layerSet[j][0] == 'M';
        setId[j] = std::make_tuple(SeedingLayerSetsBuilder::nameToEnumId(layerSet[j]), isStripMono);
      }
      // Account for the fact that strip triplet seeding may give pairs
      if(layerSet.size() == 3 && isTrackerStrip(std::get<GeomDetEnumerators::SubDetector>(std::get<0>(setId[0])))) {
        SeedingLayerSetId pairId;
        pairId[0] = setId[0];
        pairId[1] = setId[1];
        stripPairSets.emplace_back(pairId, layerSet[0]+"+"+layerSet[1]);
      }

      auto inserted = seedingLayerSetToBin.insert(std::make_pair(setId, i));
      if(!inserted.second)
        throw cms::Exception("Configuration") << "SeedingLayerSet " << seedingLayerSetNames[i] << " is specified twice, while the set list should be unique.";
    }

    // Add the "strip pairs from strip triplets" if they don't otherwise exist
    for(const auto& setIdName: stripPairSets) {
      auto inserted = seedingLayerSetToBin.insert(std::make_pair(setIdName.first, seedingLayerSetNames.size()));
      if(inserted.second)
        seedingLayerSetNames.push_back(setIdName.second);
    }

    seedingLayerSetNames.emplace_back("Other/Unknown");
  }

  // fix for the LogScale by Ryan
  if(useLogPt){
    maxPt=log10(maxPt);
    if(minPt > 0){
      minPt=log10(minPt);
    }
    else{
      edm::LogWarning("MultiTrackValidator")
	<< "minPt = "
	<< minPt << " <= 0 out of range while requesting log scale.  Using minPt = 0.1.";
      minPt=log10(0.1);
    }
  }
  if(useLogVertpos) {
    maxVertpos = std::log10(maxVertpos);
    if(minVertpos > 0) {
      minVertpos = std::log10(minVertpos);
    }
    else {
      edm::LogWarning("MultiTrackValidator")
	<< "minVertpos = " << minVertpos << " <= 0 out of range while requesting log scale.  Using minVertpos = 0.1.";
      minVertpos = -1;
    }
  }

}

MTVHistoProducerAlgoForTracker::~MTVHistoProducerAlgoForTracker() {}

std::unique_ptr<RecoTrackSelectorBase> MTVHistoProducerAlgoForTracker::makeRecoTrackSelectorFromTPSelectorParameters(const edm::ParameterSet& pset, const edm::InputTag& beamSpotTag, edm::ConsumesCollector& iC) {
  edm::ParameterSet psetTrack;
  psetTrack.copyForModify(pset);
  psetTrack.eraseSimpleParameter("minHit");
  psetTrack.eraseSimpleParameter("signalOnly");
  psetTrack.eraseSimpleParameter("intimeOnly");
  psetTrack.eraseSimpleParameter("chargedOnly");
  psetTrack.eraseSimpleParameter("stableOnly");
  psetTrack.addParameter("maxChi2", 1e10);
  psetTrack.addParameter("minHit", 0);
  psetTrack.addParameter("minPixelHit", 0);
  psetTrack.addParameter("minLayer", 0);
  psetTrack.addParameter("min3DLayer", 0);
  psetTrack.addParameter("usePV", false);
  psetTrack.addParameter("beamSpot", beamSpotTag);
  psetTrack.addParameter("quality", std::vector<std::string>{});
  psetTrack.addParameter("algorithm", std::vector<std::string>{});
  psetTrack.addParameter("originalAlgorithm", std::vector<std::string>{});
  psetTrack.addParameter("algorithmMaskContains", std::vector<std::string>{});

  return std::make_unique<RecoTrackSelectorBase>(psetTrack, iC);
}

void MTVHistoProducerAlgoForTracker::init(const edm::Event& event, const edm::EventSetup& setup) {
  trackSelectorVsEta->init(event, setup);
  trackSelectorVsPhi->init(event, setup);
  trackSelectorVsPt->init(event, setup);
}

void MTVHistoProducerAlgoForTracker::bookSimHistos(DQMStore::IBooker& ibook){
  if(h_ptSIM != nullptr)
    throw cms::Exception("LogicError") << "bookSimHistos() has already been called";

  h_ptSIM = ibook.book1D("ptSIM", "generated p_{t}", nintPt, minPt, maxPt);
  h_etaSIM = ibook.book1D("etaSIM", "generated pseudorapidity", nintEta, minEta, maxEta);
  h_tracksSIM = ibook.book1D("tracksSIM","number of simulated tracks", nintTracks, minTracks, maxTracks*10);
  h_vertposSIM = ibook.book1D("vertposSIM","Transverse position of sim vertices", nintVertpos, minVertpos, maxVertpos);
  h_bunchxSIM = ibook.book1D("bunchxSIM", "bunch crossing", 21, -15.5, 5.5 );

  if(useLogPt) {
    BinLogX(h_ptSIM->getTH1F());
  }
}

void MTVHistoProducerAlgoForTracker::bookSimTrackHistos(DQMStore::IBooker& ibook, bool doResolutionPlots) {
  h_assoceta.push_back( ibook.book1D("num_assoc(simToReco)_eta","N of associated tracks (simToReco) vs eta",nintEta,minEta,maxEta) );
  h_simuleta.push_back( ibook.book1D("num_simul_eta","N of simulated tracks vs eta",nintEta,minEta,maxEta) );

  h_assocpT.push_back( ibook.book1D("num_assoc(simToReco)_pT","N of associated tracks (simToReco) vs pT",nintPt,minPt,maxPt) );
  h_simulpT.push_back( ibook.book1D("num_simul_pT","N of simulated tracks vs pT",nintPt,minPt,maxPt) );

  h_assochit.push_back( ibook.book1D("num_assoc(simToReco)_hit","N of associated tracks (simToReco) vs hit",nintHit,minHit,maxHit) );
  h_simulhit.push_back( ibook.book1D("num_simul_hit","N of simulated tracks vs hit",nintHit,minHit,maxHit) );

  h_assoclayer.push_back( ibook.book1D("num_assoc(simToReco)_layer","N of associated tracks (simToReco) vs layer",nintLayers,minLayers,maxLayers) );
  h_simullayer.push_back( ibook.book1D("num_simul_layer","N of simulated tracks vs layer",nintLayers,minLayers,maxLayers) );

  h_assocpixellayer.push_back( ibook.book1D("num_assoc(simToReco)_pixellayer","N of associated tracks (simToReco) vs pixel layer",nintLayers,minLayers,maxLayers) );
  h_simulpixellayer.push_back( ibook.book1D("num_simul_pixellayer","N of simulated tracks vs pixel layer",nintLayers,minLayers,maxLayers) );

  h_assoc3Dlayer.push_back( ibook.book1D("num_assoc(simToReco)_3Dlayer","N of associated tracks (simToReco) vs 3D layer",nintLayers,minLayers,maxLayers) );
  h_simul3Dlayer.push_back( ibook.book1D("num_simul_3Dlayer","N of simulated tracks vs 3D layer",nintLayers,minLayers,maxLayers) );

  h_assocpu.push_back( ibook.book1D("num_assoc(simToReco)_pu","N of associated tracks (simToReco) vs pu",nintPu,minPu,maxPu) );
  h_simulpu.push_back( ibook.book1D("num_simul_pu","N of simulated tracks vs pu",nintPu,minPu,maxPu) );

  h_assocphi.push_back( ibook.book1D("num_assoc(simToReco)_phi","N of associated tracks (simToReco) vs phi",nintPhi,minPhi,maxPhi) );
  h_simulphi.push_back( ibook.book1D("num_simul_phi","N of simulated tracks vs phi",nintPhi,minPhi,maxPhi) );

  h_assocdxy.push_back( ibook.book1D("num_assoc(simToReco)_dxy","N of associated tracks (simToReco) vs dxy",nintDxy,minDxy,maxDxy) );
  h_simuldxy.push_back( ibook.book1D("num_simul_dxy","N of simulated tracks vs dxy",nintDxy,minDxy,maxDxy) );

  h_assocdz.push_back( ibook.book1D("num_assoc(simToReco)_dz","N of associated tracks (simToReco) vs dz",nintDz,minDz,maxDz) );
  h_simuldz.push_back( ibook.book1D("num_simul_dz","N of simulated tracks vs dz",nintDz,minDz,maxDz) );

  h_assocvertpos.push_back( ibook.book1D("num_assoc(simToReco)_vertpos",
					 "N of associated tracks (simToReco) vs transverse vert position",
					 nintVertpos,minVertpos,maxVertpos) );
  h_simulvertpos.push_back( ibook.book1D("num_simul_vertpos","N of simulated tracks vs transverse vert position",
					 nintVertpos,minVertpos,maxVertpos) );

  h_assoczpos.push_back( ibook.book1D("num_assoc(simToReco)_zpos","N of associated tracks (simToReco) vs z vert position",
				      nintZpos,minZpos,maxZpos) );
  h_simulzpos.push_back( ibook.book1D("num_simul_zpos","N of simulated tracks vs z vert position",nintZpos,minZpos,maxZpos) );

  h_assocdr.push_back( ibook.book1D("num_assoc(simToReco)_dr","N of associated tracks (simToReco) vs dR",nintdr,log10(mindr),log10(maxdr)) );
  h_simuldr.push_back( ibook.book1D("num_simul_dr","N of simulated tracks vs dR",nintdr,log10(mindr),log10(maxdr)) );
  BinLogX(h_assocdr.back()->getTH1F());
  BinLogX(h_simuldr.back()->getTH1F());

  h_simul_simpvz.push_back( ibook.book1D("num_simul_simpvz", "N of simulated tracks vs. sim PV z", nintPVz, minPVz, maxPVz) );
  h_assoc_simpvz.push_back( ibook.book1D("num_assoc(simToReco)_simpvz", "N of associated tracks (simToReco) vs. sim PV z", nintPVz, minPVz, maxPVz) );

  nrecHit_vs_nsimHit_sim2rec.push_back( doResolutionPlots ? ibook.book2D("nrecHit_vs_nsimHit_sim2rec","nrecHit vs nsimHit (Sim2RecAssoc)",
                                                                         nintHit,minHit,maxHit, nintHit,minHit,maxHit )
                                        : nullptr);

  // TODO: use the dynamic track algo priority order also here
  constexpr auto nalgos = reco::TrackBase::algoSize;
  h_duplicates_oriAlgo_vs_oriAlgo.push_back( ibook.book2D("duplicates_oriAlgo_vs_oriAlgo", "Duplicate tracks: originalAlgo vs originalAlgo",
                                                          nalgos,0,nalgos, nalgos,0,nalgos) );
  setBinLabelsAlgo(h_duplicates_oriAlgo_vs_oriAlgo.back(), 1);
  setBinLabelsAlgo(h_duplicates_oriAlgo_vs_oriAlgo.back(), 2);

  if(useLogPt){
    BinLogX(h_assocpT.back()->getTH1F());
    BinLogX(h_simulpT.back()->getTH1F());
  }
  if(useLogVertpos) {
    BinLogX(h_assocvertpos.back()->getTH1F());
    BinLogX(h_simulvertpos.back()->getTH1F());
  }
}

void MTVHistoProducerAlgoForTracker::bookSimTrackPVAssociationHistos(DQMStore::IBooker& ibook){
  h_assocdxypv.push_back( ibook.book1D("num_assoc(simToReco)_dxypv","N of associated tracks (simToReco) vs dxy(PV)",nintDxy,minDxy,maxDxy) );
  h_simuldxypv.push_back( ibook.book1D("num_simul_dxypv","N of simulated tracks vs dxy(PV)",nintDxy,minDxy,maxDxy) );

  h_assocdzpv.push_back( ibook.book1D("num_assoc(simToReco)_dzpv","N of associated tracks (simToReco) vs dz(PV)",nintDz,minDz,maxDz) );
  h_simuldzpv.push_back( ibook.book1D("num_simul_dzpv","N of simulated tracks vs dz(PV)",nintDz,minDz,maxDz) );

  h_assocdxypvzoomed.push_back( ibook.book1D("num_assoc(simToReco)_dxypv_zoomed","N of associated tracks (simToReco) vs dxy(PV)",nintDxy,minDxy/dxyDzZoom,maxDxy/dxyDzZoom) );
  h_simuldxypvzoomed.push_back( ibook.book1D("num_simul_dxypv_zoomed","N of simulated tracks vs dxy(PV)",nintDxy,minDxy/dxyDzZoom,maxDxy/dxyDzZoom) );

  h_assocdzpvzoomed.push_back( ibook.book1D("num_assoc(simToReco)_dzpv_zoomed","N of associated tracks (simToReco) vs dz(PV)",nintDz,minDz/dxyDzZoom,maxDz/dxyDzZoom) );
  h_simuldzpvzoomed.push_back( ibook.book1D("num_simul_dzpv_zoomed","N of simulated tracks vs dz(PV)",nintDz,minDz/dxyDzZoom,maxDz/dxyDzZoom) );

  h_assoc_dzpvcut.push_back( ibook.book1D("num_assoc(simToReco)_dzpvcut","N of associated tracks (simToReco) vs dz(PV)",nintDzpvCum,0,maxDzpvCum) );
  h_simul_dzpvcut.push_back( ibook.book1D("num_simul_dzpvcut","N of simulated tracks from sim PV",nintDzpvCum,0,maxDzpvCum) );
  h_simul2_dzpvcut.push_back( ibook.book1D("num_simul2_dzpvcut","N of simulated tracks (associated to any track) from sim PV",nintDzpvCum,0,maxDzpvCum) );

  h_assoc_dzpvcut_pt.push_back( ibook.book1D("num_assoc(simToReco)_dzpvcut_pt","#sump_{T} of associated tracks (simToReco) vs dz(PV)",nintDzpvCum,0,maxDzpvCum) );
  h_simul_dzpvcut_pt.push_back( ibook.book1D("num_simul_dzpvcut_pt","#sump_{T} of simulated tracks from sim PV",nintDzpvCum,0,maxDzpvCum) );
  h_simul2_dzpvcut_pt.push_back( ibook.book1D("num_simul2_dzpvcut_pt","#sump_{T} of simulated tracks (associated to any track) from sim PV",nintDzpvCum,0,maxDzpvCum) );
  h_assoc_dzpvcut_pt.back()->getTH1()->Sumw2();
  h_simul_dzpvcut_pt.back()->getTH1()->Sumw2();
  h_simul2_dzpvcut_pt.back()->getTH1()->Sumw2();

  h_assoc_dzpvsigcut.push_back( ibook.book1D("num_assoc(simToReco)_dzpvsigcut","N of associated tracks (simToReco) vs dz(PV)/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );
  h_simul_dzpvsigcut.push_back( ibook.book1D("num_simul_dzpvsigcut","N of simulated tracks from sim PV/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );
  h_simul2_dzpvsigcut.push_back( ibook.book1D("num_simul2_dzpvsigcut","N of simulated tracks (associated to any track) from sim PV/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );

  h_assoc_dzpvsigcut_pt.push_back( ibook.book1D("num_assoc(simToReco)_dzpvsigcut_pt","#sump_{T} of associated tracks (simToReco) vs dz(PV)/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );
  h_simul_dzpvsigcut_pt.push_back( ibook.book1D("num_simul_dzpvsigcut_pt","#sump_{T} of simulated tracks from sim PV/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );
  h_simul2_dzpvsigcut_pt.push_back( ibook.book1D("num_simul2_dzpvsigcut_pt","#sump_{T} of simulated tracks (associated to any track) from sim PV/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );
  h_assoc_dzpvsigcut_pt.back()->getTH1()->Sumw2();
  h_simul_dzpvsigcut_pt.back()->getTH1()->Sumw2();
  h_simul2_dzpvsigcut_pt.back()->getTH1()->Sumw2();
}

void MTVHistoProducerAlgoForTracker::bookRecoHistos(DQMStore::IBooker& ibook, bool doResolutionPlots) {
  h_tracks.push_back( ibook.book1D("tracks","number of reconstructed tracks", nintTracks, minTracks, maxTracks) );
  h_fakes.push_back( ibook.book1D("fakes","number of fake reco tracks", nintTracks, minTracks, maxTracks) );
  h_charge.push_back( ibook.book1D("charge","charge",3,-1.5,1.5) );

  h_hits.push_back( ibook.book1D("hits", "number of hits per track", nintHit,minHit,maxHit ) );
  h_losthits.push_back( ibook.book1D("losthits", "number of lost hits per track", nintHit,minHit,maxHit) );
  h_nchi2.push_back( ibook.book1D("chi2", "normalized #chi^{2}", 200, 0, 20 ) );
  h_nchi2_prob.push_back( ibook.book1D("chi2_prob", "normalized #chi^{2} probability",100,0,1));

  h_nmisslayers_inner.push_back( ibook.book1D("missing_inner_layers", "number of missing inner layers", nintLayers,minLayers,maxLayers ) );
  h_nmisslayers_outer.push_back( ibook.book1D("missing_outer_layers", "number of missing outer layers", nintLayers,minLayers,maxLayers ) );

  h_algo.push_back( ibook.book1D("h_algo","Tracks by algo",reco::TrackBase::algoSize, 0., double(reco::TrackBase::algoSize) ) );
  for (size_t ibin=0; ibin<reco::TrackBase::algoSize-1; ibin++)
    h_algo.back()->setBinLabel(ibin+1,reco::TrackBase::algoNames[ibin]);
  //    h_algo.setBinLabel(ibin+1,reco::TrackBase::algoNames[ibin]);

  /// these are needed to calculate efficiency during the harvesting for the automated validation
  h_recoeta.push_back( ibook.book1D("num_reco_eta","N of reco track vs eta",nintEta,minEta,maxEta) );
  h_reco2eta.push_back( ibook.book1D("num_reco2_eta","N of selected reco track vs eta",nintEta,minEta,maxEta) );
  h_assoc2eta.push_back( ibook.book1D("num_assoc(recoToSim)_eta","N of associated (recoToSim) tracks vs eta",nintEta,minEta,maxEta) );
  h_loopereta.push_back( ibook.book1D("num_duplicate_eta","N of associated (recoToSim) duplicate tracks vs eta",nintEta,minEta,maxEta) );
  if(!doSeedPlots_) h_misideta.push_back( ibook.book1D("num_chargemisid_eta","N of associated (recoToSim) charge misIDed tracks vs eta",nintEta,minEta,maxEta) );
  h_pileupeta.push_back( ibook.book1D("num_pileup_eta","N of associated (recoToSim) pileup tracks vs eta",nintEta,minEta,maxEta) );
  //
  h_recopT.push_back( ibook.book1D("num_reco_pT","N of reco track vs pT",nintPt,minPt,maxPt) );
  h_reco2pT.push_back( ibook.book1D("num_reco2_pT","N of selected reco track vs pT",nintPt,minPt,maxPt) );
  h_assoc2pT.push_back( ibook.book1D("num_assoc(recoToSim)_pT","N of associated (recoToSim) tracks vs pT",nintPt,minPt,maxPt) );
  h_looperpT.push_back( ibook.book1D("num_duplicate_pT","N of associated (recoToSim) duplicate tracks vs pT",nintPt,minPt,maxPt) );
  if(!doSeedPlots_) h_misidpT.push_back( ibook.book1D("num_chargemisid_pT","N of associated (recoToSim) charge misIDed tracks vs pT",nintPt,minPt,maxPt) );
  h_pileuppT.push_back( ibook.book1D("num_pileup_pT","N of associated (recoToSim) pileup tracks vs pT",nintPt,minPt,maxPt) );
  //
  h_recohit.push_back( ibook.book1D("num_reco_hit","N of reco track vs hit",nintHit,minHit,maxHit) );
  h_assoc2hit.push_back( ibook.book1D("num_assoc(recoToSim)_hit","N of associated (recoToSim) tracks vs hit",nintHit,minHit,maxHit) );
  h_looperhit.push_back( ibook.book1D("num_duplicate_hit","N of associated (recoToSim) duplicate tracks vs hit",nintHit,minHit,maxHit) );
  if(!doSeedPlots_) h_misidhit.push_back( ibook.book1D("num_chargemisid_hit","N of associated (recoToSim) charge misIDed tracks vs hit",nintHit,minHit,maxHit) );
  h_pileuphit.push_back( ibook.book1D("num_pileup_hit","N of associated (recoToSim) pileup tracks vs hit",nintHit,minHit,maxHit) );
  //
  h_recolayer.push_back( ibook.book1D("num_reco_layer","N of reco track vs layer",nintLayers,minLayers,maxLayers) );
  h_assoc2layer.push_back( ibook.book1D("num_assoc(recoToSim)_layer","N of associated (recoToSim) tracks vs layer",nintLayers,minLayers,maxLayers) );
  h_looperlayer.push_back( ibook.book1D("num_duplicate_layer","N of associated (recoToSim) duplicate tracks vs layer",nintLayers,minLayers,maxLayers) );
  if(!doSeedPlots_) h_misidlayer.push_back( ibook.book1D("num_chargemisid_layer","N of associated (recoToSim) charge misIDed tracks vs layer",nintLayers,minLayers,maxLayers) );
  h_pileuplayer.push_back( ibook.book1D("num_pileup_layer","N of associated (recoToSim) pileup tracks vs layer",nintLayers,minLayers,maxLayers) );
  //
  h_recopixellayer.push_back( ibook.book1D("num_reco_pixellayer","N of reco track vs pixellayer",nintLayers,minLayers,maxLayers) );
  h_assoc2pixellayer.push_back( ibook.book1D("num_assoc(recoToSim)_pixellayer","N of associated (recoToSim) tracks vs pixellayer",nintLayers,minLayers,maxLayers) );
  h_looperpixellayer.push_back( ibook.book1D("num_duplicate_pixellayer","N of associated (recoToSim) duplicate tracks vs pixellayer",nintLayers,minLayers,maxLayers) );
  if(!doSeedPlots_) h_misidpixellayer.push_back( ibook.book1D("num_chargemisid_pixellayer","N of associated (recoToSim) charge misIDed tracks vs pixellayer",nintLayers,minLayers,maxLayers) );
  h_pileuppixellayer.push_back( ibook.book1D("num_pileup_pixellayer","N of associated (recoToSim) pileup tracks vs pixellayer",nintLayers,minLayers,maxLayers) );
  //
  h_reco3Dlayer.push_back( ibook.book1D("num_reco_3Dlayer","N of reco track vs 3D layer",nintLayers,minLayers,maxLayers) );
  h_assoc23Dlayer.push_back( ibook.book1D("num_assoc(recoToSim)_3Dlayer","N of associated (recoToSim) tracks vs 3D layer",nintLayers,minLayers,maxLayers) );
  h_looper3Dlayer.push_back( ibook.book1D("num_duplicate_3Dlayer","N of associated (recoToSim) duplicate tracks vs 3D layer",nintLayers,minLayers,maxLayers) );
  if(!doSeedPlots_) h_misid3Dlayer.push_back( ibook.book1D("num_chargemisid_3Dlayer","N of associated (recoToSim) charge misIDed tracks vs 3D layer",nintLayers,minLayers,maxLayers) );
  h_pileup3Dlayer.push_back( ibook.book1D("num_pileup_3Dlayer","N of associated (recoToSim) pileup tracks vs 3D layer",nintLayers,minLayers,maxLayers) );
  //
  h_recopu.push_back( ibook.book1D("num_reco_pu","N of reco track vs pu",nintPu,minPu,maxPu) );
  h_reco2pu.push_back( ibook.book1D("num_reco2_pu","N of selected reco track vs pu",nintPu,minPu,maxPu) );
  h_assoc2pu.push_back( ibook.book1D("num_assoc(recoToSim)_pu","N of associated (recoToSim) tracks vs pu",nintPu,minPu,maxPu) );
  h_looperpu.push_back( ibook.book1D("num_duplicate_pu","N of associated (recoToSim) duplicate tracks vs pu",nintPu,minPu,maxPu) );
  if(!doSeedPlots_) h_misidpu.push_back( ibook.book1D("num_chargemisid_pu","N of associated (recoToSim) charge misIDed tracks vs pu",nintPu,minPu,maxPu) );
  h_pileuppu.push_back( ibook.book1D("num_pileup_pu","N of associated (recoToSim) pileup tracks vs pu",nintPu,minPu,maxPu) );
  //
  h_recophi.push_back( ibook.book1D("num_reco_phi","N of reco track vs phi",nintPhi,minPhi,maxPhi) );
  h_assoc2phi.push_back( ibook.book1D("num_assoc(recoToSim)_phi","N of associated (recoToSim) tracks vs phi",nintPhi,minPhi,maxPhi) );
  h_looperphi.push_back( ibook.book1D("num_duplicate_phi","N of associated (recoToSim) duplicate tracks vs phi",nintPhi,minPhi,maxPhi) );
  if(!doSeedPlots_) h_misidphi.push_back( ibook.book1D("num_chargemisid_phi","N of associated (recoToSim) charge misIDed tracks vs phi",nintPhi,minPhi,maxPhi) );
  h_pileupphi.push_back( ibook.book1D("num_pileup_phi","N of associated (recoToSim) pileup tracks vs phi",nintPhi,minPhi,maxPhi) );

  h_recodxy.push_back( ibook.book1D("num_reco_dxy","N of reco track vs dxy",nintDxy,minDxy,maxDxy) );
  h_assoc2dxy.push_back( ibook.book1D("num_assoc(recoToSim)_dxy","N of associated (recoToSim) tracks vs dxy",nintDxy,minDxy,maxDxy) );
  h_looperdxy.push_back( ibook.book1D("num_duplicate_dxy","N of associated (recoToSim) looper tracks vs dxy",nintDxy,minDxy,maxDxy) );
  if(!doSeedPlots_) h_misiddxy.push_back( ibook.book1D("num_chargemisid_dxy","N of associated (recoToSim) charge misIDed tracks vs dxy",nintDxy,minDxy,maxDxy) );
  h_pileupdxy.push_back( ibook.book1D("num_pileup_dxy","N of associated (recoToSim) pileup tracks vs dxy",nintDxy,minDxy,maxDxy) );

  h_recodz.push_back( ibook.book1D("num_reco_dz","N of reco track vs dz",nintDz,minDz,maxDz) );
  h_assoc2dz.push_back( ibook.book1D("num_assoc(recoToSim)_dz","N of associated (recoToSim) tracks vs dz",nintDz,minDz,maxDz) );
  h_looperdz.push_back( ibook.book1D("num_duplicate_dz","N of associated (recoToSim) looper tracks vs dz",nintDz,minDz,maxDz) );
  if(!doSeedPlots_) h_misiddz.push_back( ibook.book1D("num_chargemisid_versus_dz","N of associated (recoToSim) charge misIDed tracks vs dz",nintDz,minDz,maxDz) );
  h_pileupdz.push_back( ibook.book1D("num_pileup_dz","N of associated (recoToSim) pileup tracks vs dz",nintDz,minDz,maxDz) );

  h_recovertpos.push_back( ibook.book1D("num_reco_vertpos","N of reconstructed tracks vs transverse ref point position",nintVertpos,minVertpos,maxVertpos) );
  h_assoc2vertpos.push_back( ibook.book1D("num_assoc(recoToSim)_vertpos","N of associated (recoToSim) tracks vs transverse ref point position",nintVertpos,minVertpos,maxVertpos) );
  h_loopervertpos.push_back( ibook.book1D("num_duplicate_vertpos","N of associated (recoToSim) looper tracks vs transverse ref point position",nintVertpos,minVertpos,maxVertpos) );
  h_pileupvertpos.push_back( ibook.book1D("num_pileup_vertpos","N of associated (recoToSim) pileup tracks vs transverse ref point position",nintVertpos,minVertpos,maxVertpos) );

  h_recozpos.push_back( ibook.book1D("num_reco_zpos","N of reconstructed tracks vs transverse ref point position",nintZpos,minZpos,maxZpos) );
  h_assoc2zpos.push_back( ibook.book1D("num_assoc(recoToSim)_zpos","N of associated (recoToSim) tracks vs transverse ref point position",nintZpos,minZpos,maxZpos) );
  h_looperzpos.push_back( ibook.book1D("num_duplicate_zpos","N of associated (recoToSim) looper tracks vs transverse ref point position",nintZpos,minZpos,maxZpos) );
  h_pileupzpos.push_back( ibook.book1D("num_pileup_zpos","N of associated (recoToSim) pileup tracks vs transverse ref point position",nintZpos,minZpos,maxZpos) );

  h_recodr.push_back( ibook.book1D("num_reco_dr","N of reconstructed tracks vs dR",nintdr,log10(mindr),log10(maxdr)) );
  h_assoc2dr.push_back( ibook.book1D("num_assoc(recoToSim)_dr","N of associated tracks (recoToSim) vs dR",nintdr,log10(mindr),log10(maxdr)) );
  h_looperdr.push_back( ibook.book1D("num_duplicate_dr","N of associated (recoToSim) looper tracks vs dR",nintdr,log10(mindr),log10(maxdr)) );
  h_pileupdr.push_back( ibook.book1D("num_pileup_dr","N of associated (recoToSim) pileup tracks vs dR",nintdr,log10(mindr),log10(maxdr)) );
  BinLogX(h_recodr.back()->getTH1F());
  BinLogX(h_assoc2dr.back()->getTH1F());
  BinLogX(h_looperdr.back()->getTH1F());
  BinLogX(h_pileupdr.back()->getTH1F());

  h_reco_simpvz.push_back( ibook.book1D("num_reco_simpvz", "N of reco track vs. sim PV z", nintPVz, minPVz, maxPVz) );
  h_assoc2_simpvz.push_back( ibook.book1D("num_assoc(recoToSim)_simpvz", "N of associated tracks (recoToSim) vs. sim PV z", nintPVz, minPVz, maxPVz) );
  h_pileup_simpvz.push_back( ibook.book1D("num_pileup_simpvz", "N of associated (recoToSim) pileup tracks vs. sim PV z", nintPVz, minPVz, maxPVz) );

  h_recochi2.push_back( ibook.book1D("num_reco_chi2","N of reco track vs normalized #chi^{2}",nintChi2,minChi2,maxChi2) );
  h_assoc2chi2.push_back( ibook.book1D("num_assoc(recoToSim)_chi2","N of associated (recoToSim) tracks vs normalized #chi^{2}",nintChi2,minChi2,maxChi2) );
  h_looperchi2.push_back( ibook.book1D("num_duplicate_chi2","N of associated (recoToSim) looper tracks vs normalized #chi^{2}",nintChi2,minChi2,maxChi2) );
  if(!doSeedPlots_) h_misidchi2.push_back( ibook.book1D("num_chargemisid_chi2","N of associated (recoToSim) charge misIDed tracks vs normalized #chi^{2}",nintChi2,minChi2,maxChi2) );
  h_pileupchi2.push_back( ibook.book1D("num_pileup_chi2","N of associated (recoToSim) pileup tracks vs normalized #chi^{2}",nintChi2,minChi2,maxChi2) );


  if(!seedingLayerSetNames.empty()) {
    const auto size = seedingLayerSetNames.size();
    h_reco_seedingLayerSet.push_back(ibook.book1D("num_reco_seedingLayerSet", "N of reco track vs. seedingLayerSet", size,0,size));
    h_assoc2_seedingLayerSet.push_back(ibook.book1D("num_assoc(recoToSim)_seedingLayerSet", "N of associated track (recoToSim) tracks vs. seedingLayerSet", size,0,size));
    h_looper_seedingLayerSet.push_back(ibook.book1D("num_duplicate_seedingLayerSet", "N of reco associated (recoToSim) looper vs. seedingLayerSet", size,0,size));
    h_pileup_seedingLayerSet.push_back(ibook.book1D("num_pileup_seedingLayerSet", "N of reco associated (recoToSim) pileup vs. seedingLayerSet", size,0,size));

    setBinLabels(h_reco_seedingLayerSet.back(), seedingLayerSetNames);
    setBinLabels(h_assoc2_seedingLayerSet.back(), seedingLayerSetNames);
    setBinLabels(h_looper_seedingLayerSet.back(), seedingLayerSetNames);
    setBinLabels(h_pileup_seedingLayerSet.back(), seedingLayerSetNames);
  }

  /////////////////////////////////

  auto bookResolutionPlots1D = [&](std::vector<MonitorElement*>& vec, auto&&... params) {
    vec.push_back( doResolutionPlots ? ibook.book1D(std::forward<decltype(params)>(params)...) : nullptr );
  };
  auto bookResolutionPlots2D = [&](std::vector<MonitorElement*>& vec, auto&&... params) {
    vec.push_back( doResolutionPlots ? ibook.book2D(std::forward<decltype(params)>(params)...) : nullptr );
  };
  auto bookResolutionPlotsProfile2D = [&](std::vector<MonitorElement*>& vec, auto&&... params) {
    vec.push_back( doResolutionPlots ? ibook.bookProfile2D(std::forward<decltype(params)>(params)...) : nullptr );
  };

  bookResolutionPlots1D(h_eta, "eta", "pseudorapidity residue", 1000, -0.1, 0.1);
  bookResolutionPlots1D(h_pt, "pullPt", "pull of p_{t}", 100, -10, 10 );
  bookResolutionPlots1D(h_pullTheta, "pullTheta","pull of #theta parameter",250,-25,25);
  bookResolutionPlots1D(h_pullPhi, "pullPhi","pull of #phi parameter",250,-25,25);
  bookResolutionPlots1D(h_pullDxy, "pullDxy","pull of dxy parameter",250,-25,25);
  bookResolutionPlots1D(h_pullDz, "pullDz","pull of dz parameter",250,-25,25);
  bookResolutionPlots1D(h_pullQoverp, "pullQoverp","pull of qoverp parameter",250,-25,25);

  /* TO BE FIXED -----------
  if (associators[ww]=="TrackAssociatorByChi2"){
    h_assochi2.push_back( ibook.book1D("assocChi2","track association #chi^{2}",1000000,0,100000) );
    h_assochi2_prob.push_back(ibook.book1D("assocChi2_prob","probability of association #chi^{2}",100,0,1));
  } else if (associators[ww]=="quickTrackAssociatorByHits"){
    h_assocFraction.push_back( ibook.book1D("assocFraction","fraction of shared hits",200,0,2) );
    h_assocSharedHit.push_back(ibook.book1D("assocSharedHit","number of shared hits",20,0,20));
  }
  */
  h_assocFraction.push_back( ibook.book1D("assocFraction","fraction of shared hits",200,0,2) );
  h_assocSharedHit.push_back(ibook.book1D("assocSharedHit","number of shared hits",41,-0.5,40.5));
  // ----------------------

  // use the standard error of the mean as the errors in the profile
  chi2_vs_nhits.push_back( ibook.bookProfile("chi2mean_vs_nhits","mean #chi^{2} vs nhits",nintHit,minHit,maxHit, 100,0,10, " ") );

  bookResolutionPlots2D(etares_vs_eta, "etares_vs_eta","etaresidue vs eta",nintEta,minEta,maxEta,200,-0.1,0.1);
  bookResolutionPlots2D(nrec_vs_nsim, "nrec_vs_nsim","Number of selected reco tracks vs. number of selected sim tracks;TrackingParticles;Reco tracks", nintTracks,minTracks,maxTracks, nintTracks,minTracks,maxTracks);

  chi2_vs_eta.push_back( ibook.bookProfile("chi2mean","mean #chi^{2} vs #eta",nintEta,minEta,maxEta, 200, 0, 20, " " ));
  chi2_vs_phi.push_back( ibook.bookProfile("chi2mean_vs_phi","mean #chi^{2} vs #phi",nintPhi,minPhi,maxPhi, 200, 0, 20, " " ) );

  nhits_vs_eta.push_back( ibook.bookProfile("hits_eta","mean hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nPXBhits_vs_eta.push_back( ibook.bookProfile("PXBhits_vs_eta","mean # PXB its vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nPXFhits_vs_eta.push_back( ibook.bookProfile("PXFhits_vs_eta","mean # PXF hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nPXLhits_vs_eta.push_back( ibook.bookProfile("PXLhits_vs_eta","mean # PXL hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nTIBhits_vs_eta.push_back( ibook.bookProfile("TIBhits_vs_eta","mean # TIB hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nTIDhits_vs_eta.push_back( ibook.bookProfile("TIDhits_vs_eta","mean # TID hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nTOBhits_vs_eta.push_back( ibook.bookProfile("TOBhits_vs_eta","mean # TOB hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nTEChits_vs_eta.push_back( ibook.bookProfile("TEChits_vs_eta","mean # TEC hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nSTRIPhits_vs_eta.push_back( ibook.bookProfile("STRIPhits_vs_eta","mean # STRIP hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );

  nLayersWithMeas_vs_eta.push_back( ibook.bookProfile("LayersWithMeas_eta","mean # Layers with measurement vs eta",
                                                      nintEta,minEta,maxEta,nintLayers,minLayers,maxLayers, " ") );
  nPXLlayersWithMeas_vs_eta.push_back( ibook.bookProfile("PXLlayersWithMeas_vs_eta","mean # PXL Layers with measurement vs eta",
                                                         nintEta,minEta,maxEta,nintLayers,minLayers,maxLayers, " ") );
  nSTRIPlayersWithMeas_vs_eta.push_back( ibook.bookProfile("STRIPlayersWithMeas_vs_eta","mean # STRIP Layers with measurement vs eta",
                                                           nintEta,minEta,maxEta,nintLayers,minLayers,maxLayers, " ") );
  nSTRIPlayersWith1dMeas_vs_eta.push_back( ibook.bookProfile("STRIPlayersWith1dMeas_vs_eta","mean # STRIP Layers with 1D measurement vs eta",
                                                             nintEta,minEta,maxEta,nintLayers,minLayers,maxLayers, " ") );
  nSTRIPlayersWith2dMeas_vs_eta.push_back( ibook.bookProfile("STRIPlayersWith2dMeas_vs_eta","mean # STRIP Layers with 2D measurement vs eta",
                                                             nintEta,minEta,maxEta,nintLayers,minLayers,maxLayers, " ") );

  nhits_vs_phi.push_back( ibook.bookProfile("hits_phi","mean # hits vs #phi",nintPhi,minPhi,maxPhi,nintHit,minHit,maxHit, " ") );

  nlosthits_vs_eta.push_back( ibook.bookProfile("losthits_vs_eta","mean # lost hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );

  //resolution of track parameters
  //                       dPt/Pt    cotTheta        Phi            TIP            LIP
  // log10(pt)<0.5        100,0.1    240,0.08     100,0.015      100,0.1000    150,0.3000
  // 0.5<log10(pt)<1.5    100,0.1    120,0.01     100,0.003      100,0.0100    150,0.0500
  // >1.5                 100,0.3    100,0.005    100,0.0008     100,0.0060    120,0.0300

  bookResolutionPlots2D(ptres_vs_eta, "ptres_vs_eta","ptres_vs_eta",
                                      nintEta,minEta,maxEta, ptRes_nbin, ptRes_rangeMin, ptRes_rangeMax);

  bookResolutionPlots2D(ptres_vs_phi, "ptres_vs_phi","p_{t} res vs #phi",
                                      nintPhi,minPhi,maxPhi, ptRes_nbin, ptRes_rangeMin, ptRes_rangeMax);

  bookResolutionPlots2D(ptres_vs_pt, "ptres_vs_pt","ptres_vs_pt",nintPt,minPt,maxPt, ptRes_nbin, ptRes_rangeMin, ptRes_rangeMax);

  bookResolutionPlots2D(cotThetares_vs_eta, "cotThetares_vs_eta","cotThetares_vs_eta",
                                            nintEta,minEta,maxEta,cotThetaRes_nbin, cotThetaRes_rangeMin, cotThetaRes_rangeMax);

  bookResolutionPlots2D(cotThetares_vs_pt, "cotThetares_vs_pt","cotThetares_vs_pt",
                                           nintPt,minPt,maxPt, cotThetaRes_nbin, cotThetaRes_rangeMin, cotThetaRes_rangeMax);


  bookResolutionPlots2D(phires_vs_eta, "phires_vs_eta","phires_vs_eta",
                                       nintEta,minEta,maxEta, phiRes_nbin, phiRes_rangeMin, phiRes_rangeMax);

  bookResolutionPlots2D(phires_vs_pt, "phires_vs_pt","phires_vs_pt",
                                      nintPt,minPt,maxPt, phiRes_nbin, phiRes_rangeMin, phiRes_rangeMax);

  bookResolutionPlots2D(phires_vs_phi, "phires_vs_phi","#phi res vs #phi",
                                       nintPhi,minPhi,maxPhi,phiRes_nbin, phiRes_rangeMin, phiRes_rangeMax);

  bookResolutionPlots2D(dxyres_vs_eta, "dxyres_vs_eta","dxyres_vs_eta",
                                       nintEta,minEta,maxEta,dxyRes_nbin, dxyRes_rangeMin, dxyRes_rangeMax);

  bookResolutionPlots2D(dxyres_vs_pt, "dxyres_vs_pt","dxyres_vs_pt",
                                      nintPt,minPt,maxPt,dxyRes_nbin, dxyRes_rangeMin, dxyRes_rangeMax);

  bookResolutionPlots2D(dzres_vs_eta, "dzres_vs_eta","dzres_vs_eta",
                                      nintEta,minEta,maxEta,dzRes_nbin, dzRes_rangeMin, dzRes_rangeMax);

  bookResolutionPlots2D(dzres_vs_pt, "dzres_vs_pt","dzres_vs_pt",nintPt,minPt,maxPt,dzRes_nbin, dzRes_rangeMin, dzRes_rangeMax);

  bookResolutionPlotsProfile2D(ptmean_vs_eta_phi, "ptmean_vs_eta_phi","mean p_{t} vs #eta and #phi",
                                                  nintPhi,minPhi,maxPhi,nintEta,minEta,maxEta,1000,0,1000);
  bookResolutionPlotsProfile2D(phimean_vs_eta_phi, "phimean_vs_eta_phi","mean #phi vs #eta and #phi",
                                                   nintPhi,minPhi,maxPhi,nintEta,minEta,maxEta,nintPhi,minPhi,maxPhi);

  //pulls of track params vs eta: to be used with fitslicesytool
  bookResolutionPlots2D(dxypull_vs_eta, "dxypull_vs_eta","dxypull_vs_eta",nintEta,minEta,maxEta,100,-10,10);
  bookResolutionPlots2D(ptpull_vs_eta, "ptpull_vs_eta","ptpull_vs_eta",nintEta,minEta,maxEta,100,-10,10);
  bookResolutionPlots2D(dzpull_vs_eta, "dzpull_vs_eta","dzpull_vs_eta",nintEta,minEta,maxEta,100,-10,10);
  bookResolutionPlots2D(phipull_vs_eta, "phipull_vs_eta","phipull_vs_eta",nintEta,minEta,maxEta,100,-10,10);
  bookResolutionPlots2D(thetapull_vs_eta, "thetapull_vs_eta","thetapull_vs_eta",nintEta,minEta,maxEta,100,-10,10);

  //      h_ptshiftetamean.push_back( ibook.book1D("h_ptshifteta_Mean","<#deltapT/pT>[%] vs #eta",nintEta,minEta,maxEta) );


  //pulls of track params vs phi
  bookResolutionPlots2D(ptpull_vs_phi, "ptpull_vs_phi","p_{t} pull vs #phi",nintPhi,minPhi,maxPhi,100,-10,10);
  bookResolutionPlots2D(phipull_vs_phi, "phipull_vs_phi","#phi pull vs #phi",nintPhi,minPhi,maxPhi,100,-10,10);
  bookResolutionPlots2D(thetapull_vs_phi, "thetapull_vs_phi","#theta pull vs #phi",nintPhi,minPhi,maxPhi,100,-10,10);


  bookResolutionPlots2D(nrecHit_vs_nsimHit_rec2sim, "nrecHit_vs_nsimHit_rec2sim","nrecHit vs nsimHit (Rec2simAssoc)", nintHit,minHit,maxHit, nintHit,minHit,maxHit);

  if(useLogPt){
    if(doResolutionPlots) {
      BinLogX(dzres_vs_pt.back()->getTH2F());
      BinLogX(dxyres_vs_pt.back()->getTH2F());
      BinLogX(phires_vs_pt.back()->getTH2F());
      BinLogX(cotThetares_vs_pt.back()->getTH2F());
      BinLogX(ptres_vs_pt.back()->getTH2F());
    }
    BinLogX(h_looperpT.back()->getTH1F());
    if(!doSeedPlots_) BinLogX(h_misidpT.back()->getTH1F());
    BinLogX(h_recopT.back()->getTH1F());
    BinLogX(h_reco2pT.back()->getTH1F());
    BinLogX(h_assoc2pT.back()->getTH1F());
    BinLogX(h_pileuppT.back()->getTH1F());
  }
  if(useLogVertpos) {
    BinLogX(h_loopervertpos.back()->getTH1F());
    BinLogX(h_recovertpos.back()->getTH1F());
    BinLogX(h_assoc2vertpos.back()->getTH1F());
    BinLogX(h_pileupvertpos.back()->getTH1F());
  }
}

void MTVHistoProducerAlgoForTracker::bookRecoPVAssociationHistos(DQMStore::IBooker& ibook){
  h_recodxypv.push_back( ibook.book1D("num_reco_dxypv","N of reco track vs dxy(PV)",nintDxy,minDxy,maxDxy) );
  h_assoc2dxypv.push_back( ibook.book1D("num_assoc(recoToSim)_dxypv","N of associated (recoToSim) tracks vs dxy(PV)",nintDxy,minDxy,maxDxy) );
  h_looperdxypv.push_back( ibook.book1D("num_duplicate_dxypv","N of associated (recoToSim) looper tracks vs dxy(PV)",nintDxy,minDxy,maxDxy) );
  if(!doSeedPlots_) h_misiddxypv.push_back( ibook.book1D("num_chargemisid_dxypv","N of associated (recoToSim) charge misIDed tracks vs dxy(PV)",nintDxy,minDxy,maxDxy) );
  h_pileupdxypv.push_back( ibook.book1D("num_pileup_dxypv","N of associated (recoToSim) pileup tracks vs dxy(PV)",nintDxy,minDxy,maxDxy) );

  h_recodzpv.push_back( ibook.book1D("num_reco_dzpv","N of reco track vs dz(PV)",nintDz,minDz,maxDz) );
  h_assoc2dzpv.push_back( ibook.book1D("num_assoc(recoToSim)_dzpv","N of associated (recoToSim) tracks vs dz(PV)",nintDz,minDz,maxDz) );
  h_looperdzpv.push_back( ibook.book1D("num_duplicate_dzpv","N of associated (recoToSim) looper tracks vs dz(PV)",nintDz,minDz,maxDz) );
  if(!doSeedPlots_) h_misiddzpv.push_back( ibook.book1D("num_chargemisid_versus_dzpv","N of associated (recoToSim) charge misIDed tracks vs dz(PV)",nintDz,minDz,maxDz) );
  h_pileupdzpv.push_back( ibook.book1D("num_pileup_dzpv","N of associated (recoToSim) pileup tracks vs dz(PV)",nintDz,minDz,maxDz) );

  h_recodxypvzoomed.push_back( ibook.book1D("num_reco_dxypv_zoomed","N of reco track vs dxy(PV)",nintDxy,minDxy/dxyDzZoom,maxDxy/dxyDzZoom) );
  h_assoc2dxypvzoomed.push_back( ibook.book1D("num_assoc(recoToSim)_dxypv_zoomed","N of associated (recoToSim) tracks vs dxy(PV)",nintDxy,minDxy/dxyDzZoom,maxDxy/dxyDzZoom) );
  h_looperdxypvzoomed.push_back( ibook.book1D("num_duplicate_dxypv_zoomed","N of associated (recoToSim) looper tracks vs dxy(PV)",nintDxy,minDxy/dxyDzZoom,maxDxy/dxyDzZoom) );
  if(!doSeedPlots_) h_misiddxypvzoomed.push_back( ibook.book1D("num_chargemisid_dxypv_zoomed","N of associated (recoToSim) charge misIDed tracks vs dxy(PV)",nintDxy,minDxy/dxyDzZoom,maxDxy/dxyDzZoom) );
  h_pileupdxypvzoomed.push_back( ibook.book1D("num_pileup_dxypv_zoomed","N of associated (recoToSim) pileup tracks vs dxy(PV)",nintDxy,minDxy/dxyDzZoom,maxDxy/dxyDzZoom) );

  h_recodzpvzoomed.push_back( ibook.book1D("num_reco_dzpv_zoomed","N of reco track vs dz(PV)",nintDz,minDz/dxyDzZoom,maxDz/dxyDzZoom) );
  h_assoc2dzpvzoomed.push_back( ibook.book1D("num_assoc(recoToSim)_dzpv_zoomed","N of associated (recoToSim) tracks vs dz(PV)",nintDz,minDz/dxyDzZoom,maxDz/dxyDzZoom) );
  h_looperdzpvzoomed.push_back( ibook.book1D("num_duplicate_dzpv_zoomed","N of associated (recoToSim) looper tracks vs dz(PV)",nintDz,minDz/dxyDzZoom,maxDz/dxyDzZoom) );
  if(!doSeedPlots_) h_misiddzpvzoomed.push_back( ibook.book1D("num_chargemisid_versus_dzpv_zoomed","N of associated (recoToSim) charge misIDed tracks vs dz(PV)",nintDz,minDz/dxyDzZoom,maxDz/dxyDzZoom) );
  h_pileupdzpvzoomed.push_back( ibook.book1D("num_pileup_dzpv_zoomed","N of associated (recoToSim) pileup tracks vs dz(PV)",nintDz,minDz/dxyDzZoom,maxDz/dxyDzZoom) );

  h_reco_dzpvcut.push_back( ibook.book1D("num_reco_dzpvcut","N of reco track vs dz(PV)",nintDzpvCum,0,maxDzpvCum) );
  h_assoc2_dzpvcut.push_back( ibook.book1D("num_assoc(recoToSim)_dzpvcut","N of associated (recoToSim) tracks vs dz(PV)",nintDzpvCum,0,maxDzpvCum) );
  h_pileup_dzpvcut.push_back( ibook.book1D("num_pileup_dzpvcut", "N of associated (recoToSim) pileup tracks vs dz(PV)",nintDzpvCum,0,maxDzpvCum) );

  h_reco_dzpvcut_pt.push_back( ibook.book1D("num_reco_dzpvcut_pt","#sump_{T} of reco track vs dz(PV)",nintDzpvCum,0,maxDzpvCum) );
  h_assoc2_dzpvcut_pt.push_back( ibook.book1D("num_assoc(recoToSim)_dzpvcut_pt","#sump_{T} of associated (recoToSim) tracks vs dz(PV)",nintDzpvCum,0,maxDzpvCum) );
  h_pileup_dzpvcut_pt.push_back( ibook.book1D("num_pileup_dzpvcut_pt", "#sump_{T} of associated (recoToSim) pileup tracks vs dz(PV)",nintDzpvCum,0,maxDzpvCum) );
  h_reco_dzpvcut_pt.back()->getTH1()->Sumw2();
  h_assoc2_dzpvcut_pt.back()->getTH1()->Sumw2();
  h_pileup_dzpvcut_pt.back()->getTH1()->Sumw2();

  h_reco_dzpvsigcut.push_back( ibook.book1D("num_reco_dzpvsigcut","N of reco track vs dz(PV)/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );
  h_assoc2_dzpvsigcut.push_back( ibook.book1D("num_assoc(recoToSim)_dzpvsigcut","N of associated (recoToSim) tracks vs dz(PV)/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );
  h_pileup_dzpvsigcut.push_back( ibook.book1D("num_pileup_dzpvsigcut","N of associated (recoToSim) pileup tracks vs dz(PV)/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );

  h_reco_dzpvsigcut_pt.push_back( ibook.book1D("num_reco_dzpvsigcut_pt","#sump_{T} of reco track vs dz(PV)/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );
  h_assoc2_dzpvsigcut_pt.push_back( ibook.book1D("num_assoc(recoToSim)_dzpvsigcut_pt","#sump_{T} of associated (recoToSim) tracks vs dz(PV)/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );
  h_pileup_dzpvsigcut_pt.push_back( ibook.book1D("num_pileup_dzpvsigcut_pt","#sump_{T} of associated (recoToSim) pileup tracks vs dz(PV)/dzError",nintDzpvsigCum,0,maxDzpvsigCum) );
  h_reco_dzpvsigcut_pt.back()->getTH1()->Sumw2();
  h_assoc2_dzpvsigcut_pt.back()->getTH1()->Sumw2();
  h_pileup_dzpvsigcut_pt.back()->getTH1()->Sumw2();

}

void MTVHistoProducerAlgoForTracker::bookRecodEdxHistos(DQMStore::IBooker& ibook) {
  // dE/dx stuff
  h_dedx_estim.emplace_back(std::initializer_list<MonitorElement*>{
      ibook.book1D("h_dedx_estim1","dE/dx estimator 1",nintDeDx,minDeDx,maxDeDx),
      ibook.book1D("h_dedx_estim2","dE/dx estimator 2",nintDeDx,minDeDx,maxDeDx)
      });
  h_dedx_nom.emplace_back(std::initializer_list<MonitorElement*>{
      ibook.book1D("h_dedx_nom1","dE/dx number of measurements",nintHit,minHit,maxHit),
      ibook.book1D("h_dedx_nom2","dE/dx number of measurements",nintHit,minHit,maxHit)
      });
  h_dedx_sat.emplace_back(std::initializer_list<MonitorElement*>{
      ibook.book1D("h_dedx_sat1","dE/dx number of measurements with saturation",nintHit,minHit,maxHit),
      ibook.book1D("h_dedx_sat2","dE/dx number of measurements with saturation",nintHit,minHit,maxHit)
      });
}

void MTVHistoProducerAlgoForTracker::bookSeedHistos(DQMStore::IBooker& ibook) {
  h_seedsFitFailed.push_back(ibook.book1D("seeds_fitFailed", "Number of seeds for which the fit failed", nintTracks, minTracks, maxTracks));
  h_seedsFitFailedFraction.push_back(ibook.book1D("seeds_fitFailedFraction", "Fraction of seeds for which the fit failed", 100, 0, 1));
}

void MTVHistoProducerAlgoForTracker::bookMVAHistos(DQMStore::IBooker& ibook, size_t nMVAs) {
  h_reco_mva.emplace_back();
  h_assoc2_mva.emplace_back();

  h_reco_mvacut.emplace_back();
  h_assoc_mvacut.emplace_back();
  h_assoc2_mvacut.emplace_back();
  h_simul2_mvacut.emplace_back();

  h_reco_mva_hp.emplace_back();
  h_assoc2_mva_hp.emplace_back();

  h_reco_mvacut_hp.emplace_back();
  h_assoc_mvacut_hp.emplace_back();
  h_assoc2_mvacut_hp.emplace_back();
  h_simul2_mvacut_hp.emplace_back();

  h_assoc2_mva_vs_pt.emplace_back();
  h_fake_mva_vs_pt.emplace_back();
  h_assoc2_mva_vs_pt_hp.emplace_back();
  h_fake_mva_vs_pt_hp.emplace_back();
  h_assoc2_mva_vs_eta.emplace_back();
  h_fake_mva_vs_eta.emplace_back();
  h_assoc2_mva_vs_eta_hp.emplace_back();
  h_fake_mva_vs_eta_hp.emplace_back();

  for(size_t i=1; i <= nMVAs; ++i) {
    auto istr = std::to_string(i);
    std::string pfix;

    if(i==1) {
      h_reco_mva_hp.back().push_back(nullptr);
      h_assoc2_mva_hp.back().push_back(nullptr);

      h_reco_mvacut_hp.back().push_back(nullptr);
      h_assoc_mvacut_hp.back().push_back(nullptr);
      h_assoc2_mvacut_hp.back().push_back(nullptr);
      h_simul2_mvacut_hp.back().push_back(nullptr);

      h_assoc2_mva_vs_pt_hp.back().push_back(nullptr);
      h_fake_mva_vs_pt_hp.back().push_back(nullptr);
      h_assoc2_mva_vs_eta_hp.back().push_back(nullptr);
      h_fake_mva_vs_eta_hp.back().push_back(nullptr);
    }
    else {
      pfix = " (not loose-selected)";
      std::string pfix2 = " (not HP-selected)";

      h_reco_mva_hp.back().push_back(ibook.book1D("num_reco_mva"+istr+"_hp", "N of reco track after vs MVA"+istr+pfix2, nintMVA, minMVA, maxMVA) );
      h_assoc2_mva_hp.back().push_back(ibook.book1D("num_assoc(recoToSim)_mva"+istr+"_hp", "N of associated tracks (recoToSim) vs MVA"+istr+pfix2, nintMVA, minMVA, maxMVA) );

      h_reco_mvacut_hp.back().push_back(ibook.book1D("num_reco_mva"+istr+"cut"+"_hp", "N of reco track vs cut on MVA"+istr+pfix2, nintMVA, minMVA, maxMVA) );
      h_assoc_mvacut_hp.back().push_back(ibook.book1D("num_assoc(simToReco)_mva"+istr+"cut_hp", "N of associated tracks (simToReco) vs cut on MVA"+istr+pfix2, nintMVA, minMVA, maxMVA) );
      h_assoc2_mvacut_hp.back().push_back(ibook.book1D("num_assoc(recoToSim)_mva"+istr+"cut_hp", "N of associated tracks (recoToSim) vs cut on MVA"+istr+pfix2, nintMVA, minMVA, maxMVA) );
      h_simul2_mvacut_hp.back().push_back(ibook.book1D("num_simul2_mva"+istr+"cut_hp", "N of simulated tracks (associated to any track) vs cut on MVA"+istr+pfix2, nintMVA, minMVA, maxMVA) );

      h_assoc2_mva_vs_pt_hp.back().push_back(ibook.bookProfile("mva_assoc(recoToSim)_mva"+istr+"_pT_hp", "MVA"+istr+" of associated tracks (recoToSim) vs. track p_{T}"+pfix2, nintPt, minPt, maxPt, nintMVA, minMVA, maxMVA));
      h_fake_mva_vs_pt_hp.back().push_back(ibook.bookProfile("mva_fake_mva"+istr+"pT_hp", "MVA"+istr+" of non-associated tracks (recoToSim) vs. track p_{T}"+pfix2, nintPt, minPt, maxPt, nintMVA, minMVA, maxMVA));
      h_assoc2_mva_vs_eta_hp.back().push_back(ibook.bookProfile("mva_assoc(recoToSim)_mva"+istr+"_eta_hp", "MVA"+istr+" of associated tracks (recoToSim) vs. track #eta"+pfix2, nintEta, minEta, maxEta, nintMVA, minMVA, maxMVA));
      h_fake_mva_vs_eta_hp.back().push_back(ibook.bookProfile("mva_fake_mva"+istr+"eta_hp", "MVA"+istr+" of non-associated tracks (recoToSim) vs. track #eta"+pfix2, nintEta, minEta, maxEta, nintMVA, minMVA, maxMVA));
    }

    h_reco_mva.back().push_back(ibook.book1D("num_reco_mva"+istr, "N of reco track vs MVA"+istr+pfix, nintMVA, minMVA, maxMVA) );
    h_assoc2_mva.back().push_back(ibook.book1D("num_assoc(recoToSim)_mva"+istr, "N of associated tracks (recoToSim) vs MVA"+istr+pfix, nintMVA, minMVA, maxMVA) );

    h_reco_mvacut.back().push_back(ibook.book1D("num_reco_mva"+istr+"cut", "N of reco track vs cut on MVA"+istr+pfix, nintMVA, minMVA, maxMVA) );
    h_assoc_mvacut.back().push_back(ibook.book1D("num_assoc(simToReco)_mva"+istr+"cut", "N of associated tracks (simToReco) vs cut on MVA"+istr+pfix, nintMVA, minMVA, maxMVA) );
    h_assoc2_mvacut.back().push_back(ibook.book1D("num_assoc(recoToSim)_mva"+istr+"cut", "N of associated tracks (recoToSim) vs cut on MVA"+istr+pfix, nintMVA, minMVA, maxMVA) );
    h_simul2_mvacut.back().push_back(ibook.book1D("num_simul2_mva"+istr+"cut", "N of simulated tracks (associated to any track) vs cut on MVA"+istr+pfix, nintMVA, minMVA, maxMVA) );

    h_assoc2_mva_vs_pt.back().push_back(ibook.bookProfile("mva_assoc(recoToSim)_mva"+istr+"_pT", "MVA"+istr+" of associated tracks (recoToSim) vs. track p_{T}"+pfix, nintPt, minPt, maxPt, nintMVA, minMVA, maxMVA));
    h_fake_mva_vs_pt.back().push_back(ibook.bookProfile("mva_fake_mva"+istr+"_pT", "MVA"+istr+" of non-associated tracks (recoToSim) vs. track p_{T}"+pfix, nintPt, minPt, maxPt, nintMVA, minMVA, maxMVA));
    h_assoc2_mva_vs_eta.back().push_back(ibook.bookProfile("mva_assoc(recoToSim)_mva"+istr+"_eta", "MVA"+istr+" of associated tracks (recoToSim) vs. track #eta"+pfix, nintEta, minEta, maxEta, nintMVA, minMVA, maxMVA));
    h_fake_mva_vs_eta.back().push_back(ibook.bookProfile("mva_fake_mva"+istr+"_eta", "MVA"+istr+" of non-associated tracks (recoToSim) vs. track #eta"+pfix, nintEta, minEta, maxEta, nintMVA, minMVA, maxMVA));

    if(useLogPt){
      BinLogX(h_assoc2_mva_vs_pt.back().back()->getTProfile());
      BinLogX(h_fake_mva_vs_pt.back().back()->getTProfile());
      if(i > 1) {
        BinLogX(h_assoc2_mva_vs_pt_hp.back().back()->getTProfile());
        BinLogX(h_fake_mva_vs_pt_hp.back().back()->getTProfile());
      }
    }
  }
}

void MTVHistoProducerAlgoForTracker::fill_generic_simTrack_histos(const TrackingParticle::Vector& momentumTP,
								  const TrackingParticle::Point& vertexTP,
                                                                  int bx){
  if(bx == 0) {
    h_ptSIM->Fill(sqrt(momentumTP.perp2()));
    h_etaSIM->Fill(momentumTP.eta());
    h_vertposSIM->Fill(sqrt(vertexTP.perp2()));
  }
  h_bunchxSIM->Fill(bx);
}



void MTVHistoProducerAlgoForTracker::fill_recoAssociated_simTrack_histos(int count,
									 const TrackingParticle& tp,
									 const TrackingParticle::Vector& momentumTP,
									 const TrackingParticle::Point& vertexTP,
									 double dxySim, double dzSim,
									 double dxyPVSim, double dzPVSim,
                                                                         int nSimHits,
                                                                         int nSimLayers, int nSimPixelLayers, int nSimStripMonoAndStereoLayers,
									 const reco::Track* track,
									 int numVertices,
									 double dR,
									 const math::XYZPoint *pvPosition,
                                                                         const TrackingVertex::LorentzVector *simPVPosition,
                                                                         const math::XYZPoint& bsPosition,
                                                                         const std::vector<float>& mvas,
                                                                         unsigned int selectsLoose, unsigned int selectsHP) {
  bool isMatched = track;
  const auto eta = getEta(momentumTP.eta());
  const auto phi = momentumTP.phi();
  const auto pt = getPt(sqrt(momentumTP.perp2()));
  const auto nSim3DLayers = nSimPixelLayers + nSimStripMonoAndStereoLayers;

  const auto vertexTPwrtBS = vertexTP - bsPosition;
  const auto vertxy = std::sqrt(vertexTPwrtBS.perp2());
  const auto vertz = vertexTPwrtBS.z();

  //efficiency vs. cut on MVA
  //
  // Note that this includes also pileup TPs, as "signalOnly"
  // selection is applied only in the TpSelector*. Have to think if
  // this is really what we want.
  if(isMatched) {
    for(size_t i=0; i<mvas.size(); ++i) {
      if(i<=selectsLoose) {
        h_simul2_mvacut[count][i]->Fill(maxMVA);
        h_assoc_mvacut[count][i]->Fill(mvas[i]);
      }
      if(i>=1 && i<=selectsHP) {
        h_simul2_mvacut_hp[count][i]->Fill(maxMVA);
        h_assoc_mvacut_hp[count][i]->Fill(mvas[i]);
      }
    }
  }

  if((*TpSelectorForEfficiencyVsEta)(tp)){
    //effic vs eta
    fillPlotNoFlow(h_simuleta[count], eta);
    if (isMatched) fillPlotNoFlow(h_assoceta[count], eta);
  }

  if((*TpSelectorForEfficiencyVsPhi)(tp)){
    fillPlotNoFlow(h_simulphi[count], phi);
    if (isMatched) fillPlotNoFlow(h_assocphi[count], phi);
    //effic vs hits
    fillPlotNoFlow(h_simulhit[count], nSimHits);
    fillPlotNoFlow(h_simullayer[count], nSimLayers);
    fillPlotNoFlow(h_simulpixellayer[count], nSimPixelLayers);
    fillPlotNoFlow(h_simul3Dlayer[count], nSim3DLayers);
    if(isMatched) {
      fillPlotNoFlow(h_assochit[count], nSimHits);
      fillPlotNoFlow(h_assoclayer[count], nSimLayers);
      fillPlotNoFlow(h_assocpixellayer[count], nSimPixelLayers);
      fillPlotNoFlow(h_assoc3Dlayer[count], nSim3DLayers);
      if(nrecHit_vs_nsimHit_sim2rec[count]) nrecHit_vs_nsimHit_sim2rec[count]->Fill( track->numberOfValidHits(),nSimHits);
    }
    //effic vs pu
    fillPlotNoFlow(h_simulpu[count], numVertices);
    if(isMatched) fillPlotNoFlow(h_assocpu[count],numVertices);
    //efficiency vs dR
    fillPlotNoFlow(h_simuldr[count],dR);
    if (isMatched) fillPlotNoFlow(h_assocdr[count],dR);
  }

  if((*TpSelectorForEfficiencyVsPt)(tp)){
    fillPlotNoFlow(h_simulpT[count], pt);
    if (isMatched) fillPlotNoFlow(h_assocpT[count], pt);
  }

  if((*TpSelectorForEfficiencyVsVTXR)(tp)){
    fillPlotNoFlow(h_simuldxy[count],dxySim);
    if (isMatched) fillPlotNoFlow(h_assocdxy[count],dxySim);
    if(pvPosition) {
      fillPlotNoFlow(h_simuldxypv[count], dxyPVSim);
      fillPlotNoFlow(h_simuldxypvzoomed[count], dxyPVSim);
      if (isMatched) {
        fillPlotNoFlow(h_assocdxypv[count], dxyPVSim);
        fillPlotNoFlow(h_assocdxypvzoomed[count], dxyPVSim);
      }
    }

    fillPlotNoFlow(h_simulvertpos[count], vertxy);
    if (isMatched) fillPlotNoFlow(h_assocvertpos[count], vertxy);
  }


  if((*TpSelectorForEfficiencyVsVTXZ)(tp)){
    fillPlotNoFlow(h_simuldz[count],dzSim);
    if (isMatched) fillPlotNoFlow(h_assocdz[count],dzSim);

    fillPlotNoFlow(h_simulzpos[count], vertz);
    if (isMatched) fillPlotNoFlow(h_assoczpos[count], vertz);

    if(pvPosition) {
      fillPlotNoFlow(h_simuldzpv[count], dzPVSim);
      fillPlotNoFlow(h_simuldzpvzoomed[count], dzPVSim);

      h_simul_dzpvcut[count]->Fill(0);
      h_simul_dzpvsigcut[count]->Fill(0);
      h_simul_dzpvcut_pt[count]->Fill(0, pt);
      h_simul_dzpvsigcut_pt[count]->Fill(0, pt);

      if(isMatched) {
        fillPlotNoFlow(h_assocdzpv[count], dzPVSim);
        fillPlotNoFlow(h_assocdzpvzoomed[count], dzPVSim);

        h_simul2_dzpvcut[count]->Fill(0);
        h_simul2_dzpvsigcut[count]->Fill(0);
        h_simul2_dzpvcut_pt[count]->Fill(0, pt);
        h_simul2_dzpvsigcut_pt[count]->Fill(0, pt);
        const double dzpvcut = std::abs(track->dz(*pvPosition));
        const double dzpvsigcut = dzpvcut / track->dzError();
        h_assoc_dzpvcut[count]->Fill(dzpvcut);
        h_assoc_dzpvsigcut[count]->Fill(dzpvsigcut);
        h_assoc_dzpvcut_pt[count]->Fill(dzpvcut, pt);
        h_assoc_dzpvsigcut_pt[count]->Fill(dzpvsigcut, pt);
      }
    }
    if(simPVPosition) {
      const auto simpvz = simPVPosition->z();
      h_simul_simpvz[count]->Fill(simpvz);
      if(isMatched) {
        h_assoc_simpvz[count]->Fill(simpvz);
      }
    }
  }

}

void MTVHistoProducerAlgoForTracker::fill_duplicate_histos(int count,
                                                           const reco::Track& track1,
                                                           const reco::Track& track2) {
  h_duplicates_oriAlgo_vs_oriAlgo[count]->Fill(track1.originalAlgo(), track2.originalAlgo());
}

void MTVHistoProducerAlgoForTracker::fill_simTrackBased_histos(int numSimTracks){
  h_tracksSIM->Fill(numSimTracks);
}

// dE/dx
void MTVHistoProducerAlgoForTracker::fill_dedx_recoTrack_histos(int count, const edm::RefToBase<reco::Track>& trackref, const std::vector< const edm::ValueMap<reco::DeDxData> *>& v_dEdx) {
  for (unsigned int i=0; i<v_dEdx.size(); i++) {
    const edm::ValueMap<reco::DeDxData>& dEdxTrack = *(v_dEdx[i]);
    const reco::DeDxData& dedx = dEdxTrack[trackref];
    h_dedx_estim[count][i]->Fill(dedx.dEdx());
    h_dedx_nom[count][i]->Fill(dedx.numberOfMeasurements());
    h_dedx_sat[count][i]->Fill(dedx.numberOfSaturatedMeasurements());
  }
}


void MTVHistoProducerAlgoForTracker::fill_generic_recoTrack_histos(int count,
								   const reco::Track& track,
                                                                   const TrackerTopology& ttopo,
								   const math::XYZPoint& bsPosition,
								   const math::XYZPoint *pvPosition,
                                                                   const TrackingVertex::LorentzVector *simPVPosition,
								   bool isMatched,
								   bool isSigMatched,
								   bool isChargeMatched,
								   int numAssocRecoTracks,
								   int numVertices,
								   int nSimHits,
								   double sharedFraction,
								   double dR,
                                                                   const std::vector<float>& mvas,
                                                                   unsigned int selectsLoose, unsigned int selectsHP) {

  //Fill track algo histogram
  fillPlotNoFlow(h_algo[count],track.algo());
  int sharedHits = sharedFraction *  track.numberOfValidHits();

  //Compute fake rate vs eta
  const auto eta = getEta(track.momentum().eta());
  const auto phi = track.momentum().phi();
  const auto pt = getPt(sqrt(track.momentum().perp2()));
  const auto dxy = track.dxy(bsPosition);
  const auto dz = track.dz(bsPosition);
  const auto dxypv = pvPosition ? track.dxy(*pvPosition) : 0.0;
  const auto dzpv = pvPosition ? track.dz(*pvPosition) : 0.0;
  const auto dzpvsig = pvPosition ? dzpv / track.dzError() : 0.0;
  const auto nhits = track.found();
  const auto nlayers = track.hitPattern().trackerLayersWithMeasurement();
  const auto nPixelLayers = track.hitPattern().pixelLayersWithMeasurement();
  const auto n3DLayers = nPixelLayers + track.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
  const auto refPointWrtBS = track.referencePoint() - bsPosition;
  const auto vertxy = std::sqrt(refPointWrtBS.perp2());
  const auto vertz = refPointWrtBS.z();
  const auto deltar = min(max(dR,h_recodr[count]->getTH1()->GetXaxis()->GetXmin()),h_recodr[count]->getTH1()->GetXaxis()->GetXmax());
  const auto chi2 = track.normalizedChi2();
  const bool fillSeedingLayerSets = !seedingLayerSetNames.empty();
  const unsigned int seedingLayerSetBin = fillSeedingLayerSets ? getSeedingLayerSetBin(track, ttopo) : 0;
  const auto simpvz = simPVPosition ? simPVPosition->z() : 0.0;

  const bool paramsValid = !trackFromSeedFitFailed(track);

  if(paramsValid) {
    fillPlotNoFlow(h_recoeta[count], eta);
    fillPlotNoFlow(h_recophi[count], phi);
    fillPlotNoFlow(h_recopT[count], pt);
    fillPlotNoFlow(h_recodxy[count], dxy);
    fillPlotNoFlow(h_recodz[count], dz);
    fillPlotNoFlow(h_recochi2[count], chi2);
    fillPlotNoFlow(h_recovertpos[count], vertxy);
    fillPlotNoFlow(h_recozpos[count], vertz);
    h_recodr[count]->Fill(deltar);
    if(fillSeedingLayerSets) h_reco_seedingLayerSet[count]->Fill(seedingLayerSetBin);
    if(pvPosition) {
      fillPlotNoFlow(h_recodxypv[count], dxypv);
      fillPlotNoFlow(h_recodzpv[count], dzpv);
      fillPlotNoFlow(h_recodxypvzoomed[count], dxypv);
      fillPlotNoFlow(h_recodzpvzoomed[count], dzpv);

      h_reco_dzpvcut[count]->Fill(std::abs(dzpv));
      h_reco_dzpvsigcut[count]->Fill(std::abs(dzpvsig));
      h_reco_dzpvcut_pt[count]->Fill(std::abs(dzpv), pt);
      h_reco_dzpvsigcut_pt[count]->Fill(std::abs(dzpvsig), pt);
    }
    if(simPVPosition) {
      h_reco_simpvz[count]->Fill(simpvz);
    }
    if((*trackSelectorVsEta)(track)) {
      fillPlotNoFlow(h_reco2eta[count], eta);
    }
    if((*trackSelectorVsPt)(track)) {
      fillPlotNoFlow(h_reco2pT[count], pt);
    }
  }
  fillPlotNoFlow(h_recohit[count], nhits);
  fillPlotNoFlow(h_recolayer[count], nlayers);
  fillPlotNoFlow(h_recopixellayer[count], nPixelLayers);
  fillPlotNoFlow(h_reco3Dlayer[count], n3DLayers);
  fillPlotNoFlow(h_recopu[count],numVertices);
  if((*trackSelectorVsPhi)(track)) {
    fillPlotNoFlow(h_reco2pu[count], numVertices);
  }

  fillMVAHistos(h_reco_mva[count], h_reco_mvacut[count], h_reco_mva_hp[count], h_reco_mvacut_hp[count], mvas, selectsLoose, selectsHP);

  if (isMatched) {
    if(paramsValid) {
      fillPlotNoFlow(h_assoc2eta[count], eta);
      fillPlotNoFlow(h_assoc2phi[count], phi);
      fillPlotNoFlow(h_assoc2pT[count], pt);
      fillPlotNoFlow(h_assoc2dxy[count], dxy);
      fillPlotNoFlow(h_assoc2dz[count], dz);
      fillPlotNoFlow(h_assoc2hit[count], nhits);
      fillPlotNoFlow(h_assoc2chi2[count], chi2);
      fillPlotNoFlow(h_assoc2vertpos[count], vertxy);
      fillPlotNoFlow(h_assoc2zpos[count], vertz);
      h_assoc2dr[count]->Fill(deltar);
      if(fillSeedingLayerSets) h_assoc2_seedingLayerSet[count]->Fill(seedingLayerSetBin);
      if(pvPosition) {
        fillPlotNoFlow(h_assoc2dxypv[count], dxypv);
        fillPlotNoFlow(h_assoc2dzpv[count], dzpv);
        fillPlotNoFlow(h_assoc2dxypvzoomed[count], dxypv);
        fillPlotNoFlow(h_assoc2dzpvzoomed[count], dzpv);

        h_assoc2_dzpvcut[count]->Fill(std::abs(dzpv));
        h_assoc2_dzpvsigcut[count]->Fill(std::abs(dzpvsig));
        h_assoc2_dzpvcut_pt[count]->Fill(std::abs(dzpv), pt);
        h_assoc2_dzpvsigcut_pt[count]->Fill(std::abs(dzpvsig), pt);
      }
      if(simPVPosition) {
        h_assoc2_simpvz[count]->Fill(simpvz);
      }
    }
    fillPlotNoFlow(h_assoc2layer[count], nlayers);
    fillPlotNoFlow(h_assoc2pixellayer[count], nPixelLayers);
    fillPlotNoFlow(h_assoc23Dlayer[count], n3DLayers);
    fillPlotNoFlow(h_assoc2pu[count],numVertices);

    fillMVAHistos(h_assoc2_mva[count], h_assoc2_mvacut[count], h_assoc2_mva_hp[count], h_assoc2_mvacut_hp[count], mvas, selectsLoose, selectsHP);
    fillMVAHistos(pt, h_assoc2_mva_vs_pt[count], h_assoc2_mva_vs_pt_hp[count], mvas, selectsLoose, selectsHP);
    fillMVAHistos(eta, h_assoc2_mva_vs_eta[count], h_assoc2_mva_vs_eta_hp[count], mvas, selectsLoose, selectsHP);

    if(nrecHit_vs_nsimHit_rec2sim[count]) nrecHit_vs_nsimHit_rec2sim[count]->Fill( track.numberOfValidHits(),nSimHits);
    h_assocFraction[count]->Fill( sharedFraction);
    h_assocSharedHit[count]->Fill( sharedHits);

    if (!doSeedPlots_ && !isChargeMatched) {
      fillPlotNoFlow(h_misideta[count], eta);
      fillPlotNoFlow(h_misidphi[count], phi);
      fillPlotNoFlow(h_misidpT[count], pt);
      fillPlotNoFlow(h_misiddxy[count], dxy);
      fillPlotNoFlow(h_misiddz[count], dz);
      fillPlotNoFlow(h_misidhit[count], nhits);
      fillPlotNoFlow(h_misidlayer[count], nlayers);
      fillPlotNoFlow(h_misidpixellayer[count], nPixelLayers);
      fillPlotNoFlow(h_misid3Dlayer[count], n3DLayers);
      fillPlotNoFlow(h_misidpu[count], numVertices);
      fillPlotNoFlow(h_misidchi2[count], chi2);
      if(pvPosition) {
        fillPlotNoFlow(h_misiddxypv[count], dxypv);
        fillPlotNoFlow(h_misiddzpv[count], dzpv);
        fillPlotNoFlow(h_misiddxypvzoomed[count], dxypv);
        fillPlotNoFlow(h_misiddzpvzoomed[count], dzpv);
      }
    }

    if (numAssocRecoTracks>1) {
      if(paramsValid) {
        fillPlotNoFlow(h_loopereta[count], eta);
        fillPlotNoFlow(h_looperphi[count], phi);
        fillPlotNoFlow(h_looperpT[count], pt);
        fillPlotNoFlow(h_looperdxy[count], dxy);
        fillPlotNoFlow(h_looperdz[count], dz);
        fillPlotNoFlow(h_looperchi2[count], chi2);
        fillPlotNoFlow(h_loopervertpos[count], vertxy);
        fillPlotNoFlow(h_looperzpos[count], vertz);
        h_looperdr[count]->Fill(deltar);
        if(fillSeedingLayerSets) h_looper_seedingLayerSet[count]->Fill(seedingLayerSetBin);
        if(pvPosition) {
          fillPlotNoFlow(h_looperdxypv[count], dxypv);
          fillPlotNoFlow(h_looperdzpv[count], dzpv);
          fillPlotNoFlow(h_looperdxypvzoomed[count], dxypv);
          fillPlotNoFlow(h_looperdzpvzoomed[count], dzpv);
        }
      }
      fillPlotNoFlow(h_looperhit[count], nhits);
      fillPlotNoFlow(h_looperlayer[count], nlayers);
      fillPlotNoFlow(h_looperpixellayer[count], nPixelLayers);
      fillPlotNoFlow(h_looper3Dlayer[count], n3DLayers);
      fillPlotNoFlow(h_looperpu[count], numVertices);
    }
    if(!isSigMatched) {
      if(paramsValid) {
        fillPlotNoFlow(h_pileupeta[count], eta);
        fillPlotNoFlow(h_pileupphi[count], phi);
        fillPlotNoFlow(h_pileuppT[count], pt);
        fillPlotNoFlow(h_pileupdxy[count], dxy);
        fillPlotNoFlow(h_pileupdz[count], dz);
        fillPlotNoFlow(h_pileupchi2[count], chi2);
        fillPlotNoFlow(h_pileupvertpos[count], vertxy);
        fillPlotNoFlow(h_pileupzpos[count], vertz);
        h_pileupdr[count]->Fill(deltar);
        if(fillSeedingLayerSets) h_pileup_seedingLayerSet[count]->Fill(seedingLayerSetBin);
        if(pvPosition) {
          fillPlotNoFlow(h_pileupdxypv[count], dxypv);
          fillPlotNoFlow(h_pileupdzpv[count], dzpv);
          fillPlotNoFlow(h_pileupdxypvzoomed[count], dxypv);
          fillPlotNoFlow(h_pileupdzpvzoomed[count], dzpv);

          h_pileup_dzpvcut[count]->Fill(std::abs(dzpv));
          h_pileup_dzpvsigcut[count]->Fill(std::abs(dzpvsig));
          h_pileup_dzpvcut_pt[count]->Fill(std::abs(dzpv), pt);
          h_pileup_dzpvsigcut_pt[count]->Fill(std::abs(dzpvsig), pt);
        }
        if(simPVPosition) {
          h_pileup_simpvz[count]->Fill(simpvz);
        }
      }
      fillPlotNoFlow(h_pileuphit[count], nhits);
      fillPlotNoFlow(h_pileuplayer[count], nlayers);
      fillPlotNoFlow(h_pileuppixellayer[count], nPixelLayers);
      fillPlotNoFlow(h_pileup3Dlayer[count], n3DLayers);
      fillPlotNoFlow(h_pileuppu[count], numVertices);
    }
  }
  else { // !isMatched
    fillMVAHistos(pt, h_fake_mva_vs_pt[count], h_fake_mva_vs_pt_hp[count], mvas, selectsLoose, selectsHP);
    fillMVAHistos(eta, h_fake_mva_vs_eta[count], h_fake_mva_vs_eta_hp[count], mvas, selectsLoose, selectsHP);
  }
}


void MTVHistoProducerAlgoForTracker::fill_simAssociated_recoTrack_histos(int count,
									 const reco::Track& track){
    //nchi2 and hits global distributions
    h_hits[count]->Fill(track.numberOfValidHits());
    h_losthits[count]->Fill(track.numberOfLostHits());
    h_nmisslayers_inner[count]->Fill(track.hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS));
    h_nmisslayers_outer[count]->Fill(track.hitPattern().numberOfHits(reco::HitPattern::MISSING_OUTER_HITS));
    if(trackFromSeedFitFailed(track))
      return;

    h_nchi2[count]->Fill(track.normalizedChi2());
    h_nchi2_prob[count]->Fill(TMath::Prob(track.chi2(),(int)track.ndof()));
    chi2_vs_nhits[count]->Fill(track.numberOfValidHits(),track.normalizedChi2());
    h_charge[count]->Fill( track.charge() );

    //chi2 and #hit vs eta: fill 2D histos
    const auto eta = getEta(track.eta());
    chi2_vs_eta[count]->Fill(eta, track.normalizedChi2());
    nhits_vs_eta[count]->Fill(eta, track.numberOfValidHits());
    const auto pxbHits = track.hitPattern().numberOfValidPixelBarrelHits();
    const auto pxfHits = track.hitPattern().numberOfValidPixelEndcapHits();
    const auto tibHits = track.hitPattern().numberOfValidStripTIBHits();
    const auto tidHits = track.hitPattern().numberOfValidStripTIDHits();
    const auto tobHits = track.hitPattern().numberOfValidStripTOBHits();
    const auto tecHits = track.hitPattern().numberOfValidStripTECHits();
    nPXBhits_vs_eta[count]->Fill(eta, pxbHits);
    nPXFhits_vs_eta[count]->Fill(eta, pxfHits);
    nPXLhits_vs_eta[count]->Fill(eta, pxbHits+pxfHits);
    nTIBhits_vs_eta[count]->Fill(eta, tibHits);
    nTIDhits_vs_eta[count]->Fill(eta, tidHits);
    nTOBhits_vs_eta[count]->Fill(eta, tobHits);
    nTEChits_vs_eta[count]->Fill(eta, tecHits);
    nSTRIPhits_vs_eta[count]->Fill(eta, tibHits+tidHits+tobHits+tecHits);
    nLayersWithMeas_vs_eta[count]->Fill(eta, track.hitPattern().trackerLayersWithMeasurement());
    nPXLlayersWithMeas_vs_eta[count]->Fill(eta, track.hitPattern().pixelLayersWithMeasurement());
    int LayersAll = track.hitPattern().stripLayersWithMeasurement();
    int Layers2D = track.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
    int Layers1D = LayersAll - Layers2D;
    nSTRIPlayersWithMeas_vs_eta[count]->Fill(eta, LayersAll);
    nSTRIPlayersWith1dMeas_vs_eta[count]->Fill(eta, Layers1D);
    nSTRIPlayersWith2dMeas_vs_eta[count]->Fill(eta, Layers2D);

    nlosthits_vs_eta[count]->Fill(eta, track.numberOfLostHits());
}


void MTVHistoProducerAlgoForTracker::fill_trackBased_histos(int count, int assTracks, int numRecoTracks, int numRecoTracksSelected, int numSimTracksSelected) {

   	h_tracks[count]->Fill(assTracks);
   	h_fakes[count]->Fill(numRecoTracks-assTracks);
        if(nrec_vs_nsim[count]) nrec_vs_nsim[count]->Fill(numSimTracksSelected, numRecoTracksSelected);

}



void MTVHistoProducerAlgoForTracker::fill_ResoAndPull_recoTrack_histos(int count,
								       const TrackingParticle::Vector& momentumTP,
								       const TrackingParticle::Point& vertexTP,
								       int chargeTP,
								       const reco::Track& track,
								       const math::XYZPoint& bsPosition){
  if(trackFromSeedFitFailed(track))
    return;

  // evaluation of TP parameters
  double qoverpSim = chargeTP/sqrt(momentumTP.x()*momentumTP.x()+momentumTP.y()*momentumTP.y()+momentumTP.z()*momentumTP.z());
  double lambdaSim = M_PI/2-momentumTP.theta();
  double phiSim    = momentumTP.phi();
  double dxySim    = TrackingParticleIP::dxy(vertexTP, momentumTP, bsPosition);
  double dzSim     = TrackingParticleIP::dz(vertexTP, momentumTP, bsPosition);


  //  reco::Track::ParameterVector rParameters = track.parameters(); // UNUSED

  double qoverpRec(0);
  double qoverpErrorRec(0);
  double ptRec(0);
  double ptErrorRec(0);
  double lambdaRec(0);
  double lambdaErrorRec(0);
  double phiRec(0);
  double phiErrorRec(0);

  /* TO BE FIXED LATER  -----------
  //loop to decide whether to take gsfTrack (utilisation of mode-function) or common track
  const GsfTrack* gsfTrack(0);
  if(useGsf){
    gsfTrack = dynamic_cast<const GsfTrack*>(&(*track));
    if (gsfTrack==0) edm::LogInfo("TrackValidator") << "Trying to access mode for a non-GsfTrack";
  }

  if (gsfTrack) {
    // get values from mode
    getRecoMomentum(*gsfTrack, ptRec, ptErrorRec, qoverpRec, qoverpErrorRec,
		    lambdaRec,lambdaErrorRec, phiRec, phiErrorRec);
  }

  else {
    // get values from track (without mode)
    getRecoMomentum(*track, ptRec, ptErrorRec, qoverpRec, qoverpErrorRec,
		    lambdaRec,lambdaErrorRec, phiRec, phiErrorRec);
  }
  */
  getRecoMomentum(track, ptRec, ptErrorRec, qoverpRec, qoverpErrorRec,
		  lambdaRec,lambdaErrorRec, phiRec, phiErrorRec);
  // -------------

  double ptError = ptErrorRec;
  double ptres=ptRec-sqrt(momentumTP.perp2());
  double etares=track.eta()-momentumTP.Eta();


  double dxyRec    = track.dxy(bsPosition);
  double dzRec     = track.dz(bsPosition);

  const auto phiRes = phiRec-phiSim;
  const auto dxyRes = dxyRec-dxySim;
  const auto dzRes = dzRec-dzSim;
  const auto cotThetaRes = 1/tan(M_PI*0.5-lambdaRec)-1/tan(M_PI*0.5-lambdaSim);

  // eta residue; pt, k, theta, phi, dxy, dz pulls
  double qoverpPull=(qoverpRec-qoverpSim)/qoverpErrorRec;
  double thetaPull=(lambdaRec-lambdaSim)/lambdaErrorRec;
  double phiPull=phiRes/phiErrorRec;
  double dxyPull=dxyRes/track.dxyError();
  double dzPull=dzRes/track.dzError();

#ifdef EDM_ML_DEBUG
  double contrib_Qoverp = ((qoverpRec-qoverpSim)/qoverpErrorRec)*
    ((qoverpRec-qoverpSim)/qoverpErrorRec)/5;
  double contrib_dxy = ((dxyRec-dxySim)/track.dxyError())*((dxyRec-dxySim)/track.dxyError())/5;
  double contrib_dz = ((dzRec-dzSim)/track.dzError())*((dzRec-dzSim)/track.dzError())/5;
  double contrib_theta = ((lambdaRec-lambdaSim)/lambdaErrorRec)*
    ((lambdaRec-lambdaSim)/lambdaErrorRec)/5;
  double contrib_phi = ((phiRec-phiSim)/phiErrorRec)*
    ((phiRec-phiSim)/phiErrorRec)/5;

  LogTrace("TrackValidatorTEST")
    //<< "assocChi2=" << tp.begin()->second << "\n"
    << "" <<  "\n"
    << "ptREC=" << ptRec << "\n" << "etaREC=" << track.eta() << "\n" << "qoverpREC=" << qoverpRec << "\n"
    << "dxyREC=" << dxyRec << "\n" << "dzREC=" << dzRec << "\n"
    << "thetaREC=" << track.theta() << "\n" << "phiREC=" << phiRec << "\n"
    << "" <<  "\n"
    << "qoverpError()=" << qoverpErrorRec << "\n" << "dxyError()=" << track.dxyError() << "\n"<< "dzError()="
    << track.dzError() << "\n"
    << "thetaError()=" << lambdaErrorRec << "\n" << "phiError()=" << phiErrorRec << "\n"
    << "" <<  "\n"
    << "ptSIM=" << sqrt(momentumTP.perp2()) << "\n"<< "etaSIM=" << momentumTP.Eta() << "\n"<< "qoverpSIM=" << qoverpSim << "\n"
    << "dxySIM=" << dxySim << "\n"<< "dzSIM=" << dzSim << "\n" << "thetaSIM=" << M_PI/2-lambdaSim << "\n"
    << "phiSIM=" << phiSim << "\n"
    << "" << "\n"
    << "contrib_Qoverp=" << contrib_Qoverp << "\n"<< "contrib_dxy=" << contrib_dxy << "\n"<< "contrib_dz=" << contrib_dz << "\n"
    << "contrib_theta=" << contrib_theta << "\n"<< "contrib_phi=" << contrib_phi << "\n"
    << "" << "\n"
    <<"chi2PULL="<<contrib_Qoverp+contrib_dxy+contrib_dz+contrib_theta+contrib_phi<<"\n";
#endif

  h_pullQoverp[count]->Fill(qoverpPull);
  h_pullTheta[count]->Fill(thetaPull);
  h_pullPhi[count]->Fill(phiPull);
  h_pullDxy[count]->Fill(dxyPull);
  h_pullDz[count]->Fill(dzPull);

  const auto etaSim = getEta(momentumTP.eta());
  const auto ptSim = getPt(sqrt(momentumTP.perp2()));

  h_pt[count]->Fill(ptres/ptError);
  h_eta[count]->Fill(etares);
  //etares_vs_eta[count]->Fill(getEta(track.eta()),etares);
  etares_vs_eta[count]->Fill(etaSim, etares);

  //resolution of track params: fill 2D histos
  dxyres_vs_eta[count]->Fill(etaSim, dxyRes);
  ptres_vs_eta[count]->Fill(etaSim, ptres/ptRec);
  dzres_vs_eta[count]->Fill(etaSim, dzRes);
  phires_vs_eta[count]->Fill(etaSim, phiRes);
  cotThetares_vs_eta[count]->Fill(etaSim, cotThetaRes);

  //same as before but vs pT
  dxyres_vs_pt[count]->Fill(ptSim, dxyRes);
  ptres_vs_pt[count]->Fill(ptSim, ptres/ptRec);
  dzres_vs_pt[count]->Fill(ptSim, dzRes);
  phires_vs_pt[count]->Fill(ptSim, phiRes);
  cotThetares_vs_pt[count]->Fill(ptSim, cotThetaRes);

  //pulls of track params vs eta: fill 2D histos
  dxypull_vs_eta[count]->Fill(etaSim, dxyPull);
  ptpull_vs_eta[count]->Fill(etaSim, ptres/ptError);
  dzpull_vs_eta[count]->Fill(etaSim, dzPull);
  phipull_vs_eta[count]->Fill(etaSim, phiPull);
  thetapull_vs_eta[count]->Fill(etaSim, thetaPull);

  //plots vs phi
  nhits_vs_phi[count]->Fill(phiRec,track.numberOfValidHits());
  chi2_vs_phi[count]->Fill(phiRec,track.normalizedChi2());
  ptmean_vs_eta_phi[count]->Fill(phiRec,getEta(track.eta()),ptRec);
  phimean_vs_eta_phi[count]->Fill(phiRec,getEta(track.eta()),phiRec);

  ptres_vs_phi[count]->Fill(phiSim, ptres/ptRec);
  phires_vs_phi[count]->Fill(phiSim, phiRes);
  ptpull_vs_phi[count]->Fill(phiSim, ptres/ptError);
  phipull_vs_phi[count]->Fill(phiSim, phiPull);
  thetapull_vs_phi[count]->Fill(phiSim, thetaPull);


}



void
MTVHistoProducerAlgoForTracker::getRecoMomentum (const reco::Track& track, double& pt, double& ptError,
						 double& qoverp, double& qoverpError, double& lambda,double& lambdaError,
						 double& phi, double& phiError ) const {
  pt = track.pt();
  ptError = track.ptError();
  qoverp = track.qoverp();
  qoverpError = track.qoverpError();
  lambda = track.lambda();
  lambdaError = track.lambdaError();
  phi = track.phi();
  phiError = track.phiError();
  //   cout <<"test1" << endl;



}

void
MTVHistoProducerAlgoForTracker::getRecoMomentum (const reco::GsfTrack& gsfTrack, double& pt, double& ptError,
						 double& qoverp, double& qoverpError, double& lambda,double& lambdaError,
						 double& phi, double& phiError  ) const {

  pt = gsfTrack.ptMode();
  ptError = gsfTrack.ptModeError();
  qoverp = gsfTrack.qoverpMode();
  qoverpError = gsfTrack.qoverpModeError();
  lambda = gsfTrack.lambdaMode();
  lambdaError = gsfTrack.lambdaModeError();
  phi = gsfTrack.phiMode();
  phiError = gsfTrack.phiModeError();
  //   cout <<"test2" << endl;

}

double
MTVHistoProducerAlgoForTracker::getEta(double eta) {
  if (useFabsEta) return fabs(eta);
  else return eta;
}

double
MTVHistoProducerAlgoForTracker::getPt(double pt) {
  if (useInvPt && pt!=0) return 1/pt;
  else return pt;
}

unsigned int MTVHistoProducerAlgoForTracker::getSeedingLayerSetBin(const reco::Track& track, const TrackerTopology& ttopo) {
  if(track.seedRef().isNull() || !track.seedRef().isAvailable())
    return seedingLayerSetNames.size()-1;

  const TrajectorySeed& seed = *(track.seedRef());
  const auto hitRange = seed.recHits();
  SeedingLayerSetId searchId;
  const int nhits = std::distance(hitRange.first, hitRange.second);
  if(nhits > static_cast<int>(std::tuple_size<SeedingLayerSetId>::value)) {
    LogDebug("TrackValidator") << "Got seed with " << nhits << " hits, but I have a hard-coded maximum of " << std::tuple_size<SeedingLayerSetId>::value << ", classifying the seed as 'unknown'. Please increase the maximum in MTVHistoProducerAlgoForTracker.h if needed.";
    return seedingLayerSetNames.size()-1;
  }
  int i=0;
  for(auto iHit = hitRange.first; iHit != hitRange.second; ++iHit, ++i) {
    DetId detId = iHit->geographicalId();

    if(detId.det() != DetId::Tracker) {
      throw cms::Exception("LogicError") << "Encountered seed hit detId " << detId.rawId() << " not from Tracker, but " << detId.det();
    }

    GeomDetEnumerators::SubDetector subdet;
    bool subdetStrip = false;
    switch(detId.subdetId()) {
    case PixelSubdetector::PixelBarrel: subdet = GeomDetEnumerators::PixelBarrel; break;
    case PixelSubdetector::PixelEndcap: subdet = GeomDetEnumerators::PixelEndcap; break;
    case StripSubdetector::TIB: subdet = GeomDetEnumerators::TIB; subdetStrip = true; break;
    case StripSubdetector::TID: subdet = GeomDetEnumerators::TID; subdetStrip = true; break;
    case StripSubdetector::TOB: subdet = GeomDetEnumerators::TOB; subdetStrip = true; break;
    case StripSubdetector::TEC: subdet = GeomDetEnumerators::TEC; subdetStrip = true; break;
    default: throw cms::Exception("LogicError") << "Unknown subdetId " << detId.subdetId();
    };

    ctfseeding::SeedingLayer::Side side;
    switch(ttopo.side(detId)) {
    case 0: side = ctfseeding::SeedingLayer::Barrel; break;
    case 1: side = ctfseeding::SeedingLayer::NegEndcap; break;
    case 2: side = ctfseeding::SeedingLayer::PosEndcap; break;
    default: throw cms::Exception("LogicError") << "Unknown side " << ttopo.side(detId);
    };

    // This is an ugly assumption, but a generic solution would
    // require significantly more effort
    // The "if hit is strip mono or not" is checked only for the last
    // hit and only if nhits is 3, because the "mono-only" definition
    // is only used by strip triplet seeds
    bool isStripMono = false;
    if(nhits == 3 && i == nhits-1 && subdetStrip) {
      isStripMono = trackerHitRTTI::isSingle(*iHit);
    }
    searchId[i] = SeedingLayerId(SeedingLayerSetsBuilder::SeedingLayerId(subdet, side, ttopo.layer(detId)), isStripMono);
  }
  auto found = seedingLayerSetToBin.find(searchId);
  if(found == seedingLayerSetToBin.end()) {
    return seedingLayerSetNames.size()-1;
  }
  return found->second;
}

void MTVHistoProducerAlgoForTracker::fill_recoAssociated_simTrack_histos(int count,
									 const reco::GenParticle& tp,
									 const TrackingParticle::Vector& momentumTP,
									 const TrackingParticle::Point& vertexTP,
									 double dxySim, double dzSim, int nSimHits,
									 const reco::Track* track,
									 int numVertices){

  bool isMatched = track;

  if((*GpSelectorForEfficiencyVsEta)(tp)){
    //effic vs eta
    fillPlotNoFlow(h_simuleta[count],getEta(momentumTP.eta()));
    if (isMatched) fillPlotNoFlow(h_assoceta[count],getEta(momentumTP.eta()));
  }

  if((*GpSelectorForEfficiencyVsPhi)(tp)){
    fillPlotNoFlow(h_simulphi[count],momentumTP.phi());
    if (isMatched) fillPlotNoFlow(h_assocphi[count],momentumTP.phi());
    //effic vs hits
    fillPlotNoFlow(h_simulhit[count],(int)nSimHits);
    if(isMatched) {
      fillPlotNoFlow(h_assochit[count],(int)nSimHits);
      if(nrecHit_vs_nsimHit_sim2rec[count]) nrecHit_vs_nsimHit_sim2rec[count]->Fill(track->numberOfValidHits(),nSimHits);
    }
    //effic vs pu
    fillPlotNoFlow(h_simulpu[count],numVertices);
    if (isMatched) fillPlotNoFlow(h_assocpu[count],numVertices);
    //efficiency vs dR
    //not implemented for now
  }

  if((*GpSelectorForEfficiencyVsPt)(tp)){
    fillPlotNoFlow(h_simulpT[count],getPt(sqrt(momentumTP.perp2())));
    if (isMatched) fillPlotNoFlow(h_assocpT[count],getPt(sqrt(momentumTP.perp2())));
  }

  if((*GpSelectorForEfficiencyVsVTXR)(tp)){
    fillPlotNoFlow(h_simuldxy[count],dxySim);
    if (isMatched) fillPlotNoFlow(h_assocdxy[count],dxySim);

    fillPlotNoFlow(h_simulvertpos[count],sqrt(vertexTP.perp2()));
    if (isMatched) fillPlotNoFlow(h_assocvertpos[count],sqrt(vertexTP.perp2()));
  }

  if((*GpSelectorForEfficiencyVsVTXZ)(tp)){
    fillPlotNoFlow(h_simuldz[count],dzSim);
    if (isMatched) fillPlotNoFlow(h_assocdz[count],dzSim);

    fillPlotNoFlow(h_simulzpos[count],vertexTP.z());
    if (isMatched) fillPlotNoFlow(h_assoczpos[count],vertexTP.z());
  }

}

void MTVHistoProducerAlgoForTracker::fill_seed_histos(int count,
                                                      int seedsFitFailed,
                                                      int seedsTotal) {
  fillPlotNoFlow(h_seedsFitFailed[count], seedsFitFailed);
  h_seedsFitFailedFraction[count]->Fill(static_cast<double>(seedsFitFailed)/seedsTotal);
}
