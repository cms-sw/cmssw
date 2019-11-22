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
  typedef dqm::reco::DQMStore DQMStore;

  void BinLogX(TH1* h) {
    TAxis* axis = h->GetXaxis();
    int bins = axis->GetNbins();

    float from = axis->GetXmin();
    float to = axis->GetXmax();
    float width = (to - from) / bins;
    std::vector<float> new_bins(bins + 1, 0);

    for (int i = 0; i <= bins; i++) {
      new_bins[i] = TMath::Power(10, from + i * width);
    }
    axis->Set(bins, new_bins.data());
  }

  void BinLogY(TH1* h) {
    TAxis* axis = h->GetYaxis();
    int bins = axis->GetNbins();

    float from = axis->GetXmin();
    float to = axis->GetXmax();
    float width = (to - from) / bins;
    std::vector<float> new_bins(bins + 1, 0);

    for (int i = 0; i <= bins; i++) {
      new_bins[i] = TMath::Power(10, from + i * width);
    }
    axis->Set(bins, new_bins.data());
  }

  template <typename... Args>
  dqm::reco::MonitorElement* make1DIfLogX(DQMStore::IBooker& ibook, bool logx, Args&&... args) {
    auto h = std::make_unique<TH1F>(std::forward<Args>(args)...);
    if (logx)
      BinLogX(h.get());
    const auto& name = h->GetName();
    return ibook.book1D(name, h.release());
  }

  template <typename... Args>
  dqm::reco::MonitorElement* makeProfileIfLogX(DQMStore::IBooker& ibook, bool logx, Args&&... args) {
    auto h = std::make_unique<TProfile>(std::forward<Args>(args)...);
    if (logx)
      BinLogX(h.get());
    const auto& name = h->GetName();
    return ibook.bookProfile(name, h.release());
  }

  template <typename... Args>
  dqm::reco::MonitorElement* make2DIfLogX(DQMStore::IBooker& ibook, bool logx, Args&&... args) {
    auto h = std::make_unique<TH2F>(std::forward<Args>(args)...);
    if (logx)
      BinLogX(h.get());
    const auto& name = h->GetName();
    return ibook.book2D(name, h.release());
  }

  template <typename... Args>
  dqm::reco::MonitorElement* make2DIfLogY(DQMStore::IBooker& ibook, bool logy, Args&&... args) {
    auto h = std::make_unique<TH2F>(std::forward<Args>(args)...);
    if (logy)
      BinLogY(h.get());
    const auto& name = h->GetName();
    return ibook.book2D(name, h.release());
  }

  void setBinLabels(dqm::reco::MonitorElement*& h, const std::vector<std::string>& labels) {
    for (size_t i = 0; i < labels.size(); ++i) {
      h->setBinLabel(i + 1, labels[i]);
    }
    h->disableAlphanumeric();
  }

  void setBinLabelsAlgo(dqm::reco::MonitorElement*& h, int axis = 1) {
    for (size_t i = 0; i < reco::TrackBase::algoSize; ++i) {
      h->setBinLabel(i + 1, reco::TrackBase::algoName(static_cast<reco::TrackBase::TrackAlgorithm>(i)), axis);
    }
    h->disableAlphanumeric();
  }

  void fillMVAHistos(const std::vector<dqm::reco::MonitorElement*>& h_mva,
                     const std::vector<dqm::reco::MonitorElement*>& h_mvacut,
                     const std::vector<dqm::reco::MonitorElement*>& h_mva_hp,
                     const std::vector<dqm::reco::MonitorElement*>& h_mvacut_hp,
                     const std::vector<float>& mvas,
                     unsigned int selectsLoose,
                     unsigned int selectsHP) {
    // Fill MVA1 histos with all tracks, MVA2 histos only with tracks
    // not selected by MVA1, etc.
    for (size_t i = 0; i < mvas.size(); ++i) {
      if (i <= selectsLoose) {
        h_mva[i]->Fill(mvas[i]);
        h_mvacut[i]->Fill(mvas[i]);
      }
      if (i >= 1 && i <= selectsHP) {
        h_mva_hp[i]->Fill(mvas[i]);
        h_mvacut_hp[i]->Fill(mvas[i]);
      }
    }
  }

  void fillMVAHistos(double xval,
                     const std::vector<dqm::reco::MonitorElement*>& h_mva,
                     const std::vector<dqm::reco::MonitorElement*>& h_mva_hp,
                     const std::vector<float>& mvas,
                     unsigned int selectsLoose,
                     unsigned int selectsHP) {
    // Fill MVA1 histos with all tracks, MVA2 histos only with tracks
    // not selected by MVA1, etc.
    for (size_t i = 0; i < mvas.size(); ++i) {
      if (i <= selectsLoose) {
        h_mva[i]->Fill(xval, mvas[i]);
      }
      if (i >= 1 && i <= selectsHP) {
        h_mva_hp[i]->Fill(xval, mvas[i]);
      }
    }
  }
}  // namespace

MTVHistoProducerAlgoForTracker::MTVHistoProducerAlgoForTracker(const edm::ParameterSet& pset, const bool doSeedPlots)
    : doSeedPlots_(doSeedPlots), doMTDPlots_(pset.getUntrackedParameter<bool>("doMTDPlots")) {
  //parameters for _vs_eta plots
  minEta = pset.getParameter<double>("minEta");
  maxEta = pset.getParameter<double>("maxEta");
  nintEta = pset.getParameter<int>("nintEta");
  useFabsEta = pset.getParameter<bool>("useFabsEta");

  //parameters for _vs_pt plots
  minPt = pset.getParameter<double>("minPt");
  maxPt = pset.getParameter<double>("maxPt");
  nintPt = pset.getParameter<int>("nintPt");
  useInvPt = pset.getParameter<bool>("useInvPt");
  useLogPt = pset.getUntrackedParameter<bool>("useLogPt", false);

  //parameters for _vs_Hit plots
  minHit = pset.getParameter<double>("minHit");
  maxHit = pset.getParameter<double>("maxHit");
  nintHit = pset.getParameter<int>("nintHit");

  //parameters for _vs_Pu plots
  minPu = pset.getParameter<double>("minPu");
  maxPu = pset.getParameter<double>("maxPu");
  nintPu = pset.getParameter<int>("nintPu");

  //parameters for _vs_Layer plots
  minLayers = pset.getParameter<double>("minLayers");
  maxLayers = pset.getParameter<double>("maxLayers");
  nintLayers = pset.getParameter<int>("nintLayers");

  //parameters for _vs_phi plots
  minPhi = pset.getParameter<double>("minPhi");
  maxPhi = pset.getParameter<double>("maxPhi");
  nintPhi = pset.getParameter<int>("nintPhi");

  //parameters for _vs_Dxy plots
  minDxy = pset.getParameter<double>("minDxy");
  maxDxy = pset.getParameter<double>("maxDxy");
  nintDxy = pset.getParameter<int>("nintDxy");

  //parameters for _vs_Dz plots
  minDz = pset.getParameter<double>("minDz");
  maxDz = pset.getParameter<double>("maxDz");
  nintDz = pset.getParameter<int>("nintDz");

  dxyDzZoom = pset.getParameter<double>("dxyDzZoom");

  //parameters for _vs_ProductionVertexTransvPosition plots
  minVertpos = pset.getParameter<double>("minVertpos");
  maxVertpos = pset.getParameter<double>("maxVertpos");
  nintVertpos = pset.getParameter<int>("nintVertpos");
  useLogVertpos = pset.getUntrackedParameter<bool>("useLogVertpos");

  //parameters for _vs_ProductionVertexZPosition plots
  minZpos = pset.getParameter<double>("minZpos");
  maxZpos = pset.getParameter<double>("maxZpos");
  nintZpos = pset.getParameter<int>("nintZpos");

  //parameters for _vs_dR plots
  mindr = pset.getParameter<double>("mindr");
  maxdr = pset.getParameter<double>("maxdr");
  nintdr = pset.getParameter<int>("nintdr");

  //parameters for _vs_dR_jet plots
  mindrj = pset.getParameter<double>("mindrj");
  maxdrj = pset.getParameter<double>("maxdrj");
  nintdrj = pset.getParameter<int>("nintdrj");

  // paramers for _vs_chi2 plots
  minChi2 = pset.getParameter<double>("minChi2");
  maxChi2 = pset.getParameter<double>("maxChi2");
  nintChi2 = pset.getParameter<int>("nintChi2");

  //parameters for dE/dx plots
  minDeDx = pset.getParameter<double>("minDeDx");
  maxDeDx = pset.getParameter<double>("maxDeDx");
  nintDeDx = pset.getParameter<int>("nintDeDx");

  //parameters for Pileup plots
  minVertcount = pset.getParameter<double>("minVertcount");
  maxVertcount = pset.getParameter<double>("maxVertcount");
  nintVertcount = pset.getParameter<int>("nintVertcount");

  //parameters for number of tracks plots
  minTracks = pset.getParameter<double>("minTracks");
  maxTracks = pset.getParameter<double>("maxTracks");
  nintTracks = pset.getParameter<int>("nintTracks");

  //parameters for vs. PV z plots
  minPVz = pset.getParameter<double>("minPVz");
  maxPVz = pset.getParameter<double>("maxPVz");
  nintPVz = pset.getParameter<int>("nintPVz");

  //parameters for vs. MVA plots
  minMVA = pset.getParameter<double>("minMVA");
  maxMVA = pset.getParameter<double>("maxMVA");
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
    sel = std::make_unique<TrackingParticleSelector>(
        ParameterAdapter<TrackingParticleSelector>::make(pset.getParameter<ParameterSet>(name)));
  };
  auto initTrackSelector = [&](auto& sel, auto& name) {
    sel = makeRecoTrackSelectorFromTPSelectorParameters(pset.getParameter<ParameterSet>(name));
  };
  auto initGPselector = [&](auto& sel, auto& name) {
    sel = std::make_unique<GenParticleCustomSelector>(
        ParameterAdapter<GenParticleCustomSelector>::make(pset.getParameter<ParameterSet>(name)));
  };

  initTPselector(generalTpSelector, "generalTpSelector");
  initTPselector(TpSelectorForEfficiencyVsEta, "TpSelectorForEfficiencyVsEta");
  initTPselector(TpSelectorForEfficiencyVsPhi, "TpSelectorForEfficiencyVsPhi");
  initTPselector(TpSelectorForEfficiencyVsPt, "TpSelectorForEfficiencyVsPt");
  initTPselector(TpSelectorForEfficiencyVsVTXR, "TpSelectorForEfficiencyVsVTXR");
  initTPselector(TpSelectorForEfficiencyVsVTXZ, "TpSelectorForEfficiencyVsVTXZ");

  initTrackSelector(trackSelectorVsEta, "TpSelectorForEfficiencyVsEta");
  initTrackSelector(trackSelectorVsPhi, "TpSelectorForEfficiencyVsPhi");
  initTrackSelector(trackSelectorVsPt, "TpSelectorForEfficiencyVsPt");

  initGPselector(generalGpSelector, "generalGpSelector");
  initGPselector(GpSelectorForEfficiencyVsEta, "GpSelectorForEfficiencyVsEta");
  initGPselector(GpSelectorForEfficiencyVsPhi, "GpSelectorForEfficiencyVsPhi");
  initGPselector(GpSelectorForEfficiencyVsPt, "GpSelectorForEfficiencyVsPt");
  initGPselector(GpSelectorForEfficiencyVsVTXR, "GpSelectorForEfficiencyVsVTXR");
  initGPselector(GpSelectorForEfficiencyVsVTXZ, "GpSelectorForEfficiencyVsVTXZ");

  // SeedingLayerSets
  // If enabled, use last bin to denote other or unknown cases
  seedingLayerSetNames = pset.getParameter<std::vector<std::string>>("seedingLayerSets");
  std::vector<std::pair<SeedingLayerSetId, std::string>> stripPairSets;
  if (!seedingLayerSetNames.empty()) {
    std::vector<std::vector<std::string>> layerSets = SeedingLayerSetsBuilder::layerNamesInSets(seedingLayerSetNames);
    for (size_t i = 0; i < layerSets.size(); ++i) {
      const auto& layerSet = layerSets[i];
      if (layerSet.size() > std::tuple_size<SeedingLayerSetId>::value) {
        throw cms::Exception("Configuration")
            << "Got seedingLayerSet " << seedingLayerSetNames[i] << " with " << layerSet.size()
            << " elements, but I have a hard-coded maximum of " << std::tuple_size<SeedingLayerSetId>::value
            << ". Please increase the maximum in MTVHistoProducerAlgoForTracker.h";
      }
      SeedingLayerSetId setId;
      for (size_t j = 0; j < layerSet.size(); ++j) {
        // SeedingLayerSetsBuilder::fillDescriptions() kind-of
        // suggests that the 'M' prefix stands for strip mono hits
        // (maybe it should force), so making the assumption here is
        // (still) a bit ugly. But, this is the easiest way.
        bool isStripMono = !layerSet[j].empty() && layerSet[j][0] == 'M';
        setId[j] = std::make_tuple(SeedingLayerSetsBuilder::nameToEnumId(layerSet[j]), isStripMono);
      }
      // Account for the fact that strip triplet seeding may give pairs
      if (layerSet.size() == 3 && isTrackerStrip(std::get<GeomDetEnumerators::SubDetector>(std::get<0>(setId[0])))) {
        SeedingLayerSetId pairId;
        pairId[0] = setId[0];
        pairId[1] = setId[1];
        stripPairSets.emplace_back(pairId, layerSet[0] + "+" + layerSet[1]);
      }

      auto inserted = seedingLayerSetToBin.insert(std::make_pair(setId, i));
      if (!inserted.second)
        throw cms::Exception("Configuration") << "SeedingLayerSet " << seedingLayerSetNames[i]
                                              << " is specified twice, while the set list should be unique.";
    }

    // Add the "strip pairs from strip triplets" if they don't otherwise exist
    for (const auto& setIdName : stripPairSets) {
      auto inserted = seedingLayerSetToBin.insert(std::make_pair(setIdName.first, seedingLayerSetNames.size()));
      if (inserted.second)
        seedingLayerSetNames.push_back(setIdName.second);
    }

    seedingLayerSetNames.emplace_back("Other/Unknown");
  }

  // fix for the LogScale by Ryan
  if (useLogPt) {
    maxPt = log10(maxPt);
    if (minPt > 0) {
      minPt = log10(minPt);
    } else {
      edm::LogWarning("MultiTrackValidator")
          << "minPt = " << minPt << " <= 0 out of range while requesting log scale.  Using minPt = 0.1.";
      minPt = log10(0.1);
    }
  }
  if (useLogVertpos) {
    maxVertpos = std::log10(maxVertpos);
    if (minVertpos > 0) {
      minVertpos = std::log10(minVertpos);
    } else {
      edm::LogWarning("MultiTrackValidator")
          << "minVertpos = " << minVertpos << " <= 0 out of range while requesting log scale.  Using minVertpos = 0.1.";
      minVertpos = -1;
    }
  }
}

MTVHistoProducerAlgoForTracker::~MTVHistoProducerAlgoForTracker() {}

std::unique_ptr<RecoTrackSelectorBase> MTVHistoProducerAlgoForTracker::makeRecoTrackSelectorFromTPSelectorParameters(
    const edm::ParameterSet& pset) {
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
  psetTrack.addParameter("quality", std::vector<std::string>{});
  psetTrack.addParameter("algorithm", std::vector<std::string>{});
  psetTrack.addParameter("originalAlgorithm", std::vector<std::string>{});
  psetTrack.addParameter("algorithmMaskContains", std::vector<std::string>{});
  psetTrack.addParameter("invertRapidityCut", false);
  psetTrack.addParameter("minPhi", -3.2);
  psetTrack.addParameter("maxPhi", 3.2);
  return std::make_unique<RecoTrackSelectorBase>(psetTrack);
}

void MTVHistoProducerAlgoForTracker::bookSimHistos(DQMStore::IBooker& ibook, Histograms& histograms) {
  histograms.h_ptSIM = make1DIfLogX(ibook, useLogPt, "ptSIM", "generated p_{t}", nintPt, minPt, maxPt);
  histograms.h_etaSIM = ibook.book1D("etaSIM", "generated pseudorapidity", nintEta, minEta, maxEta);
  histograms.h_tracksSIM =
      ibook.book1D("tracksSIM", "number of simulated tracks", nintTracks, minTracks, maxTracks * 10);
  histograms.h_vertposSIM =
      ibook.book1D("vertposSIM", "Transverse position of sim vertices", nintVertpos, minVertpos, maxVertpos);
  histograms.h_bunchxSIM = ibook.book1D("bunchxSIM", "bunch crossing", 21, -15.5, 5.5);
}

void MTVHistoProducerAlgoForTracker::bookSimTrackHistos(DQMStore::IBooker& ibook,
                                                        Histograms& histograms,
                                                        bool doResolutionPlots) {
  histograms.h_assoceta.push_back(
      ibook.book1D("num_assoc(simToReco)_eta", "N of associated tracks (simToReco) vs eta", nintEta, minEta, maxEta));
  histograms.h_simuleta.push_back(
      ibook.book1D("num_simul_eta", "N of simulated tracks vs eta", nintEta, minEta, maxEta));

  histograms.h_assocpT.push_back(make1DIfLogX(
      ibook, useLogPt, "num_assoc(simToReco)_pT", "N of associated tracks (simToReco) vs pT", nintPt, minPt, maxPt));
  histograms.h_simulpT.push_back(
      make1DIfLogX(ibook, useLogPt, "num_simul_pT", "N of simulated tracks vs pT", nintPt, minPt, maxPt));

  histograms.h_assocpTvseta.push_back(make2DIfLogY(ibook,
                                                   useLogPt,
                                                   "num_assoc(simToReco)_pTvseta",
                                                   "N of associated tracks (simToReco) in (pT-eta) please",
                                                   nintEta,
                                                   minEta,
                                                   maxEta,
                                                   nintPt,
                                                   minPt,
                                                   maxPt));
  histograms.h_simulpTvseta.push_back(make2DIfLogY(ibook,
                                                   useLogPt,
                                                   "num_simul_pTvseta",
                                                   "N of simulated tracks in (pT-eta) plane",
                                                   nintEta,
                                                   minEta,
                                                   maxEta,
                                                   nintPt,
                                                   minPt,
                                                   maxPt));

  histograms.h_assochit.push_back(
      ibook.book1D("num_assoc(simToReco)_hit", "N of associated tracks (simToReco) vs hit", nintHit, minHit, maxHit));
  histograms.h_simulhit.push_back(
      ibook.book1D("num_simul_hit", "N of simulated tracks vs hit", nintHit, minHit, maxHit));

  histograms.h_assoclayer.push_back(ibook.book1D(
      "num_assoc(simToReco)_layer", "N of associated tracks (simToReco) vs layer", nintLayers, minLayers, maxLayers));
  histograms.h_simullayer.push_back(
      ibook.book1D("num_simul_layer", "N of simulated tracks vs layer", nintLayers, minLayers, maxLayers));

  histograms.h_assocpixellayer.push_back(ibook.book1D("num_assoc(simToReco)_pixellayer",
                                                      "N of associated tracks (simToReco) vs pixel layer",
                                                      nintLayers,
                                                      minLayers,
                                                      maxLayers));
  histograms.h_simulpixellayer.push_back(
      ibook.book1D("num_simul_pixellayer", "N of simulated tracks vs pixel layer", nintLayers, minLayers, maxLayers));

  histograms.h_assoc3Dlayer.push_back(ibook.book1D("num_assoc(simToReco)_3Dlayer",
                                                   "N of associated tracks (simToReco) vs 3D layer",
                                                   nintLayers,
                                                   minLayers,
                                                   maxLayers));
  histograms.h_simul3Dlayer.push_back(
      ibook.book1D("num_simul_3Dlayer", "N of simulated tracks vs 3D layer", nintLayers, minLayers, maxLayers));

  histograms.h_assocpu.push_back(
      ibook.book1D("num_assoc(simToReco)_pu", "N of associated tracks (simToReco) vs pu", nintPu, minPu, maxPu));
  histograms.h_simulpu.push_back(ibook.book1D("num_simul_pu", "N of simulated tracks vs pu", nintPu, minPu, maxPu));

  histograms.h_assocphi.push_back(
      ibook.book1D("num_assoc(simToReco)_phi", "N of associated tracks (simToReco) vs phi", nintPhi, minPhi, maxPhi));
  histograms.h_simulphi.push_back(
      ibook.book1D("num_simul_phi", "N of simulated tracks vs phi", nintPhi, minPhi, maxPhi));

  histograms.h_assocdxy.push_back(
      ibook.book1D("num_assoc(simToReco)_dxy", "N of associated tracks (simToReco) vs dxy", nintDxy, minDxy, maxDxy));
  histograms.h_simuldxy.push_back(
      ibook.book1D("num_simul_dxy", "N of simulated tracks vs dxy", nintDxy, minDxy, maxDxy));

  histograms.h_assocdz.push_back(
      ibook.book1D("num_assoc(simToReco)_dz", "N of associated tracks (simToReco) vs dz", nintDz, minDz, maxDz));
  histograms.h_simuldz.push_back(ibook.book1D("num_simul_dz", "N of simulated tracks vs dz", nintDz, minDz, maxDz));

  histograms.h_assocvertpos.push_back(make1DIfLogX(ibook,
                                                   useLogVertpos,
                                                   "num_assoc(simToReco)_vertpos",
                                                   "N of associated tracks (simToReco) vs transverse vert position",
                                                   nintVertpos,
                                                   minVertpos,
                                                   maxVertpos));
  histograms.h_simulvertpos.push_back(make1DIfLogX(ibook,
                                                   useLogVertpos,
                                                   "num_simul_vertpos",
                                                   "N of simulated tracks vs transverse vert position",
                                                   nintVertpos,
                                                   minVertpos,
                                                   maxVertpos));

  histograms.h_assoczpos.push_back(ibook.book1D(
      "num_assoc(simToReco)_zpos", "N of associated tracks (simToReco) vs z vert position", nintZpos, minZpos, maxZpos));
  histograms.h_simulzpos.push_back(
      ibook.book1D("num_simul_zpos", "N of simulated tracks vs z vert position", nintZpos, minZpos, maxZpos));

  histograms.h_assocdr.push_back(make1DIfLogX(ibook,
                                              true,
                                              "num_assoc(simToReco)_dr",
                                              "N of associated tracks (simToReco) vs dR",
                                              nintdr,
                                              log10(mindr),
                                              log10(maxdr)));
  histograms.h_simuldr.push_back(
      make1DIfLogX(ibook, true, "num_simul_dr", "N of simulated tracks vs dR", nintdr, log10(mindr), log10(maxdr)));

  histograms.h_assocdrj.push_back(make1DIfLogX(ibook,
                                               true,
                                               "num_assoc(simToReco)_drj",
                                               "N of associated tracks (simToReco) vs dR(TP,jet)",
                                               nintdrj,
                                               log10(mindrj),
                                               log10(maxdrj)));
  histograms.h_simuldrj.push_back(make1DIfLogX(
      ibook, true, "num_simul_drj", "N of simulated tracks vs dR(TP,jet)", nintdrj, log10(mindrj), log10(maxdrj)));

  histograms.h_simul_simpvz.push_back(
      ibook.book1D("num_simul_simpvz", "N of simulated tracks vs. sim PV z", nintPVz, minPVz, maxPVz));
  histograms.h_assoc_simpvz.push_back(ibook.book1D(
      "num_assoc(simToReco)_simpvz", "N of associated tracks (simToReco) vs. sim PV z", nintPVz, minPVz, maxPVz));

  histograms.nrecHit_vs_nsimHit_sim2rec.push_back(doResolutionPlots ? ibook.book2D("nrecHit_vs_nsimHit_sim2rec",
                                                                                   "nrecHit vs nsimHit (Sim2RecAssoc)",
                                                                                   nintHit,
                                                                                   minHit,
                                                                                   maxHit,
                                                                                   nintHit,
                                                                                   minHit,
                                                                                   maxHit)
                                                                    : nullptr);

  // TODO: use the dynamic track algo priority order also here
  constexpr auto nalgos = reco::TrackBase::algoSize;
  histograms.h_duplicates_oriAlgo_vs_oriAlgo.push_back(ibook.book2D("duplicates_oriAlgo_vs_oriAlgo",
                                                                    "Duplicate tracks: originalAlgo vs originalAlgo",
                                                                    nalgos,
                                                                    0,
                                                                    nalgos,
                                                                    nalgos,
                                                                    0,
                                                                    nalgos));
  setBinLabelsAlgo(histograms.h_duplicates_oriAlgo_vs_oriAlgo.back(), 1);
  setBinLabelsAlgo(histograms.h_duplicates_oriAlgo_vs_oriAlgo.back(), 2);
}

void MTVHistoProducerAlgoForTracker::bookSimTrackPVAssociationHistos(DQMStore::IBooker& ibook, Histograms& histograms) {
  histograms.h_assocdxypv.push_back(ibook.book1D(
      "num_assoc(simToReco)_dxypv", "N of associated tracks (simToReco) vs dxy(PV)", nintDxy, minDxy, maxDxy));
  histograms.h_simuldxypv.push_back(
      ibook.book1D("num_simul_dxypv", "N of simulated tracks vs dxy(PV)", nintDxy, minDxy, maxDxy));

  histograms.h_assocdzpv.push_back(
      ibook.book1D("num_assoc(simToReco)_dzpv", "N of associated tracks (simToReco) vs dz(PV)", nintDz, minDz, maxDz));
  histograms.h_simuldzpv.push_back(
      ibook.book1D("num_simul_dzpv", "N of simulated tracks vs dz(PV)", nintDz, minDz, maxDz));

  histograms.h_assocdxypvzoomed.push_back(ibook.book1D("num_assoc(simToReco)_dxypv_zoomed",
                                                       "N of associated tracks (simToReco) vs dxy(PV)",
                                                       nintDxy,
                                                       minDxy / dxyDzZoom,
                                                       maxDxy / dxyDzZoom));
  histograms.h_simuldxypvzoomed.push_back(ibook.book1D(
      "num_simul_dxypv_zoomed", "N of simulated tracks vs dxy(PV)", nintDxy, minDxy / dxyDzZoom, maxDxy / dxyDzZoom));

  histograms.h_assocdzpvzoomed.push_back(ibook.book1D("num_assoc(simToReco)_dzpv_zoomed",
                                                      "N of associated tracks (simToReco) vs dz(PV)",
                                                      nintDz,
                                                      minDz / dxyDzZoom,
                                                      maxDz / dxyDzZoom));
  histograms.h_simuldzpvzoomed.push_back(ibook.book1D(
      "num_simul_dzpv_zoomed", "N of simulated tracks vs dz(PV)", nintDz, minDz / dxyDzZoom, maxDz / dxyDzZoom));

  histograms.h_assoc_dzpvcut.push_back(ibook.book1D(
      "num_assoc(simToReco)_dzpvcut", "N of associated tracks (simToReco) vs dz(PV)", nintDzpvCum, 0, maxDzpvCum));
  histograms.h_simul_dzpvcut.push_back(
      ibook.book1D("num_simul_dzpvcut", "N of simulated tracks from sim PV", nintDzpvCum, 0, maxDzpvCum));
  histograms.h_simul2_dzpvcut.push_back(ibook.book1D(
      "num_simul2_dzpvcut", "N of simulated tracks (associated to any track) from sim PV", nintDzpvCum, 0, maxDzpvCum));

  histograms.h_assoc_dzpvcut_pt.push_back(ibook.book1D("num_assoc(simToReco)_dzpvcut_pt",
                                                       "#sump_{T} of associated tracks (simToReco) vs dz(PV)",
                                                       nintDzpvCum,
                                                       0,
                                                       maxDzpvCum));
  histograms.h_simul_dzpvcut_pt.push_back(
      ibook.book1D("num_simul_dzpvcut_pt", "#sump_{T} of simulated tracks from sim PV", nintDzpvCum, 0, maxDzpvCum));
  histograms.h_simul2_dzpvcut_pt.push_back(
      ibook.book1D("num_simul2_dzpvcut_pt",
                   "#sump_{T} of simulated tracks (associated to any track) from sim PV",
                   nintDzpvCum,
                   0,
                   maxDzpvCum));
  histograms.h_assoc_dzpvcut_pt.back()->enableSumw2();
  histograms.h_simul_dzpvcut_pt.back()->enableSumw2();
  histograms.h_simul2_dzpvcut_pt.back()->enableSumw2();

  histograms.h_assoc_dzpvsigcut.push_back(ibook.book1D("num_assoc(simToReco)_dzpvsigcut",
                                                       "N of associated tracks (simToReco) vs dz(PV)/dzError",
                                                       nintDzpvsigCum,
                                                       0,
                                                       maxDzpvsigCum));
  histograms.h_simul_dzpvsigcut.push_back(ibook.book1D(
      "num_simul_dzpvsigcut", "N of simulated tracks from sim PV/dzError", nintDzpvsigCum, 0, maxDzpvsigCum));
  histograms.h_simul2_dzpvsigcut.push_back(
      ibook.book1D("num_simul2_dzpvsigcut",
                   "N of simulated tracks (associated to any track) from sim PV/dzError",
                   nintDzpvsigCum,
                   0,
                   maxDzpvsigCum));

  histograms.h_assoc_dzpvsigcut_pt.push_back(
      ibook.book1D("num_assoc(simToReco)_dzpvsigcut_pt",
                   "#sump_{T} of associated tracks (simToReco) vs dz(PV)/dzError",
                   nintDzpvsigCum,
                   0,
                   maxDzpvsigCum));
  histograms.h_simul_dzpvsigcut_pt.push_back(ibook.book1D(
      "num_simul_dzpvsigcut_pt", "#sump_{T} of simulated tracks from sim PV/dzError", nintDzpvsigCum, 0, maxDzpvsigCum));
  histograms.h_simul2_dzpvsigcut_pt.push_back(
      ibook.book1D("num_simul2_dzpvsigcut_pt",
                   "#sump_{T} of simulated tracks (associated to any track) from sim PV/dzError",
                   nintDzpvsigCum,
                   0,
                   maxDzpvsigCum));
  histograms.h_assoc_dzpvsigcut_pt.back()->enableSumw2();
  histograms.h_simul_dzpvsigcut_pt.back()->enableSumw2();
  histograms.h_simul2_dzpvsigcut_pt.back()->enableSumw2();
}

void MTVHistoProducerAlgoForTracker::bookRecoHistos(DQMStore::IBooker& ibook,
                                                    Histograms& histograms,
                                                    bool doResolutionPlots) {
  histograms.h_tracks.push_back(
      ibook.book1D("tracks", "number of reconstructed tracks", nintTracks, minTracks, maxTracks));
  histograms.h_fakes.push_back(ibook.book1D("fakes", "number of fake reco tracks", nintTracks, minTracks, maxTracks));
  histograms.h_charge.push_back(ibook.book1D("charge", "charge", 3, -1.5, 1.5));

  histograms.h_hits.push_back(ibook.book1D("hits", "number of hits per track", nintHit, minHit, maxHit));
  histograms.h_losthits.push_back(ibook.book1D("losthits", "number of lost hits per track", nintHit, minHit, maxHit));
  histograms.h_nchi2.push_back(ibook.book1D("chi2", "normalized #chi^{2}", 200, 0, 20));
  histograms.h_nchi2_prob.push_back(ibook.book1D("chi2_prob", "normalized #chi^{2} probability", 100, 0, 1));

  histograms.h_nmisslayers_inner.push_back(
      ibook.book1D("missing_inner_layers", "number of missing inner layers", nintLayers, minLayers, maxLayers));
  histograms.h_nmisslayers_outer.push_back(
      ibook.book1D("missing_outer_layers", "number of missing outer layers", nintLayers, minLayers, maxLayers));

  histograms.h_algo.push_back(
      ibook.book1D("h_algo", "Tracks by algo", reco::TrackBase::algoSize, 0., double(reco::TrackBase::algoSize)));
  for (size_t ibin = 0; ibin < reco::TrackBase::algoSize - 1; ibin++)
    histograms.h_algo.back()->setBinLabel(ibin + 1, reco::TrackBase::algoNames[ibin]);
  histograms.h_algo.back()->disableAlphanumeric();

  /// these are needed to calculate efficiency during the harvesting for the automated validation
  histograms.h_recoeta.push_back(ibook.book1D("num_reco_eta", "N of reco track vs eta", nintEta, minEta, maxEta));
  histograms.h_reco2eta.push_back(
      ibook.book1D("num_reco2_eta", "N of selected reco track vs eta", nintEta, minEta, maxEta));
  histograms.h_assoc2eta.push_back(
      ibook.book1D("num_assoc(recoToSim)_eta", "N of associated (recoToSim) tracks vs eta", nintEta, minEta, maxEta));
  histograms.h_loopereta.push_back(ibook.book1D(
      "num_duplicate_eta", "N of associated (recoToSim) duplicate tracks vs eta", nintEta, minEta, maxEta));
  if (!doSeedPlots_)
    histograms.h_misideta.push_back(ibook.book1D(
        "num_chargemisid_eta", "N of associated (recoToSim) charge misIDed tracks vs eta", nintEta, minEta, maxEta));
  histograms.h_pileupeta.push_back(
      ibook.book1D("num_pileup_eta", "N of associated (recoToSim) pileup tracks vs eta", nintEta, minEta, maxEta));
  //
  histograms.h_recopT.push_back(
      make1DIfLogX(ibook, useLogPt, "num_reco_pT", "N of reco track vs pT", nintPt, minPt, maxPt));
  histograms.h_reco2pT.push_back(
      make1DIfLogX(ibook, useLogPt, "num_reco2_pT", "N of selected reco track vs pT", nintPt, minPt, maxPt));
  histograms.h_assoc2pT.push_back(make1DIfLogX(
      ibook, useLogPt, "num_assoc(recoToSim)_pT", "N of associated (recoToSim) tracks vs pT", nintPt, minPt, maxPt));
  histograms.h_looperpT.push_back(make1DIfLogX(
      ibook, useLogPt, "num_duplicate_pT", "N of associated (recoToSim) duplicate tracks vs pT", nintPt, minPt, maxPt));
  if (!doSeedPlots_)
    histograms.h_misidpT.push_back(make1DIfLogX(ibook,
                                                useLogPt,
                                                "num_chargemisid_pT",
                                                "N of associated (recoToSim) charge misIDed tracks vs pT",
                                                nintPt,
                                                minPt,
                                                maxPt));
  histograms.h_pileuppT.push_back(make1DIfLogX(
      ibook, useLogPt, "num_pileup_pT", "N of associated (recoToSim) pileup tracks vs pT", nintPt, minPt, maxPt));
  //
  histograms.h_recopTvseta.push_back(make2DIfLogY(ibook,
                                                  useLogPt,
                                                  "num_reco_pTvseta",
                                                  "N of reco track in (pT-eta) plane",
                                                  nintEta,
                                                  minEta,
                                                  maxEta,
                                                  nintPt,
                                                  minPt,
                                                  maxPt));
  histograms.h_reco2pTvseta.push_back(make2DIfLogY(ibook,
                                                   useLogPt,
                                                   "num_reco2_pTvseta",
                                                   "N of selected reco track in (pT-eta) plane",
                                                   nintEta,
                                                   minEta,
                                                   maxEta,
                                                   nintPt,
                                                   minPt,
                                                   maxPt));
  histograms.h_assoc2pTvseta.push_back(make2DIfLogY(ibook,
                                                    useLogPt,
                                                    "num_assoc(recoToSim)_pTvseta",
                                                    "N of associated (recoToSim) tracks in (pT-eta) plane",
                                                    nintEta,
                                                    minEta,
                                                    maxEta,
                                                    nintPt,
                                                    minPt,
                                                    maxPt));
  histograms.h_looperpTvseta.push_back(make2DIfLogY(ibook,
                                                    useLogPt,
                                                    "num_duplicate_pTvseta",
                                                    "N of associated (recoToSim) duplicate tracks in (pT-eta) plane",
                                                    nintEta,
                                                    minEta,
                                                    maxEta,
                                                    nintPt,
                                                    minPt,
                                                    maxPt));
  if (!doSeedPlots_)
    histograms.h_misidpTvseta.push_back(
        make2DIfLogY(ibook,
                     useLogPt,
                     "num_chargemisid_pTvseta",
                     "N of associated (recoToSim) charge misIDed tracks in (pT-eta) plane",
                     nintEta,
                     minEta,
                     maxEta,
                     nintPt,
                     minPt,
                     maxPt));
  histograms.h_pileuppTvseta.push_back(make2DIfLogY(ibook,
                                                    useLogPt,
                                                    "num_pileup_pTvseta",
                                                    "N of associated (recoToSim) pileup tracks in (pT-eta) plane",
                                                    nintEta,
                                                    minEta,
                                                    maxEta,
                                                    nintPt,
                                                    minPt,
                                                    maxPt));
  //
  histograms.h_recohit.push_back(ibook.book1D("num_reco_hit", "N of reco track vs hit", nintHit, minHit, maxHit));
  histograms.h_assoc2hit.push_back(
      ibook.book1D("num_assoc(recoToSim)_hit", "N of associated (recoToSim) tracks vs hit", nintHit, minHit, maxHit));
  histograms.h_looperhit.push_back(ibook.book1D(
      "num_duplicate_hit", "N of associated (recoToSim) duplicate tracks vs hit", nintHit, minHit, maxHit));
  if (!doSeedPlots_)
    histograms.h_misidhit.push_back(ibook.book1D(
        "num_chargemisid_hit", "N of associated (recoToSim) charge misIDed tracks vs hit", nintHit, minHit, maxHit));
  histograms.h_pileuphit.push_back(
      ibook.book1D("num_pileup_hit", "N of associated (recoToSim) pileup tracks vs hit", nintHit, minHit, maxHit));
  //
  histograms.h_recolayer.push_back(
      ibook.book1D("num_reco_layer", "N of reco track vs layer", nintLayers, minLayers, maxLayers));
  histograms.h_assoc2layer.push_back(ibook.book1D(
      "num_assoc(recoToSim)_layer", "N of associated (recoToSim) tracks vs layer", nintLayers, minLayers, maxLayers));
  histograms.h_looperlayer.push_back(ibook.book1D(
      "num_duplicate_layer", "N of associated (recoToSim) duplicate tracks vs layer", nintLayers, minLayers, maxLayers));
  if (!doSeedPlots_)
    histograms.h_misidlayer.push_back(ibook.book1D("num_chargemisid_layer",
                                                   "N of associated (recoToSim) charge misIDed tracks vs layer",
                                                   nintLayers,
                                                   minLayers,
                                                   maxLayers));
  histograms.h_pileuplayer.push_back(ibook.book1D(
      "num_pileup_layer", "N of associated (recoToSim) pileup tracks vs layer", nintLayers, minLayers, maxLayers));
  //
  histograms.h_recopixellayer.push_back(
      ibook.book1D("num_reco_pixellayer", "N of reco track vs pixellayer", nintLayers, minLayers, maxLayers));
  histograms.h_assoc2pixellayer.push_back(ibook.book1D("num_assoc(recoToSim)_pixellayer",
                                                       "N of associated (recoToSim) tracks vs pixellayer",
                                                       nintLayers,
                                                       minLayers,
                                                       maxLayers));
  histograms.h_looperpixellayer.push_back(ibook.book1D("num_duplicate_pixellayer",
                                                       "N of associated (recoToSim) duplicate tracks vs pixellayer",
                                                       nintLayers,
                                                       minLayers,
                                                       maxLayers));
  if (!doSeedPlots_)
    histograms.h_misidpixellayer.push_back(
        ibook.book1D("num_chargemisid_pixellayer",
                     "N of associated (recoToSim) charge misIDed tracks vs pixellayer",
                     nintLayers,
                     minLayers,
                     maxLayers));
  histograms.h_pileuppixellayer.push_back(ibook.book1D("num_pileup_pixellayer",
                                                       "N of associated (recoToSim) pileup tracks vs pixellayer",
                                                       nintLayers,
                                                       minLayers,
                                                       maxLayers));
  //
  histograms.h_reco3Dlayer.push_back(
      ibook.book1D("num_reco_3Dlayer", "N of reco track vs 3D layer", nintLayers, minLayers, maxLayers));
  histograms.h_assoc23Dlayer.push_back(ibook.book1D("num_assoc(recoToSim)_3Dlayer",
                                                    "N of associated (recoToSim) tracks vs 3D layer",
                                                    nintLayers,
                                                    minLayers,
                                                    maxLayers));
  histograms.h_looper3Dlayer.push_back(ibook.book1D("num_duplicate_3Dlayer",
                                                    "N of associated (recoToSim) duplicate tracks vs 3D layer",
                                                    nintLayers,
                                                    minLayers,
                                                    maxLayers));
  if (!doSeedPlots_)
    histograms.h_misid3Dlayer.push_back(ibook.book1D("num_chargemisid_3Dlayer",
                                                     "N of associated (recoToSim) charge misIDed tracks vs 3D layer",
                                                     nintLayers,
                                                     minLayers,
                                                     maxLayers));
  histograms.h_pileup3Dlayer.push_back(ibook.book1D(
      "num_pileup_3Dlayer", "N of associated (recoToSim) pileup tracks vs 3D layer", nintLayers, minLayers, maxLayers));
  //
  histograms.h_recopu.push_back(ibook.book1D("num_reco_pu", "N of reco track vs pu", nintPu, minPu, maxPu));
  histograms.h_reco2pu.push_back(ibook.book1D("num_reco2_pu", "N of selected reco track vs pu", nintPu, minPu, maxPu));
  histograms.h_assoc2pu.push_back(
      ibook.book1D("num_assoc(recoToSim)_pu", "N of associated (recoToSim) tracks vs pu", nintPu, minPu, maxPu));
  histograms.h_looperpu.push_back(
      ibook.book1D("num_duplicate_pu", "N of associated (recoToSim) duplicate tracks vs pu", nintPu, minPu, maxPu));
  if (!doSeedPlots_)
    histograms.h_misidpu.push_back(ibook.book1D(
        "num_chargemisid_pu", "N of associated (recoToSim) charge misIDed tracks vs pu", nintPu, minPu, maxPu));
  histograms.h_pileuppu.push_back(
      ibook.book1D("num_pileup_pu", "N of associated (recoToSim) pileup tracks vs pu", nintPu, minPu, maxPu));
  //
  histograms.h_recophi.push_back(ibook.book1D("num_reco_phi", "N of reco track vs phi", nintPhi, minPhi, maxPhi));
  histograms.h_assoc2phi.push_back(
      ibook.book1D("num_assoc(recoToSim)_phi", "N of associated (recoToSim) tracks vs phi", nintPhi, minPhi, maxPhi));
  histograms.h_looperphi.push_back(ibook.book1D(
      "num_duplicate_phi", "N of associated (recoToSim) duplicate tracks vs phi", nintPhi, minPhi, maxPhi));
  if (!doSeedPlots_)
    histograms.h_misidphi.push_back(ibook.book1D(
        "num_chargemisid_phi", "N of associated (recoToSim) charge misIDed tracks vs phi", nintPhi, minPhi, maxPhi));
  histograms.h_pileupphi.push_back(
      ibook.book1D("num_pileup_phi", "N of associated (recoToSim) pileup tracks vs phi", nintPhi, minPhi, maxPhi));

  histograms.h_recodxy.push_back(ibook.book1D("num_reco_dxy", "N of reco track vs dxy", nintDxy, minDxy, maxDxy));
  histograms.h_assoc2dxy.push_back(
      ibook.book1D("num_assoc(recoToSim)_dxy", "N of associated (recoToSim) tracks vs dxy", nintDxy, minDxy, maxDxy));
  histograms.h_looperdxy.push_back(
      ibook.book1D("num_duplicate_dxy", "N of associated (recoToSim) looper tracks vs dxy", nintDxy, minDxy, maxDxy));
  if (!doSeedPlots_)
    histograms.h_misiddxy.push_back(ibook.book1D(
        "num_chargemisid_dxy", "N of associated (recoToSim) charge misIDed tracks vs dxy", nintDxy, minDxy, maxDxy));
  histograms.h_pileupdxy.push_back(
      ibook.book1D("num_pileup_dxy", "N of associated (recoToSim) pileup tracks vs dxy", nintDxy, minDxy, maxDxy));

  histograms.h_recodz.push_back(ibook.book1D("num_reco_dz", "N of reco track vs dz", nintDz, minDz, maxDz));
  histograms.h_assoc2dz.push_back(
      ibook.book1D("num_assoc(recoToSim)_dz", "N of associated (recoToSim) tracks vs dz", nintDz, minDz, maxDz));
  histograms.h_looperdz.push_back(
      ibook.book1D("num_duplicate_dz", "N of associated (recoToSim) looper tracks vs dz", nintDz, minDz, maxDz));
  if (!doSeedPlots_)
    histograms.h_misiddz.push_back(ibook.book1D(
        "num_chargemisid_versus_dz", "N of associated (recoToSim) charge misIDed tracks vs dz", nintDz, minDz, maxDz));
  histograms.h_pileupdz.push_back(
      ibook.book1D("num_pileup_dz", "N of associated (recoToSim) pileup tracks vs dz", nintDz, minDz, maxDz));

  histograms.h_recovertpos.push_back(make1DIfLogX(ibook,
                                                  useLogVertpos,
                                                  "num_reco_vertpos",
                                                  "N of reconstructed tracks vs transverse ref point position",
                                                  nintVertpos,
                                                  minVertpos,
                                                  maxVertpos));
  histograms.h_assoc2vertpos.push_back(
      make1DIfLogX(ibook,
                   useLogVertpos,
                   "num_assoc(recoToSim)_vertpos",
                   "N of associated (recoToSim) tracks vs transverse ref point position",
                   nintVertpos,
                   minVertpos,
                   maxVertpos));
  histograms.h_loopervertpos.push_back(
      make1DIfLogX(ibook,
                   useLogVertpos,
                   "num_duplicate_vertpos",
                   "N of associated (recoToSim) looper tracks vs transverse ref point position",
                   nintVertpos,
                   minVertpos,
                   maxVertpos));
  histograms.h_pileupvertpos.push_back(
      make1DIfLogX(ibook,
                   useLogVertpos,
                   "num_pileup_vertpos",
                   "N of associated (recoToSim) pileup tracks vs transverse ref point position",
                   nintVertpos,
                   minVertpos,
                   maxVertpos));

  histograms.h_recozpos.push_back(ibook.book1D(
      "num_reco_zpos", "N of reconstructed tracks vs transverse ref point position", nintZpos, minZpos, maxZpos));
  histograms.h_assoc2zpos.push_back(ibook.book1D("num_assoc(recoToSim)_zpos",
                                                 "N of associated (recoToSim) tracks vs transverse ref point position",
                                                 nintZpos,
                                                 minZpos,
                                                 maxZpos));
  histograms.h_looperzpos.push_back(
      ibook.book1D("num_duplicate_zpos",
                   "N of associated (recoToSim) looper tracks vs transverse ref point position",
                   nintZpos,
                   minZpos,
                   maxZpos));
  histograms.h_pileupzpos.push_back(
      ibook.book1D("num_pileup_zpos",
                   "N of associated (recoToSim) pileup tracks vs transverse ref point position",
                   nintZpos,
                   minZpos,
                   maxZpos));

  histograms.h_recodr.push_back(
      make1DIfLogX(ibook, true, "num_reco_dr", "N of reconstructed tracks vs dR", nintdr, log10(mindr), log10(maxdr)));
  histograms.h_assoc2dr.push_back(make1DIfLogX(ibook,
                                               true,
                                               "num_assoc(recoToSim)_dr",
                                               "N of associated tracks (recoToSim) vs dR",
                                               nintdr,
                                               log10(mindr),
                                               log10(maxdr)));
  histograms.h_looperdr.push_back(make1DIfLogX(ibook,
                                               true,
                                               "num_duplicate_dr",
                                               "N of associated (recoToSim) looper tracks vs dR",
                                               nintdr,
                                               log10(mindr),
                                               log10(maxdr)));
  histograms.h_pileupdr.push_back(make1DIfLogX(ibook,
                                               true,
                                               "num_pileup_dr",
                                               "N of associated (recoToSim) pileup tracks vs dR",
                                               nintdr,
                                               log10(mindr),
                                               log10(maxdr)));

  histograms.h_recodrj.push_back(make1DIfLogX(
      ibook, true, "num_reco_drj", "N of reconstructed tracks vs dR(track,jet)", nintdrj, log10(mindrj), log10(maxdrj)));
  histograms.h_assoc2drj.push_back(make1DIfLogX(ibook,
                                                true,
                                                "num_assoc(recoToSim)_drj",
                                                "N of associated tracks (recoToSim) vs dR(track,jet)",
                                                nintdrj,
                                                log10(mindrj),
                                                log10(maxdrj)));
  histograms.h_looperdrj.push_back(make1DIfLogX(ibook,
                                                true,
                                                "num_duplicate_drj",
                                                "N of associated (recoToSim) looper tracks vs dR(track,jet)",
                                                nintdrj,
                                                log10(mindrj),
                                                log10(maxdrj)));
  histograms.h_pileupdrj.push_back(make1DIfLogX(ibook,
                                                true,
                                                "num_pileup_drj",
                                                "N of associated (recoToSim) pileup tracks vs dR(track,jet)",
                                                nintdrj,
                                                log10(mindrj),
                                                log10(maxdrj)));

  histograms.h_reco_simpvz.push_back(
      ibook.book1D("num_reco_simpvz", "N of reco track vs. sim PV z", nintPVz, minPVz, maxPVz));
  histograms.h_assoc2_simpvz.push_back(ibook.book1D(
      "num_assoc(recoToSim)_simpvz", "N of associated tracks (recoToSim) vs. sim PV z", nintPVz, minPVz, maxPVz));
  histograms.h_looper_simpvz.push_back(ibook.book1D(
      "num_duplicate_simpvz", "N of associated (recoToSim) looper tracks vs. sim PV z", nintPVz, minPVz, maxPVz));
  histograms.h_pileup_simpvz.push_back(ibook.book1D(
      "num_pileup_simpvz", "N of associated (recoToSim) pileup tracks vs. sim PV z", nintPVz, minPVz, maxPVz));

  histograms.h_recochi2.push_back(
      ibook.book1D("num_reco_chi2", "N of reco track vs normalized #chi^{2}", nintChi2, minChi2, maxChi2));
  histograms.h_assoc2chi2.push_back(ibook.book1D("num_assoc(recoToSim)_chi2",
                                                 "N of associated (recoToSim) tracks vs normalized #chi^{2}",
                                                 nintChi2,
                                                 minChi2,
                                                 maxChi2));
  histograms.h_looperchi2.push_back(ibook.book1D("num_duplicate_chi2",
                                                 "N of associated (recoToSim) looper tracks vs normalized #chi^{2}",
                                                 nintChi2,
                                                 minChi2,
                                                 maxChi2));
  if (!doSeedPlots_)
    histograms.h_misidchi2.push_back(
        ibook.book1D("num_chargemisid_chi2",
                     "N of associated (recoToSim) charge misIDed tracks vs normalized #chi^{2}",
                     nintChi2,
                     minChi2,
                     maxChi2));
  histograms.h_pileupchi2.push_back(ibook.book1D("num_pileup_chi2",
                                                 "N of associated (recoToSim) pileup tracks vs normalized #chi^{2}",
                                                 nintChi2,
                                                 minChi2,
                                                 maxChi2));

  histograms.h_recochi2prob.push_back(
      ibook.book1D("num_reco_chi2prob", "N of reco track vs normalized #chi^{2}", 100, 0., 1.));
  histograms.h_assoc2chi2prob.push_back(ibook.book1D(
      "num_assoc(recoToSim)_chi2prob", "N of associated (recoToSim) tracks vs normalized #chi^{2}", 100, 0., 1.));
  histograms.h_looperchi2prob.push_back(ibook.book1D(
      "num_duplicate_chi2prob", "N of associated (recoToSim) looper tracks vs normalized #chi^{2}", 100, 0., 1.));
  if (!doSeedPlots_)
    histograms.h_misidchi2prob.push_back(
        ibook.book1D("num_chargemisid_chi2prob",
                     "N of associated (recoToSim) charge misIDed tracks vs normalized #chi^{2}",
                     100,
                     0.,
                     1.));
  histograms.h_pileupchi2prob.push_back(ibook.book1D(
      "num_pileup_chi2prob", "N of associated (recoToSim) pileup tracks vs normalized #chi^{2}", 100, 0., 1.));

  if (!seedingLayerSetNames.empty()) {
    const auto size = seedingLayerSetNames.size();
    histograms.h_reco_seedingLayerSet.push_back(
        ibook.book1D("num_reco_seedingLayerSet", "N of reco track vs. seedingLayerSet", size, 0, size));
    histograms.h_assoc2_seedingLayerSet.push_back(
        ibook.book1D("num_assoc(recoToSim)_seedingLayerSet",
                     "N of associated track (recoToSim) tracks vs. seedingLayerSet",
                     size,
                     0,
                     size));
    histograms.h_looper_seedingLayerSet.push_back(ibook.book1D(
        "num_duplicate_seedingLayerSet", "N of reco associated (recoToSim) looper vs. seedingLayerSet", size, 0, size));
    histograms.h_pileup_seedingLayerSet.push_back(ibook.book1D(
        "num_pileup_seedingLayerSet", "N of reco associated (recoToSim) pileup vs. seedingLayerSet", size, 0, size));

    setBinLabels(histograms.h_reco_seedingLayerSet.back(), seedingLayerSetNames);
    setBinLabels(histograms.h_assoc2_seedingLayerSet.back(), seedingLayerSetNames);
    setBinLabels(histograms.h_looper_seedingLayerSet.back(), seedingLayerSetNames);
    setBinLabels(histograms.h_pileup_seedingLayerSet.back(), seedingLayerSetNames);
  }

  /////////////////////////////////

  auto bookResolutionPlots1D = [&](std::vector<dqm::reco::MonitorElement*>& vec, auto&&... params) {
    vec.push_back(doResolutionPlots ? ibook.book1D(std::forward<decltype(params)>(params)...) : nullptr);
  };
  auto bookResolutionPlots2D = [&](std::vector<dqm::reco::MonitorElement*>& vec, bool logx, auto&&... params) {
    vec.push_back(doResolutionPlots ? make2DIfLogX(ibook, logx, std::forward<decltype(params)>(params)...) : nullptr);
  };
  auto bookResolutionPlotsProfile2D = [&](std::vector<dqm::reco::MonitorElement*>& vec, auto&&... params) {
    vec.push_back(doResolutionPlots ? ibook.bookProfile2D(std::forward<decltype(params)>(params)...) : nullptr);
  };

  bookResolutionPlots1D(histograms.h_eta, "eta", "pseudorapidity residue", 1000, -0.1, 0.1);
  bookResolutionPlots1D(histograms.h_pt, "pullPt", "pull of p_{t}", 100, -10, 10);
  bookResolutionPlots1D(histograms.h_pullTheta, "pullTheta", "pull of #theta parameter", 250, -25, 25);
  bookResolutionPlots1D(histograms.h_pullPhi, "pullPhi", "pull of #phi parameter", 250, -25, 25);
  bookResolutionPlots1D(histograms.h_pullDxy, "pullDxy", "pull of dxy parameter", 250, -25, 25);
  bookResolutionPlots1D(histograms.h_pullDz, "pullDz", "pull of dz parameter", 250, -25, 25);
  bookResolutionPlots1D(histograms.h_pullQoverp, "pullQoverp", "pull of qoverp parameter", 250, -25, 25);

  /* TO BE FIXED -----------
  if (associators[ww]=="TrackAssociatorByChi2"){
    histograms.h_assochi2.push_back( ibook.book1D("assocChi2","track association #chi^{2}",1000000,0,100000) );
    histograms.h_assochi2_prob.push_back(ibook.book1D("assocChi2_prob","probability of association #chi^{2}",100,0,1));
  } else if (associators[ww]=="quickTrackAssociatorByHits"){
    histograms.h_assocFraction.push_back( ibook.book1D("assocFraction","fraction of shared hits",200,0,2) );
    histograms.h_assocSharedHit.push_back(ibook.book1D("assocSharedHit","number of shared hits",20,0,20));
  }
  */
  histograms.h_assocFraction.push_back(ibook.book1D("assocFraction", "fraction of shared hits", 200, 0, 2));
  histograms.h_assocSharedHit.push_back(ibook.book1D("assocSharedHit", "number of shared hits", 41, -0.5, 40.5));
  // ----------------------

  // use the standard error of the mean as the errors in the profile
  histograms.chi2_vs_nhits.push_back(
      ibook.bookProfile("chi2mean_vs_nhits", "mean #chi^{2} vs nhits", nintHit, minHit, maxHit, 100, 0, 10, " "));

  bookResolutionPlots2D(
      histograms.etares_vs_eta, false, "etares_vs_eta", "etaresidue vs eta", nintEta, minEta, maxEta, 200, -0.1, 0.1);
  bookResolutionPlots2D(
      histograms.nrec_vs_nsim,
      false,
      "nrec_vs_nsim",
      "Number of selected reco tracks vs. number of selected sim tracks;TrackingParticles;Reco tracks",
      nintTracks,
      minTracks,
      maxTracks,
      nintTracks,
      minTracks,
      maxTracks);

  histograms.chi2_vs_eta.push_back(
      ibook.bookProfile("chi2mean", "mean #chi^{2} vs #eta", nintEta, minEta, maxEta, 200, 0, 20, " "));
  histograms.chi2_vs_phi.push_back(
      ibook.bookProfile("chi2mean_vs_phi", "mean #chi^{2} vs #phi", nintPhi, minPhi, maxPhi, 200, 0, 20, " "));
  histograms.chi2_vs_pt.push_back(
      makeProfileIfLogX(ibook, useLogPt, "chi2mean_vs_pt", "mean #chi^{2} vs p_{T}", nintPt, minPt, maxPt, 0, 20));

  histograms.assoc_chi2_vs_eta.push_back(
      ibook.bookProfile("assoc_chi2mean", "mean #chi^{2} vs #eta", nintEta, minEta, maxEta, 200, 0., 20., " "));
  histograms.assoc_chi2prob_vs_eta.push_back(ibook.bookProfile(
      "assoc_chi2prob_vs_eta", "mean #chi^{2} probability vs #eta", nintEta, minEta, maxEta, 100, 0., 1., " "));
  histograms.assoc_chi2_vs_pt.push_back(makeProfileIfLogX(
      ibook, useLogPt, "assoc_chi2mean_vs_pt", "mean #chi^{2} vs p_{T}", nintPt, minPt, maxPt, 0., 20.));
  histograms.assoc_chi2prob_vs_pt.push_back(makeProfileIfLogX(
      ibook, useLogPt, "assoc_chi2prob_vs_pt", "mean #chi^{2} probability vs p_{T}", nintPt, minPt, maxPt, 0., 20.));

  histograms.nhits_vs_eta.push_back(
      ibook.bookProfile("hits_eta", "mean hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));
  histograms.nPXBhits_vs_eta.push_back(ibook.bookProfile(
      "PXBhits_vs_eta", "mean # PXB its vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));
  histograms.nPXFhits_vs_eta.push_back(ibook.bookProfile(
      "PXFhits_vs_eta", "mean # PXF hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));
  histograms.nPXLhits_vs_eta.push_back(ibook.bookProfile(
      "PXLhits_vs_eta", "mean # PXL hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));
  histograms.nTIBhits_vs_eta.push_back(ibook.bookProfile(
      "TIBhits_vs_eta", "mean # TIB hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));
  histograms.nTIDhits_vs_eta.push_back(ibook.bookProfile(
      "TIDhits_vs_eta", "mean # TID hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));
  histograms.nTOBhits_vs_eta.push_back(ibook.bookProfile(
      "TOBhits_vs_eta", "mean # TOB hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));
  histograms.nTEChits_vs_eta.push_back(ibook.bookProfile(
      "TEChits_vs_eta", "mean # TEC hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));
  histograms.nSTRIPhits_vs_eta.push_back(ibook.bookProfile(
      "STRIPhits_vs_eta", "mean # STRIP hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));

  histograms.nLayersWithMeas_vs_eta.push_back(ibook.bookProfile("LayersWithMeas_eta",
                                                                "mean # Layers with measurement vs eta",
                                                                nintEta,
                                                                minEta,
                                                                maxEta,
                                                                nintLayers,
                                                                minLayers,
                                                                maxLayers,
                                                                " "));
  histograms.nPXLlayersWithMeas_vs_eta.push_back(ibook.bookProfile("PXLlayersWithMeas_vs_eta",
                                                                   "mean # PXL Layers with measurement vs eta",
                                                                   nintEta,
                                                                   minEta,
                                                                   maxEta,
                                                                   nintLayers,
                                                                   minLayers,
                                                                   maxLayers,
                                                                   " "));
  histograms.nSTRIPlayersWithMeas_vs_eta.push_back(ibook.bookProfile("STRIPlayersWithMeas_vs_eta",
                                                                     "mean # STRIP Layers with measurement vs eta",
                                                                     nintEta,
                                                                     minEta,
                                                                     maxEta,
                                                                     nintLayers,
                                                                     minLayers,
                                                                     maxLayers,
                                                                     " "));
  histograms.nSTRIPlayersWith1dMeas_vs_eta.push_back(ibook.bookProfile("STRIPlayersWith1dMeas_vs_eta",
                                                                       "mean # STRIP Layers with 1D measurement vs eta",
                                                                       nintEta,
                                                                       minEta,
                                                                       maxEta,
                                                                       nintLayers,
                                                                       minLayers,
                                                                       maxLayers,
                                                                       " "));
  histograms.nSTRIPlayersWith2dMeas_vs_eta.push_back(ibook.bookProfile("STRIPlayersWith2dMeas_vs_eta",
                                                                       "mean # STRIP Layers with 2D measurement vs eta",
                                                                       nintEta,
                                                                       minEta,
                                                                       maxEta,
                                                                       nintLayers,
                                                                       minLayers,
                                                                       maxLayers,
                                                                       " "));

  if (doMTDPlots_) {
    histograms.nMTDhits_vs_eta.push_back(ibook.bookProfile(
        "MTDhits_vs_eta", "mean # MTD hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));

    histograms.nBTLhits_vs_eta.push_back(ibook.bookProfile(
        "BTLhits_vs_eta", "mean # BTL hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));

    histograms.nETLhits_vs_eta.push_back(ibook.bookProfile(
        "ETLhits_vs_eta", "mean # ETL hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));
  }

  histograms.nhits_vs_phi.push_back(
      ibook.bookProfile("hits_phi", "mean # hits vs #phi", nintPhi, minPhi, maxPhi, nintHit, minHit, maxHit, " "));

  histograms.nlosthits_vs_eta.push_back(ibook.bookProfile(
      "losthits_vs_eta", "mean # lost hits vs eta", nintEta, minEta, maxEta, nintHit, minHit, maxHit, " "));

  //resolution of track parameters
  //                       dPt/Pt    cotTheta        Phi            TIP            LIP
  // log10(pt)<0.5        100,0.1    240,0.08     100,0.015      100,0.1000    150,0.3000
  // 0.5<log10(pt)<1.5    100,0.1    120,0.01     100,0.003      100,0.0100    150,0.0500
  // >1.5                 100,0.3    100,0.005    100,0.0008     100,0.0060    120,0.0300

  bookResolutionPlots2D(histograms.ptres_vs_eta,
                        false,
                        "ptres_vs_eta",
                        "ptres_vs_eta",
                        nintEta,
                        minEta,
                        maxEta,
                        ptRes_nbin,
                        ptRes_rangeMin,
                        ptRes_rangeMax);

  bookResolutionPlots2D(histograms.ptres_vs_phi,
                        false,
                        "ptres_vs_phi",
                        "p_{t} res vs #phi",
                        nintPhi,
                        minPhi,
                        maxPhi,
                        ptRes_nbin,
                        ptRes_rangeMin,
                        ptRes_rangeMax);

  bookResolutionPlots2D(histograms.ptres_vs_pt,
                        useLogPt,
                        "ptres_vs_pt",
                        "ptres_vs_pt",
                        nintPt,
                        minPt,
                        maxPt,
                        ptRes_nbin,
                        ptRes_rangeMin,
                        ptRes_rangeMax);

  bookResolutionPlots2D(histograms.cotThetares_vs_eta,
                        false,
                        "cotThetares_vs_eta",
                        "cotThetares_vs_eta",
                        nintEta,
                        minEta,
                        maxEta,
                        cotThetaRes_nbin,
                        cotThetaRes_rangeMin,
                        cotThetaRes_rangeMax);

  bookResolutionPlots2D(histograms.cotThetares_vs_pt,
                        useLogPt,
                        "cotThetares_vs_pt",
                        "cotThetares_vs_pt",
                        nintPt,
                        minPt,
                        maxPt,
                        cotThetaRes_nbin,
                        cotThetaRes_rangeMin,
                        cotThetaRes_rangeMax);

  bookResolutionPlots2D(histograms.phires_vs_eta,
                        false,
                        "phires_vs_eta",
                        "phires_vs_eta",
                        nintEta,
                        minEta,
                        maxEta,
                        phiRes_nbin,
                        phiRes_rangeMin,
                        phiRes_rangeMax);

  bookResolutionPlots2D(histograms.phires_vs_pt,
                        useLogPt,
                        "phires_vs_pt",
                        "phires_vs_pt",
                        nintPt,
                        minPt,
                        maxPt,
                        phiRes_nbin,
                        phiRes_rangeMin,
                        phiRes_rangeMax);

  bookResolutionPlots2D(histograms.phires_vs_phi,
                        false,
                        "phires_vs_phi",
                        "#phi res vs #phi",
                        nintPhi,
                        minPhi,
                        maxPhi,
                        phiRes_nbin,
                        phiRes_rangeMin,
                        phiRes_rangeMax);

  bookResolutionPlots2D(histograms.dxyres_vs_eta,
                        false,
                        "dxyres_vs_eta",
                        "dxyres_vs_eta",
                        nintEta,
                        minEta,
                        maxEta,
                        dxyRes_nbin,
                        dxyRes_rangeMin,
                        dxyRes_rangeMax);

  bookResolutionPlots2D(histograms.dxyres_vs_pt,
                        useLogPt,
                        "dxyres_vs_pt",
                        "dxyres_vs_pt",
                        nintPt,
                        minPt,
                        maxPt,
                        dxyRes_nbin,
                        dxyRes_rangeMin,
                        dxyRes_rangeMax);

  bookResolutionPlots2D(histograms.dzres_vs_eta,
                        false,
                        "dzres_vs_eta",
                        "dzres_vs_eta",
                        nintEta,
                        minEta,
                        maxEta,
                        dzRes_nbin,
                        dzRes_rangeMin,
                        dzRes_rangeMax);

  bookResolutionPlots2D(histograms.dzres_vs_pt,
                        useLogPt,
                        "dzres_vs_pt",
                        "dzres_vs_pt",
                        nintPt,
                        minPt,
                        maxPt,
                        dzRes_nbin,
                        dzRes_rangeMin,
                        dzRes_rangeMax);

  bookResolutionPlotsProfile2D(histograms.ptmean_vs_eta_phi,
                               "ptmean_vs_eta_phi",
                               "mean p_{t} vs #eta and #phi",
                               nintPhi,
                               minPhi,
                               maxPhi,
                               nintEta,
                               minEta,
                               maxEta,
                               1000,
                               0,
                               1000);
  bookResolutionPlotsProfile2D(histograms.phimean_vs_eta_phi,
                               "phimean_vs_eta_phi",
                               "mean #phi vs #eta and #phi",
                               nintPhi,
                               minPhi,
                               maxPhi,
                               nintEta,
                               minEta,
                               maxEta,
                               nintPhi,
                               minPhi,
                               maxPhi);

  //pulls of track params vs eta: to be used with fitslicesytool
  bookResolutionPlots2D(
      histograms.dxypull_vs_eta, false, "dxypull_vs_eta", "dxypull_vs_eta", nintEta, minEta, maxEta, 100, -10, 10);
  bookResolutionPlots2D(
      histograms.ptpull_vs_eta, false, "ptpull_vs_eta", "ptpull_vs_eta", nintEta, minEta, maxEta, 100, -10, 10);
  bookResolutionPlots2D(
      histograms.dzpull_vs_eta, false, "dzpull_vs_eta", "dzpull_vs_eta", nintEta, minEta, maxEta, 100, -10, 10);
  bookResolutionPlots2D(
      histograms.phipull_vs_eta, false, "phipull_vs_eta", "phipull_vs_eta", nintEta, minEta, maxEta, 100, -10, 10);
  bookResolutionPlots2D(
      histograms.thetapull_vs_eta, false, "thetapull_vs_eta", "thetapull_vs_eta", nintEta, minEta, maxEta, 100, -10, 10);

  //      histograms.h_ptshiftetamean.push_back( ibook.book1D("h_ptshifteta_Mean","<#deltapT/pT>[%] vs #eta",nintEta,minEta,maxEta) );

  //pulls of track params vs phi
  bookResolutionPlots2D(
      histograms.ptpull_vs_phi, false, "ptpull_vs_phi", "p_{t} pull vs #phi", nintPhi, minPhi, maxPhi, 100, -10, 10);
  bookResolutionPlots2D(
      histograms.phipull_vs_phi, false, "phipull_vs_phi", "#phi pull vs #phi", nintPhi, minPhi, maxPhi, 100, -10, 10);
  bookResolutionPlots2D(histograms.thetapull_vs_phi,
                        false,
                        "thetapull_vs_phi",
                        "#theta pull vs #phi",
                        nintPhi,
                        minPhi,
                        maxPhi,
                        100,
                        -10,
                        10);

  bookResolutionPlots2D(histograms.nrecHit_vs_nsimHit_rec2sim,
                        false,
                        "nrecHit_vs_nsimHit_rec2sim",
                        "nrecHit vs nsimHit (Rec2simAssoc)",
                        nintHit,
                        minHit,
                        maxHit,
                        nintHit,
                        minHit,
                        maxHit);
}

void MTVHistoProducerAlgoForTracker::bookRecoPVAssociationHistos(DQMStore::IBooker& ibook, Histograms& histograms) {
  histograms.h_recodxypv.push_back(
      ibook.book1D("num_reco_dxypv", "N of reco track vs dxy(PV)", nintDxy, minDxy, maxDxy));
  histograms.h_assoc2dxypv.push_back(ibook.book1D(
      "num_assoc(recoToSim)_dxypv", "N of associated (recoToSim) tracks vs dxy(PV)", nintDxy, minDxy, maxDxy));
  histograms.h_looperdxypv.push_back(ibook.book1D(
      "num_duplicate_dxypv", "N of associated (recoToSim) looper tracks vs dxy(PV)", nintDxy, minDxy, maxDxy));
  if (!doSeedPlots_)
    histograms.h_misiddxypv.push_back(ibook.book1D("num_chargemisid_dxypv",
                                                   "N of associated (recoToSim) charge misIDed tracks vs dxy(PV)",
                                                   nintDxy,
                                                   minDxy,
                                                   maxDxy));
  histograms.h_pileupdxypv.push_back(ibook.book1D(
      "num_pileup_dxypv", "N of associated (recoToSim) pileup tracks vs dxy(PV)", nintDxy, minDxy, maxDxy));

  histograms.h_recodzpv.push_back(ibook.book1D("num_reco_dzpv", "N of reco track vs dz(PV)", nintDz, minDz, maxDz));
  histograms.h_assoc2dzpv.push_back(
      ibook.book1D("num_assoc(recoToSim)_dzpv", "N of associated (recoToSim) tracks vs dz(PV)", nintDz, minDz, maxDz));
  histograms.h_looperdzpv.push_back(
      ibook.book1D("num_duplicate_dzpv", "N of associated (recoToSim) looper tracks vs dz(PV)", nintDz, minDz, maxDz));
  if (!doSeedPlots_)
    histograms.h_misiddzpv.push_back(ibook.book1D("num_chargemisid_versus_dzpv",
                                                  "N of associated (recoToSim) charge misIDed tracks vs dz(PV)",
                                                  nintDz,
                                                  minDz,
                                                  maxDz));
  histograms.h_pileupdzpv.push_back(
      ibook.book1D("num_pileup_dzpv", "N of associated (recoToSim) pileup tracks vs dz(PV)", nintDz, minDz, maxDz));

  histograms.h_recodxypvzoomed.push_back(ibook.book1D(
      "num_reco_dxypv_zoomed", "N of reco track vs dxy(PV)", nintDxy, minDxy / dxyDzZoom, maxDxy / dxyDzZoom));
  histograms.h_assoc2dxypvzoomed.push_back(ibook.book1D("num_assoc(recoToSim)_dxypv_zoomed",
                                                        "N of associated (recoToSim) tracks vs dxy(PV)",
                                                        nintDxy,
                                                        minDxy / dxyDzZoom,
                                                        maxDxy / dxyDzZoom));
  histograms.h_looperdxypvzoomed.push_back(ibook.book1D("num_duplicate_dxypv_zoomed",
                                                        "N of associated (recoToSim) looper tracks vs dxy(PV)",
                                                        nintDxy,
                                                        minDxy / dxyDzZoom,
                                                        maxDxy / dxyDzZoom));
  if (!doSeedPlots_)
    histograms.h_misiddxypvzoomed.push_back(ibook.book1D("num_chargemisid_dxypv_zoomed",
                                                         "N of associated (recoToSim) charge misIDed tracks vs dxy(PV)",
                                                         nintDxy,
                                                         minDxy / dxyDzZoom,
                                                         maxDxy / dxyDzZoom));
  histograms.h_pileupdxypvzoomed.push_back(ibook.book1D("num_pileup_dxypv_zoomed",
                                                        "N of associated (recoToSim) pileup tracks vs dxy(PV)",
                                                        nintDxy,
                                                        minDxy / dxyDzZoom,
                                                        maxDxy / dxyDzZoom));

  histograms.h_recodzpvzoomed.push_back(
      ibook.book1D("num_reco_dzpv_zoomed", "N of reco track vs dz(PV)", nintDz, minDz / dxyDzZoom, maxDz / dxyDzZoom));
  histograms.h_assoc2dzpvzoomed.push_back(ibook.book1D("num_assoc(recoToSim)_dzpv_zoomed",
                                                       "N of associated (recoToSim) tracks vs dz(PV)",
                                                       nintDz,
                                                       minDz / dxyDzZoom,
                                                       maxDz / dxyDzZoom));
  histograms.h_looperdzpvzoomed.push_back(ibook.book1D("num_duplicate_dzpv_zoomed",
                                                       "N of associated (recoToSim) looper tracks vs dz(PV)",
                                                       nintDz,
                                                       minDz / dxyDzZoom,
                                                       maxDz / dxyDzZoom));
  if (!doSeedPlots_)
    histograms.h_misiddzpvzoomed.push_back(ibook.book1D("num_chargemisid_versus_dzpv_zoomed",
                                                        "N of associated (recoToSim) charge misIDed tracks vs dz(PV)",
                                                        nintDz,
                                                        minDz / dxyDzZoom,
                                                        maxDz / dxyDzZoom));
  histograms.h_pileupdzpvzoomed.push_back(ibook.book1D("num_pileup_dzpv_zoomed",
                                                       "N of associated (recoToSim) pileup tracks vs dz(PV)",
                                                       nintDz,
                                                       minDz / dxyDzZoom,
                                                       maxDz / dxyDzZoom));

  histograms.h_reco_dzpvcut.push_back(
      ibook.book1D("num_reco_dzpvcut", "N of reco track vs dz(PV)", nintDzpvCum, 0, maxDzpvCum));
  histograms.h_assoc2_dzpvcut.push_back(ibook.book1D(
      "num_assoc(recoToSim)_dzpvcut", "N of associated (recoToSim) tracks vs dz(PV)", nintDzpvCum, 0, maxDzpvCum));
  histograms.h_pileup_dzpvcut.push_back(ibook.book1D(
      "num_pileup_dzpvcut", "N of associated (recoToSim) pileup tracks vs dz(PV)", nintDzpvCum, 0, maxDzpvCum));

  histograms.h_reco_dzpvcut_pt.push_back(
      ibook.book1D("num_reco_dzpvcut_pt", "#sump_{T} of reco track vs dz(PV)", nintDzpvCum, 0, maxDzpvCum));
  histograms.h_assoc2_dzpvcut_pt.push_back(ibook.book1D("num_assoc(recoToSim)_dzpvcut_pt",
                                                        "#sump_{T} of associated (recoToSim) tracks vs dz(PV)",
                                                        nintDzpvCum,
                                                        0,
                                                        maxDzpvCum));
  histograms.h_pileup_dzpvcut_pt.push_back(ibook.book1D("num_pileup_dzpvcut_pt",
                                                        "#sump_{T} of associated (recoToSim) pileup tracks vs dz(PV)",
                                                        nintDzpvCum,
                                                        0,
                                                        maxDzpvCum));
  histograms.h_reco_dzpvcut_pt.back()->enableSumw2();
  histograms.h_assoc2_dzpvcut_pt.back()->enableSumw2();
  histograms.h_pileup_dzpvcut_pt.back()->enableSumw2();

  histograms.h_reco_dzpvsigcut.push_back(
      ibook.book1D("num_reco_dzpvsigcut", "N of reco track vs dz(PV)/dzError", nintDzpvsigCum, 0, maxDzpvsigCum));
  histograms.h_assoc2_dzpvsigcut.push_back(ibook.book1D("num_assoc(recoToSim)_dzpvsigcut",
                                                        "N of associated (recoToSim) tracks vs dz(PV)/dzError",
                                                        nintDzpvsigCum,
                                                        0,
                                                        maxDzpvsigCum));
  histograms.h_pileup_dzpvsigcut.push_back(ibook.book1D("num_pileup_dzpvsigcut",
                                                        "N of associated (recoToSim) pileup tracks vs dz(PV)/dzError",
                                                        nintDzpvsigCum,
                                                        0,
                                                        maxDzpvsigCum));

  histograms.h_reco_dzpvsigcut_pt.push_back(ibook.book1D(
      "num_reco_dzpvsigcut_pt", "#sump_{T} of reco track vs dz(PV)/dzError", nintDzpvsigCum, 0, maxDzpvsigCum));
  histograms.h_assoc2_dzpvsigcut_pt.push_back(
      ibook.book1D("num_assoc(recoToSim)_dzpvsigcut_pt",
                   "#sump_{T} of associated (recoToSim) tracks vs dz(PV)/dzError",
                   nintDzpvsigCum,
                   0,
                   maxDzpvsigCum));
  histograms.h_pileup_dzpvsigcut_pt.push_back(
      ibook.book1D("num_pileup_dzpvsigcut_pt",
                   "#sump_{T} of associated (recoToSim) pileup tracks vs dz(PV)/dzError",
                   nintDzpvsigCum,
                   0,
                   maxDzpvsigCum));
  histograms.h_reco_dzpvsigcut_pt.back()->enableSumw2();
  histograms.h_assoc2_dzpvsigcut_pt.back()->enableSumw2();
  histograms.h_pileup_dzpvsigcut_pt.back()->enableSumw2();
}

void MTVHistoProducerAlgoForTracker::bookRecodEdxHistos(DQMStore::IBooker& ibook, Histograms& histograms) {
  // dE/dx stuff
  histograms.h_dedx_estim.emplace_back();
  histograms.h_dedx_estim.back().push_back(
      ibook.book1D("h_dedx_estim1", "dE/dx estimator 1", nintDeDx, minDeDx, maxDeDx));
  histograms.h_dedx_estim.back().push_back(
      ibook.book1D("h_dedx_estim2", "dE/dx estimator 2", nintDeDx, minDeDx, maxDeDx));

  histograms.h_dedx_nom.emplace_back();
  histograms.h_dedx_nom.back().push_back(
      ibook.book1D("h_dedx_nom1", "dE/dx number of measurements", nintHit, minHit, maxHit));
  histograms.h_dedx_nom.back().push_back(
      ibook.book1D("h_dedx_nom2", "dE/dx number of measurements", nintHit, minHit, maxHit));

  histograms.h_dedx_sat.emplace_back();
  histograms.h_dedx_sat.back().push_back(
      ibook.book1D("h_dedx_sat1", "dE/dx number of measurements with saturation", nintHit, minHit, maxHit));
  histograms.h_dedx_sat.back().push_back(
      ibook.book1D("h_dedx_sat2", "dE/dx number of measurements with saturation", nintHit, minHit, maxHit));
}

void MTVHistoProducerAlgoForTracker::bookSeedHistos(DQMStore::IBooker& ibook, Histograms& histograms) {
  histograms.h_seedsFitFailed.push_back(
      ibook.book1D("seeds_fitFailed", "Number of seeds for which the fit failed", nintTracks, minTracks, maxTracks));
  histograms.h_seedsFitFailedFraction.push_back(
      ibook.book1D("seeds_fitFailedFraction", "Fraction of seeds for which the fit failed", 100, 0, 1));
}

void MTVHistoProducerAlgoForTracker::bookMVAHistos(DQMStore::IBooker& ibook, Histograms& histograms, size_t nMVAs) {
  histograms.h_reco_mva.emplace_back();
  histograms.h_assoc2_mva.emplace_back();

  histograms.h_reco_mvacut.emplace_back();
  histograms.h_assoc_mvacut.emplace_back();
  histograms.h_assoc2_mvacut.emplace_back();
  histograms.h_simul2_mvacut.emplace_back();

  histograms.h_reco_mva_hp.emplace_back();
  histograms.h_assoc2_mva_hp.emplace_back();

  histograms.h_reco_mvacut_hp.emplace_back();
  histograms.h_assoc_mvacut_hp.emplace_back();
  histograms.h_assoc2_mvacut_hp.emplace_back();
  histograms.h_simul2_mvacut_hp.emplace_back();

  histograms.h_assoc2_mva_vs_pt.emplace_back();
  histograms.h_fake_mva_vs_pt.emplace_back();
  histograms.h_assoc2_mva_vs_pt_hp.emplace_back();
  histograms.h_fake_mva_vs_pt_hp.emplace_back();
  histograms.h_assoc2_mva_vs_eta.emplace_back();
  histograms.h_fake_mva_vs_eta.emplace_back();
  histograms.h_assoc2_mva_vs_eta_hp.emplace_back();
  histograms.h_fake_mva_vs_eta_hp.emplace_back();

  for (size_t i = 1; i <= nMVAs; ++i) {
    auto istr = std::to_string(i);
    std::string pfix;

    if (i == 1) {
      histograms.h_reco_mva_hp.back().emplace_back();
      histograms.h_assoc2_mva_hp.back().emplace_back();

      histograms.h_reco_mvacut_hp.back().emplace_back();
      histograms.h_assoc_mvacut_hp.back().emplace_back();
      histograms.h_assoc2_mvacut_hp.back().emplace_back();
      histograms.h_simul2_mvacut_hp.back().emplace_back();

      histograms.h_assoc2_mva_vs_pt_hp.back().emplace_back();
      histograms.h_fake_mva_vs_pt_hp.back().emplace_back();
      histograms.h_assoc2_mva_vs_eta_hp.back().emplace_back();
      histograms.h_fake_mva_vs_eta_hp.back().emplace_back();
    } else {
      pfix = " (not loose-selected)";
      std::string pfix2 = " (not HP-selected)";

      histograms.h_reco_mva_hp.back().push_back(ibook.book1D(
          "num_reco_mva" + istr + "_hp", "N of reco track after vs MVA" + istr + pfix2, nintMVA, minMVA, maxMVA));
      histograms.h_assoc2_mva_hp.back().push_back(
          ibook.book1D("num_assoc(recoToSim)_mva" + istr + "_hp",
                       "N of associated tracks (recoToSim) vs MVA" + istr + pfix2,
                       nintMVA,
                       minMVA,
                       maxMVA));

      histograms.h_reco_mvacut_hp.back().push_back(ibook.book1D("num_reco_mva" + istr + "cut" + "_hp",
                                                                "N of reco track vs cut on MVA" + istr + pfix2,
                                                                nintMVA,
                                                                minMVA,
                                                                maxMVA));
      histograms.h_assoc_mvacut_hp.back().push_back(
          ibook.book1D("num_assoc(simToReco)_mva" + istr + "cut_hp",
                       "N of associated tracks (simToReco) vs cut on MVA" + istr + pfix2,
                       nintMVA,
                       minMVA,
                       maxMVA));
      histograms.h_assoc2_mvacut_hp.back().push_back(
          ibook.book1D("num_assoc(recoToSim)_mva" + istr + "cut_hp",
                       "N of associated tracks (recoToSim) vs cut on MVA" + istr + pfix2,
                       nintMVA,
                       minMVA,
                       maxMVA));
      histograms.h_simul2_mvacut_hp.back().push_back(
          ibook.book1D("num_simul2_mva" + istr + "cut_hp",
                       "N of simulated tracks (associated to any track) vs cut on MVA" + istr + pfix2,
                       nintMVA,
                       minMVA,
                       maxMVA));

      histograms.h_assoc2_mva_vs_pt_hp.back().push_back(
          makeProfileIfLogX(ibook,
                            useLogPt,
                            ("mva_assoc(recoToSim)_mva" + istr + "_pT_hp").c_str(),
                            ("MVA" + istr + " of associated tracks (recoToSim) vs. track p_{T}" + pfix2).c_str(),
                            nintPt,
                            minPt,
                            maxPt,
                            minMVA,
                            maxMVA));
      histograms.h_fake_mva_vs_pt_hp.back().push_back(
          makeProfileIfLogX(ibook,
                            useLogPt,
                            ("mva_fake_mva" + istr + "pT_hp").c_str(),
                            ("MVA" + istr + " of non-associated tracks (recoToSim) vs. track p_{T}" + pfix2).c_str(),
                            nintPt,
                            minPt,
                            maxPt,
                            minMVA,
                            maxMVA));
      histograms.h_assoc2_mva_vs_eta_hp.back().push_back(
          ibook.bookProfile("mva_assoc(recoToSim)_mva" + istr + "_eta_hp",
                            "MVA" + istr + " of associated tracks (recoToSim) vs. track #eta" + pfix2,
                            nintEta,
                            minEta,
                            maxEta,
                            nintMVA,
                            minMVA,
                            maxMVA));
      histograms.h_fake_mva_vs_eta_hp.back().push_back(
          ibook.bookProfile("mva_fake_mva" + istr + "eta_hp",
                            "MVA" + istr + " of non-associated tracks (recoToSim) vs. track #eta" + pfix2,
                            nintEta,
                            minEta,
                            maxEta,
                            nintMVA,
                            minMVA,
                            maxMVA));
    }

    histograms.h_reco_mva.back().push_back(
        ibook.book1D("num_reco_mva" + istr, "N of reco track vs MVA" + istr + pfix, nintMVA, minMVA, maxMVA));
    histograms.h_assoc2_mva.back().push_back(ibook.book1D("num_assoc(recoToSim)_mva" + istr,
                                                          "N of associated tracks (recoToSim) vs MVA" + istr + pfix,
                                                          nintMVA,
                                                          minMVA,
                                                          maxMVA));

    histograms.h_reco_mvacut.back().push_back(ibook.book1D(
        "num_reco_mva" + istr + "cut", "N of reco track vs cut on MVA" + istr + pfix, nintMVA, minMVA, maxMVA));
    histograms.h_assoc_mvacut.back().push_back(
        ibook.book1D("num_assoc(simToReco)_mva" + istr + "cut",
                     "N of associated tracks (simToReco) vs cut on MVA" + istr + pfix,
                     nintMVA,
                     minMVA,
                     maxMVA));
    histograms.h_assoc2_mvacut.back().push_back(
        ibook.book1D("num_assoc(recoToSim)_mva" + istr + "cut",
                     "N of associated tracks (recoToSim) vs cut on MVA" + istr + pfix,
                     nintMVA,
                     minMVA,
                     maxMVA));
    histograms.h_simul2_mvacut.back().push_back(
        ibook.book1D("num_simul2_mva" + istr + "cut",
                     "N of simulated tracks (associated to any track) vs cut on MVA" + istr + pfix,
                     nintMVA,
                     minMVA,
                     maxMVA));

    histograms.h_assoc2_mva_vs_pt.back().push_back(
        makeProfileIfLogX(ibook,
                          useLogPt,
                          ("mva_assoc(recoToSim)_mva" + istr + "_pT").c_str(),
                          ("MVA" + istr + " of associated tracks (recoToSim) vs. track p_{T}" + pfix).c_str(),
                          nintPt,
                          minPt,
                          maxPt,
                          minMVA,
                          maxMVA));
    histograms.h_fake_mva_vs_pt.back().push_back(
        makeProfileIfLogX(ibook,
                          useLogPt,
                          ("mva_fake_mva" + istr + "_pT").c_str(),
                          ("MVA" + istr + " of non-associated tracks (recoToSim) vs. track p_{T}" + pfix).c_str(),
                          nintPt,
                          minPt,
                          maxPt,
                          minMVA,
                          maxMVA));
    histograms.h_assoc2_mva_vs_eta.back().push_back(
        ibook.bookProfile("mva_assoc(recoToSim)_mva" + istr + "_eta",
                          "MVA" + istr + " of associated tracks (recoToSim) vs. track #eta" + pfix,
                          nintEta,
                          minEta,
                          maxEta,
                          nintMVA,
                          minMVA,
                          maxMVA));
    histograms.h_fake_mva_vs_eta.back().push_back(
        ibook.bookProfile("mva_fake_mva" + istr + "_eta",
                          "MVA" + istr + " of non-associated tracks (recoToSim) vs. track #eta" + pfix,
                          nintEta,
                          minEta,
                          maxEta,
                          nintMVA,
                          minMVA,
                          maxMVA));
  }
}

void MTVHistoProducerAlgoForTracker::fill_generic_simTrack_histos(const Histograms& histograms,
                                                                  const TrackingParticle::Vector& momentumTP,
                                                                  const TrackingParticle::Point& vertexTP,
                                                                  int bx) const {
  if (bx == 0) {
    histograms.h_ptSIM->Fill(sqrt(momentumTP.perp2()));
    histograms.h_etaSIM->Fill(momentumTP.eta());
    histograms.h_vertposSIM->Fill(sqrt(vertexTP.perp2()));
  }
  histograms.h_bunchxSIM->Fill(bx);
}

void MTVHistoProducerAlgoForTracker::fill_recoAssociated_simTrack_histos(
    const Histograms& histograms,
    int count,
    const TrackingParticle& tp,
    const TrackingParticle::Vector& momentumTP,
    const TrackingParticle::Point& vertexTP,
    double dxySim,
    double dzSim,
    double dxyPVSim,
    double dzPVSim,
    int nSimHits,
    int nSimLayers,
    int nSimPixelLayers,
    int nSimStripMonoAndStereoLayers,
    const reco::Track* track,
    int numVertices,
    double dR,
    double dRJet,
    const math::XYZPoint* pvPosition,
    const TrackingVertex::LorentzVector* simPVPosition,
    const math::XYZPoint& bsPosition,
    const std::vector<float>& mvas,
    unsigned int selectsLoose,
    unsigned int selectsHP) const {
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
  if (isMatched) {
    for (size_t i = 0; i < mvas.size(); ++i) {
      if (i <= selectsLoose) {
        histograms.h_simul2_mvacut[count][i]->Fill(maxMVA);
        histograms.h_assoc_mvacut[count][i]->Fill(mvas[i]);
      }
      if (i >= 1 && i <= selectsHP) {
        histograms.h_simul2_mvacut_hp[count][i]->Fill(maxMVA);
        histograms.h_assoc_mvacut_hp[count][i]->Fill(mvas[i]);
      }
    }
  }

  if ((*TpSelectorForEfficiencyVsEta)(tp)) {
    //effic vs eta
    histograms.h_simuleta[count]->Fill(eta);
    if (isMatched)
      histograms.h_assoceta[count]->Fill(eta);
  }

  if ((*TpSelectorForEfficiencyVsPhi)(tp)) {
    histograms.h_simulphi[count]->Fill(phi);
    if (isMatched)
      histograms.h_assocphi[count]->Fill(phi);
    //effic vs hits
    histograms.h_simulhit[count]->Fill(nSimHits);
    histograms.h_simullayer[count]->Fill(nSimLayers);
    histograms.h_simulpixellayer[count]->Fill(nSimPixelLayers);
    histograms.h_simul3Dlayer[count]->Fill(nSim3DLayers);
    if (isMatched) {
      histograms.h_assochit[count]->Fill(nSimHits);
      histograms.h_assoclayer[count]->Fill(nSimLayers);
      histograms.h_assocpixellayer[count]->Fill(nSimPixelLayers);
      histograms.h_assoc3Dlayer[count]->Fill(nSim3DLayers);
      if (histograms.nrecHit_vs_nsimHit_sim2rec[count])
        histograms.nrecHit_vs_nsimHit_sim2rec[count]->Fill(track->numberOfValidHits(), nSimHits);
    }
    //effic vs pu
    histograms.h_simulpu[count]->Fill(numVertices);
    if (isMatched)
      histograms.h_assocpu[count]->Fill(numVertices);
    //efficiency vs dR
    histograms.h_simuldr[count]->Fill(dR);
    if (isMatched)
      histograms.h_assocdr[count]->Fill(dR);
    //efficiency vs dR jet
    histograms.h_simuldrj[count]->Fill(dRJet);
    if (isMatched)
      histograms.h_assocdrj[count]->Fill(dRJet);
  }

  if ((*TpSelectorForEfficiencyVsPt)(tp)) {
    histograms.h_simulpT[count]->Fill(pt);
    if (isMatched)
      histograms.h_assocpT[count]->Fill(pt);
  }

  if ((*TpSelectorForEfficiencyVsVTXR)(tp)) {
    histograms.h_simuldxy[count]->Fill(dxySim);
    if (isMatched)
      histograms.h_assocdxy[count]->Fill(dxySim);
    if (pvPosition) {
      histograms.h_simuldxypv[count]->Fill(dxyPVSim);
      histograms.h_simuldxypvzoomed[count]->Fill(dxyPVSim);
      if (isMatched) {
        histograms.h_assocdxypv[count]->Fill(dxyPVSim);
        histograms.h_assocdxypvzoomed[count]->Fill(dxyPVSim);
      }
    }

    histograms.h_simulvertpos[count]->Fill(vertxy);
    if (isMatched)
      histograms.h_assocvertpos[count]->Fill(vertxy);
  }

  if ((*TpSelectorForEfficiencyVsVTXZ)(tp)) {
    histograms.h_simuldz[count]->Fill(dzSim);
    if (isMatched)
      histograms.h_assocdz[count]->Fill(dzSim);

    histograms.h_simulzpos[count]->Fill(vertz);
    if (isMatched)
      histograms.h_assoczpos[count]->Fill(vertz);

    if (pvPosition) {
      histograms.h_simuldzpv[count]->Fill(dzPVSim);
      histograms.h_simuldzpvzoomed[count]->Fill(dzPVSim);

      histograms.h_simul_dzpvcut[count]->Fill(0);
      histograms.h_simul_dzpvsigcut[count]->Fill(0);
      histograms.h_simul_dzpvcut_pt[count]->Fill(0, pt);
      histograms.h_simul_dzpvsigcut_pt[count]->Fill(0, pt);

      if (isMatched) {
        histograms.h_assocdzpv[count]->Fill(dzPVSim);
        histograms.h_assocdzpvzoomed[count]->Fill(dzPVSim);

        histograms.h_simul2_dzpvcut[count]->Fill(0);
        histograms.h_simul2_dzpvsigcut[count]->Fill(0);
        histograms.h_simul2_dzpvcut_pt[count]->Fill(0, pt);
        histograms.h_simul2_dzpvsigcut_pt[count]->Fill(0, pt);
        const double dzpvcut = std::abs(track->dz(*pvPosition));
        const double dzpvsigcut = dzpvcut / track->dzError();
        histograms.h_assoc_dzpvcut[count]->Fill(dzpvcut);
        histograms.h_assoc_dzpvsigcut[count]->Fill(dzpvsigcut);
        histograms.h_assoc_dzpvcut_pt[count]->Fill(dzpvcut, pt);
        histograms.h_assoc_dzpvsigcut_pt[count]->Fill(dzpvsigcut, pt);
      }
    }
    if (simPVPosition) {
      const auto simpvz = simPVPosition->z();
      histograms.h_simul_simpvz[count]->Fill(simpvz);
      if (isMatched) {
        histograms.h_assoc_simpvz[count]->Fill(simpvz);
      }
    }
  }
}

void MTVHistoProducerAlgoForTracker::fill_duplicate_histos(const Histograms& histograms,
                                                           int count,
                                                           const reco::Track& track1,
                                                           const reco::Track& track2) const {
  histograms.h_duplicates_oriAlgo_vs_oriAlgo[count]->Fill(track1.originalAlgo(), track2.originalAlgo());
}

void MTVHistoProducerAlgoForTracker::fill_simTrackBased_histos(const Histograms& histograms, int numSimTracks) const {
  histograms.h_tracksSIM->Fill(numSimTracks);
}

// dE/dx
void MTVHistoProducerAlgoForTracker::fill_dedx_recoTrack_histos(
    const Histograms& histograms,
    int count,
    const edm::RefToBase<reco::Track>& trackref,
    const std::vector<const edm::ValueMap<reco::DeDxData>*>& v_dEdx) const {
  for (unsigned int i = 0; i < v_dEdx.size(); i++) {
    const edm::ValueMap<reco::DeDxData>& dEdxTrack = *(v_dEdx[i]);
    const reco::DeDxData& dedx = dEdxTrack[trackref];
    histograms.h_dedx_estim[count][i]->Fill(dedx.dEdx());
    histograms.h_dedx_nom[count][i]->Fill(dedx.numberOfMeasurements());
    histograms.h_dedx_sat[count][i]->Fill(dedx.numberOfSaturatedMeasurements());
  }
}

void MTVHistoProducerAlgoForTracker::fill_generic_recoTrack_histos(const Histograms& histograms,
                                                                   int count,
                                                                   const reco::Track& track,
                                                                   const TrackerTopology& ttopo,
                                                                   const math::XYZPoint& bsPosition,
                                                                   const math::XYZPoint* pvPosition,
                                                                   const TrackingVertex::LorentzVector* simPVPosition,
                                                                   bool isMatched,
                                                                   bool isSigMatched,
                                                                   bool isChargeMatched,
                                                                   int numAssocRecoTracks,
                                                                   int numVertices,
                                                                   int nSimHits,
                                                                   double sharedFraction,
                                                                   double dR,
                                                                   double dRJet,
                                                                   const std::vector<float>& mvas,
                                                                   unsigned int selectsLoose,
                                                                   unsigned int selectsHP) const {
  //Fill track algo histogram
  histograms.h_algo[count]->Fill(track.algo());
  int sharedHits = sharedFraction * track.numberOfValidHits();

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
  const auto chi2 = track.normalizedChi2();
  const auto chi2prob = TMath::Prob(track.chi2(), (int)track.ndof());
  const bool fillSeedingLayerSets = !seedingLayerSetNames.empty();
  const unsigned int seedingLayerSetBin = fillSeedingLayerSets ? getSeedingLayerSetBin(track, ttopo) : 0;
  const auto simpvz = simPVPosition ? simPVPosition->z() : 0.0;

  const bool paramsValid = !trackFromSeedFitFailed(track);

  if (paramsValid) {
    histograms.h_recoeta[count]->Fill(eta);
    histograms.h_recophi[count]->Fill(phi);
    histograms.h_recopT[count]->Fill(pt);
    histograms.h_recopTvseta[count]->Fill(eta, pt);
    histograms.h_recodxy[count]->Fill(dxy);
    histograms.h_recodz[count]->Fill(dz);
    histograms.h_recochi2[count]->Fill(chi2);
    histograms.h_recochi2prob[count]->Fill(chi2prob);
    histograms.h_recovertpos[count]->Fill(vertxy);
    histograms.h_recozpos[count]->Fill(vertz);
    histograms.h_recodr[count]->Fill(dR);
    histograms.h_recodrj[count]->Fill(dRJet);
    if (fillSeedingLayerSets)
      histograms.h_reco_seedingLayerSet[count]->Fill(seedingLayerSetBin);
    if (pvPosition) {
      histograms.h_recodxypv[count]->Fill(dxypv);
      histograms.h_recodzpv[count]->Fill(dzpv);
      histograms.h_recodxypvzoomed[count]->Fill(dxypv);
      histograms.h_recodzpvzoomed[count]->Fill(dzpv);

      histograms.h_reco_dzpvcut[count]->Fill(std::abs(dzpv));
      histograms.h_reco_dzpvsigcut[count]->Fill(std::abs(dzpvsig));
      histograms.h_reco_dzpvcut_pt[count]->Fill(std::abs(dzpv), pt);
      histograms.h_reco_dzpvsigcut_pt[count]->Fill(std::abs(dzpvsig), pt);
    }
    if (simPVPosition) {
      histograms.h_reco_simpvz[count]->Fill(simpvz);
    }
    if ((*trackSelectorVsEta)(track, bsPosition)) {
      histograms.h_reco2eta[count]->Fill(eta);
    }
    if ((*trackSelectorVsPt)(track, bsPosition)) {
      histograms.h_reco2pT[count]->Fill(pt);
      histograms.h_reco2pTvseta[count]->Fill(eta, pt);
    }
  }
  histograms.h_recohit[count]->Fill(nhits);
  histograms.h_recolayer[count]->Fill(nlayers);
  histograms.h_recopixellayer[count]->Fill(nPixelLayers);
  histograms.h_reco3Dlayer[count]->Fill(n3DLayers);
  histograms.h_recopu[count]->Fill(numVertices);
  if ((*trackSelectorVsPhi)(track, bsPosition)) {
    histograms.h_reco2pu[count]->Fill(numVertices);
  }

  fillMVAHistos(histograms.h_reco_mva[count],
                histograms.h_reco_mvacut[count],
                histograms.h_reco_mva_hp[count],
                histograms.h_reco_mvacut_hp[count],
                mvas,
                selectsLoose,
                selectsHP);

  if (isMatched) {
    if (paramsValid) {
      histograms.h_assoc2eta[count]->Fill(eta);
      histograms.h_assoc2phi[count]->Fill(phi);
      histograms.h_assoc2pT[count]->Fill(pt);
      histograms.h_assoc2pTvseta[count]->Fill(eta, pt);
      histograms.h_assoc2dxy[count]->Fill(dxy);
      histograms.h_assoc2dz[count]->Fill(dz);
      histograms.h_assoc2hit[count]->Fill(nhits);
      histograms.h_assoc2chi2[count]->Fill(chi2);
      histograms.h_assoc2chi2prob[count]->Fill(chi2prob);
      histograms.assoc_chi2_vs_eta[count]->Fill(eta, chi2);
      histograms.assoc_chi2prob_vs_eta[count]->Fill(eta, chi2prob);
      histograms.assoc_chi2_vs_pt[count]->Fill(pt, chi2);
      histograms.assoc_chi2prob_vs_pt[count]->Fill(pt, chi2prob);
      histograms.h_assoc2vertpos[count]->Fill(vertxy);
      histograms.h_assoc2zpos[count]->Fill(vertz);
      histograms.h_assoc2dr[count]->Fill(dR);
      histograms.h_assoc2drj[count]->Fill(dRJet);
      if (fillSeedingLayerSets)
        histograms.h_assoc2_seedingLayerSet[count]->Fill(seedingLayerSetBin);
      if (pvPosition) {
        histograms.h_assoc2dxypv[count]->Fill(dxypv);
        histograms.h_assoc2dzpv[count]->Fill(dzpv);
        histograms.h_assoc2dxypvzoomed[count]->Fill(dxypv);
        histograms.h_assoc2dzpvzoomed[count]->Fill(dzpv);

        histograms.h_assoc2_dzpvcut[count]->Fill(std::abs(dzpv));
        histograms.h_assoc2_dzpvsigcut[count]->Fill(std::abs(dzpvsig));
        histograms.h_assoc2_dzpvcut_pt[count]->Fill(std::abs(dzpv), pt);
        histograms.h_assoc2_dzpvsigcut_pt[count]->Fill(std::abs(dzpvsig), pt);
      }
      if (simPVPosition) {
        histograms.h_assoc2_simpvz[count]->Fill(simpvz);
      }
    }
    histograms.h_assoc2layer[count]->Fill(nlayers);
    histograms.h_assoc2pixellayer[count]->Fill(nPixelLayers);
    histograms.h_assoc23Dlayer[count]->Fill(n3DLayers);
    histograms.h_assoc2pu[count]->Fill(numVertices);

    fillMVAHistos(histograms.h_assoc2_mva[count],
                  histograms.h_assoc2_mvacut[count],
                  histograms.h_assoc2_mva_hp[count],
                  histograms.h_assoc2_mvacut_hp[count],
                  mvas,
                  selectsLoose,
                  selectsHP);
    fillMVAHistos(pt,
                  histograms.h_assoc2_mva_vs_pt[count],
                  histograms.h_assoc2_mva_vs_pt_hp[count],
                  mvas,
                  selectsLoose,
                  selectsHP);
    fillMVAHistos(eta,
                  histograms.h_assoc2_mva_vs_eta[count],
                  histograms.h_assoc2_mva_vs_eta_hp[count],
                  mvas,
                  selectsLoose,
                  selectsHP);

    if (histograms.nrecHit_vs_nsimHit_rec2sim[count])
      histograms.nrecHit_vs_nsimHit_rec2sim[count]->Fill(track.numberOfValidHits(), nSimHits);
    histograms.h_assocFraction[count]->Fill(sharedFraction);
    histograms.h_assocSharedHit[count]->Fill(sharedHits);

    if (!doSeedPlots_ && !isChargeMatched) {
      histograms.h_misideta[count]->Fill(eta);
      histograms.h_misidphi[count]->Fill(phi);
      histograms.h_misidpT[count]->Fill(pt);
      histograms.h_misidpTvseta[count]->Fill(eta, pt);
      histograms.h_misiddxy[count]->Fill(dxy);
      histograms.h_misiddz[count]->Fill(dz);
      histograms.h_misidhit[count]->Fill(nhits);
      histograms.h_misidlayer[count]->Fill(nlayers);
      histograms.h_misidpixellayer[count]->Fill(nPixelLayers);
      histograms.h_misid3Dlayer[count]->Fill(n3DLayers);
      histograms.h_misidpu[count]->Fill(numVertices);
      histograms.h_misidchi2[count]->Fill(chi2);
      histograms.h_misidchi2prob[count]->Fill(chi2prob);
      if (pvPosition) {
        histograms.h_misiddxypv[count]->Fill(dxypv);
        histograms.h_misiddzpv[count]->Fill(dzpv);
        histograms.h_misiddxypvzoomed[count]->Fill(dxypv);
        histograms.h_misiddzpvzoomed[count]->Fill(dzpv);
      }
    }

    if (numAssocRecoTracks > 1) {
      if (paramsValid) {
        histograms.h_loopereta[count]->Fill(eta);
        histograms.h_looperphi[count]->Fill(phi);
        histograms.h_looperpT[count]->Fill(pt);
        histograms.h_looperpTvseta[count]->Fill(eta, pt);
        histograms.h_looperdxy[count]->Fill(dxy);
        histograms.h_looperdz[count]->Fill(dz);
        histograms.h_looperchi2[count]->Fill(chi2);
        histograms.h_looperchi2prob[count]->Fill(chi2prob);
        histograms.h_loopervertpos[count]->Fill(vertxy);
        histograms.h_looperzpos[count]->Fill(vertz);
        histograms.h_looperdr[count]->Fill(dR);
        histograms.h_looperdrj[count]->Fill(dRJet);
        if (fillSeedingLayerSets)
          histograms.h_looper_seedingLayerSet[count]->Fill(seedingLayerSetBin);
        if (pvPosition) {
          histograms.h_looperdxypv[count]->Fill(dxypv);
          histograms.h_looperdzpv[count]->Fill(dzpv);
          histograms.h_looperdxypvzoomed[count]->Fill(dxypv);
          histograms.h_looperdzpvzoomed[count]->Fill(dzpv);
        }
        if (simPVPosition) {
          histograms.h_looper_simpvz[count]->Fill(simpvz);
        }
      }
      histograms.h_looperhit[count]->Fill(nhits);
      histograms.h_looperlayer[count]->Fill(nlayers);
      histograms.h_looperpixellayer[count]->Fill(nPixelLayers);
      histograms.h_looper3Dlayer[count]->Fill(n3DLayers);
      histograms.h_looperpu[count]->Fill(numVertices);
    }
    if (!isSigMatched) {
      if (paramsValid) {
        histograms.h_pileupeta[count]->Fill(eta);
        histograms.h_pileupphi[count]->Fill(phi);
        histograms.h_pileuppT[count]->Fill(pt);
        histograms.h_pileuppTvseta[count]->Fill(eta, pt);
        histograms.h_pileupdxy[count]->Fill(dxy);
        histograms.h_pileupdz[count]->Fill(dz);
        histograms.h_pileupchi2[count]->Fill(chi2);
        histograms.h_pileupchi2prob[count]->Fill(chi2prob);
        histograms.h_pileupvertpos[count]->Fill(vertxy);
        histograms.h_pileupzpos[count]->Fill(vertz);
        histograms.h_pileupdr[count]->Fill(dR);
        histograms.h_pileupdrj[count]->Fill(dRJet);
        if (fillSeedingLayerSets)
          histograms.h_pileup_seedingLayerSet[count]->Fill(seedingLayerSetBin);
        if (pvPosition) {
          histograms.h_pileupdxypv[count]->Fill(dxypv);
          histograms.h_pileupdzpv[count]->Fill(dzpv);
          histograms.h_pileupdxypvzoomed[count]->Fill(dxypv);
          histograms.h_pileupdzpvzoomed[count]->Fill(dzpv);

          histograms.h_pileup_dzpvcut[count]->Fill(std::abs(dzpv));
          histograms.h_pileup_dzpvsigcut[count]->Fill(std::abs(dzpvsig));
          histograms.h_pileup_dzpvcut_pt[count]->Fill(std::abs(dzpv), pt);
          histograms.h_pileup_dzpvsigcut_pt[count]->Fill(std::abs(dzpvsig), pt);
        }
        if (simPVPosition) {
          histograms.h_pileup_simpvz[count]->Fill(simpvz);
        }
      }
      histograms.h_pileuphit[count]->Fill(nhits);
      histograms.h_pileuplayer[count]->Fill(nlayers);
      histograms.h_pileuppixellayer[count]->Fill(nPixelLayers);
      histograms.h_pileup3Dlayer[count]->Fill(n3DLayers);
      histograms.h_pileuppu[count]->Fill(numVertices);
    }
  } else {  // !isMatched
    fillMVAHistos(
        pt, histograms.h_fake_mva_vs_pt[count], histograms.h_fake_mva_vs_pt_hp[count], mvas, selectsLoose, selectsHP);
    fillMVAHistos(
        eta, histograms.h_fake_mva_vs_eta[count], histograms.h_fake_mva_vs_eta_hp[count], mvas, selectsLoose, selectsHP);
  }
}

void MTVHistoProducerAlgoForTracker::fill_simAssociated_recoTrack_histos(const Histograms& histograms,
                                                                         int count,
                                                                         const reco::Track& track) const {
  //nchi2 and hits global distributions
  histograms.h_hits[count]->Fill(track.numberOfValidHits());
  histograms.h_losthits[count]->Fill(track.numberOfLostHits());
  histograms.h_nmisslayers_inner[count]->Fill(
      track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS));
  histograms.h_nmisslayers_outer[count]->Fill(
      track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS));
  if (trackFromSeedFitFailed(track))
    return;

  histograms.h_nchi2[count]->Fill(track.normalizedChi2());
  histograms.h_nchi2_prob[count]->Fill(TMath::Prob(track.chi2(), (int)track.ndof()));
  histograms.chi2_vs_nhits[count]->Fill(track.numberOfValidHits(), track.normalizedChi2());
  histograms.h_charge[count]->Fill(track.charge());

  //chi2 and #hit vs eta: fill 2D histos
  const auto eta = getEta(track.eta());
  histograms.chi2_vs_eta[count]->Fill(eta, track.normalizedChi2());
  histograms.nhits_vs_eta[count]->Fill(eta, track.numberOfValidHits());
  const auto pt = getPt(sqrt(track.momentum().perp2()));
  histograms.chi2_vs_pt[count]->Fill(pt, track.normalizedChi2());
  const auto pxbHits = track.hitPattern().numberOfValidPixelBarrelHits();
  const auto pxfHits = track.hitPattern().numberOfValidPixelEndcapHits();
  const auto tibHits = track.hitPattern().numberOfValidStripTIBHits();
  const auto tidHits = track.hitPattern().numberOfValidStripTIDHits();
  const auto tobHits = track.hitPattern().numberOfValidStripTOBHits();
  const auto tecHits = track.hitPattern().numberOfValidStripTECHits();
  histograms.nPXBhits_vs_eta[count]->Fill(eta, pxbHits);
  histograms.nPXFhits_vs_eta[count]->Fill(eta, pxfHits);
  histograms.nPXLhits_vs_eta[count]->Fill(eta, pxbHits + pxfHits);
  histograms.nTIBhits_vs_eta[count]->Fill(eta, tibHits);
  histograms.nTIDhits_vs_eta[count]->Fill(eta, tidHits);
  histograms.nTOBhits_vs_eta[count]->Fill(eta, tobHits);
  histograms.nTEChits_vs_eta[count]->Fill(eta, tecHits);
  histograms.nSTRIPhits_vs_eta[count]->Fill(eta, tibHits + tidHits + tobHits + tecHits);
  histograms.nLayersWithMeas_vs_eta[count]->Fill(eta, track.hitPattern().trackerLayersWithMeasurement());
  histograms.nPXLlayersWithMeas_vs_eta[count]->Fill(eta, track.hitPattern().pixelLayersWithMeasurement());
  if (doMTDPlots_) {
    //  const auto mtdHits = track.hitPattern().numberOfValidTimingHits();
    const auto btlHits = track.hitPattern().numberOfValidTimingBTLHits();
    const auto etlHits = track.hitPattern().numberOfValidTimingETLHits();
    histograms.nMTDhits_vs_eta[count]->Fill(eta, btlHits + etlHits);
    histograms.nBTLhits_vs_eta[count]->Fill(eta, btlHits);
    histograms.nETLhits_vs_eta[count]->Fill(eta, etlHits);
  }
  int LayersAll = track.hitPattern().stripLayersWithMeasurement();
  int Layers2D = track.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
  int Layers1D = LayersAll - Layers2D;
  histograms.nSTRIPlayersWithMeas_vs_eta[count]->Fill(eta, LayersAll);
  histograms.nSTRIPlayersWith1dMeas_vs_eta[count]->Fill(eta, Layers1D);
  histograms.nSTRIPlayersWith2dMeas_vs_eta[count]->Fill(eta, Layers2D);

  histograms.nlosthits_vs_eta[count]->Fill(eta, track.numberOfLostHits());
}

void MTVHistoProducerAlgoForTracker::fill_trackBased_histos(const Histograms& histograms,
                                                            int count,
                                                            int assTracks,
                                                            int numRecoTracks,
                                                            int numRecoTracksSelected,
                                                            int numSimTracksSelected) const {
  histograms.h_tracks[count]->Fill(assTracks);
  histograms.h_fakes[count]->Fill(numRecoTracks - assTracks);
  if (histograms.nrec_vs_nsim[count])
    histograms.nrec_vs_nsim[count]->Fill(numSimTracksSelected, numRecoTracksSelected);
}

void MTVHistoProducerAlgoForTracker::fill_ResoAndPull_recoTrack_histos(const Histograms& histograms,
                                                                       int count,
                                                                       const TrackingParticle::Vector& momentumTP,
                                                                       const TrackingParticle::Point& vertexTP,
                                                                       int chargeTP,
                                                                       const reco::Track& track,
                                                                       const math::XYZPoint& bsPosition) const {
  if (trackFromSeedFitFailed(track))
    return;

  // evaluation of TP parameters
  double qoverpSim = chargeTP / sqrt(momentumTP.x() * momentumTP.x() + momentumTP.y() * momentumTP.y() +
                                     momentumTP.z() * momentumTP.z());
  double lambdaSim = M_PI / 2 - momentumTP.theta();
  double phiSim = momentumTP.phi();
  double dxySim = TrackingParticleIP::dxy(vertexTP, momentumTP, bsPosition);
  double dzSim = TrackingParticleIP::dz(vertexTP, momentumTP, bsPosition);

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
  getRecoMomentum(track, ptRec, ptErrorRec, qoverpRec, qoverpErrorRec, lambdaRec, lambdaErrorRec, phiRec, phiErrorRec);
  // -------------

  double ptError = ptErrorRec;
  double ptres = ptRec - sqrt(momentumTP.perp2());
  double etares = track.eta() - momentumTP.Eta();

  double dxyRec = track.dxy(bsPosition);
  double dzRec = track.dz(bsPosition);

  const auto phiRes = phiRec - phiSim;
  const auto dxyRes = dxyRec - dxySim;
  const auto dzRes = dzRec - dzSim;
  const auto cotThetaRes = 1 / tan(M_PI * 0.5 - lambdaRec) - 1 / tan(M_PI * 0.5 - lambdaSim);

  // eta residue; pt, k, theta, phi, dxy, dz pulls
  double qoverpPull = (qoverpRec - qoverpSim) / qoverpErrorRec;
  double thetaPull = (lambdaRec - lambdaSim) / lambdaErrorRec;
  double phiPull = phiRes / phiErrorRec;
  double dxyPull = dxyRes / track.dxyError();
  double dzPull = dzRes / track.dzError();

#ifdef EDM_ML_DEBUG
  double contrib_Qoverp = ((qoverpRec - qoverpSim) / qoverpErrorRec) * ((qoverpRec - qoverpSim) / qoverpErrorRec) / 5;
  double contrib_dxy = ((dxyRec - dxySim) / track.dxyError()) * ((dxyRec - dxySim) / track.dxyError()) / 5;
  double contrib_dz = ((dzRec - dzSim) / track.dzError()) * ((dzRec - dzSim) / track.dzError()) / 5;
  double contrib_theta = ((lambdaRec - lambdaSim) / lambdaErrorRec) * ((lambdaRec - lambdaSim) / lambdaErrorRec) / 5;
  double contrib_phi = ((phiRec - phiSim) / phiErrorRec) * ((phiRec - phiSim) / phiErrorRec) / 5;

  LogTrace("TrackValidatorTEST")
      //<< "assocChi2=" << tp.begin()->second << "\n"
      << ""
      << "\n"
      << "ptREC=" << ptRec << "\n"
      << "etaREC=" << track.eta() << "\n"
      << "qoverpREC=" << qoverpRec << "\n"
      << "dxyREC=" << dxyRec << "\n"
      << "dzREC=" << dzRec << "\n"
      << "thetaREC=" << track.theta() << "\n"
      << "phiREC=" << phiRec << "\n"
      << ""
      << "\n"
      << "qoverpError()=" << qoverpErrorRec << "\n"
      << "dxyError()=" << track.dxyError() << "\n"
      << "dzError()=" << track.dzError() << "\n"
      << "thetaError()=" << lambdaErrorRec << "\n"
      << "phiError()=" << phiErrorRec << "\n"
      << ""
      << "\n"
      << "ptSIM=" << sqrt(momentumTP.perp2()) << "\n"
      << "etaSIM=" << momentumTP.Eta() << "\n"
      << "qoverpSIM=" << qoverpSim << "\n"
      << "dxySIM=" << dxySim << "\n"
      << "dzSIM=" << dzSim << "\n"
      << "thetaSIM=" << M_PI / 2 - lambdaSim << "\n"
      << "phiSIM=" << phiSim << "\n"
      << ""
      << "\n"
      << "contrib_Qoverp=" << contrib_Qoverp << "\n"
      << "contrib_dxy=" << contrib_dxy << "\n"
      << "contrib_dz=" << contrib_dz << "\n"
      << "contrib_theta=" << contrib_theta << "\n"
      << "contrib_phi=" << contrib_phi << "\n"
      << ""
      << "\n"
      << "chi2PULL=" << contrib_Qoverp + contrib_dxy + contrib_dz + contrib_theta + contrib_phi << "\n";
#endif

  histograms.h_pullQoverp[count]->Fill(qoverpPull);
  histograms.h_pullTheta[count]->Fill(thetaPull);
  histograms.h_pullPhi[count]->Fill(phiPull);
  histograms.h_pullDxy[count]->Fill(dxyPull);
  histograms.h_pullDz[count]->Fill(dzPull);

  const auto etaSim = getEta(momentumTP.eta());
  const auto ptSim = getPt(sqrt(momentumTP.perp2()));

  histograms.h_pt[count]->Fill(ptres / ptError);
  histograms.h_eta[count]->Fill(etares);
  //histograms.etares_vs_eta[count]->Fill(getEta(track.eta()),etares);
  histograms.etares_vs_eta[count]->Fill(etaSim, etares);

  //resolution of track params: fill 2D histos
  histograms.dxyres_vs_eta[count]->Fill(etaSim, dxyRes);
  histograms.ptres_vs_eta[count]->Fill(etaSim, ptres / ptRec);
  histograms.dzres_vs_eta[count]->Fill(etaSim, dzRes);
  histograms.phires_vs_eta[count]->Fill(etaSim, phiRes);
  histograms.cotThetares_vs_eta[count]->Fill(etaSim, cotThetaRes);

  //same as before but vs pT
  histograms.dxyres_vs_pt[count]->Fill(ptSim, dxyRes);
  histograms.ptres_vs_pt[count]->Fill(ptSim, ptres / ptRec);
  histograms.dzres_vs_pt[count]->Fill(ptSim, dzRes);
  histograms.phires_vs_pt[count]->Fill(ptSim, phiRes);
  histograms.cotThetares_vs_pt[count]->Fill(ptSim, cotThetaRes);

  //pulls of track params vs eta: fill 2D histos
  histograms.dxypull_vs_eta[count]->Fill(etaSim, dxyPull);
  histograms.ptpull_vs_eta[count]->Fill(etaSim, ptres / ptError);
  histograms.dzpull_vs_eta[count]->Fill(etaSim, dzPull);
  histograms.phipull_vs_eta[count]->Fill(etaSim, phiPull);
  histograms.thetapull_vs_eta[count]->Fill(etaSim, thetaPull);

  //plots vs phi
  histograms.nhits_vs_phi[count]->Fill(phiRec, track.numberOfValidHits());
  histograms.chi2_vs_phi[count]->Fill(phiRec, track.normalizedChi2());
  histograms.ptmean_vs_eta_phi[count]->Fill(phiRec, getEta(track.eta()), ptRec);
  histograms.phimean_vs_eta_phi[count]->Fill(phiRec, getEta(track.eta()), phiRec);

  histograms.ptres_vs_phi[count]->Fill(phiSim, ptres / ptRec);
  histograms.phires_vs_phi[count]->Fill(phiSim, phiRes);
  histograms.ptpull_vs_phi[count]->Fill(phiSim, ptres / ptError);
  histograms.phipull_vs_phi[count]->Fill(phiSim, phiPull);
  histograms.thetapull_vs_phi[count]->Fill(phiSim, thetaPull);
}

void MTVHistoProducerAlgoForTracker::getRecoMomentum(const reco::Track& track,
                                                     double& pt,
                                                     double& ptError,
                                                     double& qoverp,
                                                     double& qoverpError,
                                                     double& lambda,
                                                     double& lambdaError,
                                                     double& phi,
                                                     double& phiError) const {
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

void MTVHistoProducerAlgoForTracker::getRecoMomentum(const reco::GsfTrack& gsfTrack,
                                                     double& pt,
                                                     double& ptError,
                                                     double& qoverp,
                                                     double& qoverpError,
                                                     double& lambda,
                                                     double& lambdaError,
                                                     double& phi,
                                                     double& phiError) const {
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

double MTVHistoProducerAlgoForTracker::getEta(double eta) const {
  if (useFabsEta)
    return fabs(eta);
  else
    return eta;
}

double MTVHistoProducerAlgoForTracker::getPt(double pt) const {
  if (useInvPt && pt != 0)
    return 1 / pt;
  else
    return pt;
}

unsigned int MTVHistoProducerAlgoForTracker::getSeedingLayerSetBin(const reco::Track& track,
                                                                   const TrackerTopology& ttopo) const {
  if (track.seedRef().isNull() || !track.seedRef().isAvailable())
    return seedingLayerSetNames.size() - 1;

  const TrajectorySeed& seed = *(track.seedRef());
  const auto hitRange = seed.recHits();
  SeedingLayerSetId searchId;
  const int nhits = std::distance(hitRange.first, hitRange.second);
  if (nhits > static_cast<int>(std::tuple_size<SeedingLayerSetId>::value)) {
    LogDebug("TrackValidator") << "Got seed with " << nhits << " hits, but I have a hard-coded maximum of "
                               << std::tuple_size<SeedingLayerSetId>::value
                               << ", classifying the seed as 'unknown'. Please increase the maximum in "
                                  "MTVHistoProducerAlgoForTracker.h if needed.";
    return seedingLayerSetNames.size() - 1;
  }
  int i = 0;
  for (auto iHit = hitRange.first; iHit != hitRange.second; ++iHit, ++i) {
    DetId detId = iHit->geographicalId();

    if (detId.det() != DetId::Tracker) {
      throw cms::Exception("LogicError") << "Encountered seed hit detId " << detId.rawId() << " not from Tracker, but "
                                         << detId.det();
    }

    GeomDetEnumerators::SubDetector subdet;
    bool subdetStrip = false;
    switch (detId.subdetId()) {
      case PixelSubdetector::PixelBarrel:
        subdet = GeomDetEnumerators::PixelBarrel;
        break;
      case PixelSubdetector::PixelEndcap:
        subdet = GeomDetEnumerators::PixelEndcap;
        break;
      case StripSubdetector::TIB:
        subdet = GeomDetEnumerators::TIB;
        subdetStrip = true;
        break;
      case StripSubdetector::TID:
        subdet = GeomDetEnumerators::TID;
        subdetStrip = true;
        break;
      case StripSubdetector::TOB:
        subdet = GeomDetEnumerators::TOB;
        subdetStrip = true;
        break;
      case StripSubdetector::TEC:
        subdet = GeomDetEnumerators::TEC;
        subdetStrip = true;
        break;
      default:
        throw cms::Exception("LogicError") << "Unknown subdetId " << detId.subdetId();
    };

    TrackerDetSide side = static_cast<TrackerDetSide>(ttopo.side(detId));

    // Even with the recent addition of
    // SeedingLayerSetsBuilder::fillDescription() this assumption is a
    // bit ugly.
    const bool isStripMono = subdetStrip && trackerHitRTTI::isSingle(*iHit);
    searchId[i] =
        SeedingLayerId(SeedingLayerSetsBuilder::SeedingLayerId(subdet, side, ttopo.layer(detId)), isStripMono);
  }
  auto found = seedingLayerSetToBin.find(searchId);
  if (found == seedingLayerSetToBin.end()) {
    return seedingLayerSetNames.size() - 1;
  }
  return found->second;
}

void MTVHistoProducerAlgoForTracker::fill_recoAssociated_simTrack_histos(const Histograms& histograms,
                                                                         int count,
                                                                         const reco::GenParticle& tp,
                                                                         const TrackingParticle::Vector& momentumTP,
                                                                         const TrackingParticle::Point& vertexTP,
                                                                         double dxySim,
                                                                         double dzSim,
                                                                         int nSimHits,
                                                                         const reco::Track* track,
                                                                         int numVertices) const {
  bool isMatched = track;

  if ((*GpSelectorForEfficiencyVsEta)(tp)) {
    //effic vs eta
    histograms.h_simuleta[count]->Fill(getEta(momentumTP.eta()));
    if (isMatched)
      histograms.h_assoceta[count]->Fill(getEta(momentumTP.eta()));
  }

  if ((*GpSelectorForEfficiencyVsPhi)(tp)) {
    histograms.h_simulphi[count]->Fill(momentumTP.phi());
    if (isMatched)
      histograms.h_assocphi[count]->Fill(momentumTP.phi());
    //effic vs hits
    histograms.h_simulhit[count]->Fill((int)nSimHits);
    if (isMatched) {
      histograms.h_assochit[count]->Fill((int)nSimHits);
      if (histograms.nrecHit_vs_nsimHit_sim2rec[count])
        histograms.nrecHit_vs_nsimHit_sim2rec[count]->Fill(track->numberOfValidHits(), nSimHits);
    }
    //effic vs pu
    histograms.h_simulpu[count]->Fill(numVertices);
    if (isMatched)
      histograms.h_assocpu[count]->Fill(numVertices);
    //efficiency vs dR
    //not implemented for now
  }

  if ((*GpSelectorForEfficiencyVsPt)(tp)) {
    histograms.h_simulpT[count]->Fill(getPt(sqrt(momentumTP.perp2())));
    histograms.h_simulpTvseta[count]->Fill(getEta(momentumTP.eta()), getPt(sqrt(momentumTP.perp2())));
    if (isMatched) {
      histograms.h_assocpT[count]->Fill(getPt(sqrt(momentumTP.perp2())));
      histograms.h_assocpTvseta[count]->Fill(getEta(momentumTP.eta()), getPt(sqrt(momentumTP.perp2())));
    }
  }

  if ((*GpSelectorForEfficiencyVsVTXR)(tp)) {
    histograms.h_simuldxy[count]->Fill(dxySim);
    if (isMatched)
      histograms.h_assocdxy[count]->Fill(dxySim);

    histograms.h_simulvertpos[count]->Fill(sqrt(vertexTP.perp2()));
    if (isMatched)
      histograms.h_assocvertpos[count]->Fill(sqrt(vertexTP.perp2()));
  }

  if ((*GpSelectorForEfficiencyVsVTXZ)(tp)) {
    histograms.h_simuldz[count]->Fill(dzSim);
    if (isMatched)
      histograms.h_assocdz[count]->Fill(dzSim);

    histograms.h_simulzpos[count]->Fill(vertexTP.z());
    if (isMatched)
      histograms.h_assoczpos[count]->Fill(vertexTP.z());
  }
}

void MTVHistoProducerAlgoForTracker::fill_seed_histos(const Histograms& histograms,
                                                      int count,
                                                      int seedsFitFailed,
                                                      int seedsTotal) const {
  histograms.h_seedsFitFailed[count]->Fill(seedsFitFailed);
  histograms.h_seedsFitFailedFraction[count]->Fill(static_cast<double>(seedsFitFailed) / seedsTotal);
}
