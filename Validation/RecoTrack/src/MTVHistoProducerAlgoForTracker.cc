#include "Validation/RecoTrack/interface/MTVHistoProducerAlgoForTracker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ref.h"
#include <DataFormats/TrackReco/interface/TrackFwd.h>

#include "DQMServices/ClientConfig/interface/FitSlicesYTool.h"

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

  template<typename T> void fillPlotNoFlow(MonitorElement *h, T val) {
    h->Fill(std::min(std::max(val,((T) h->getTH1()->GetXaxis()->GetXmin())),((T) h->getTH1()->GetXaxis()->GetXmax())));
  }
}

MTVHistoProducerAlgoForTracker::MTVHistoProducerAlgoForTracker(const edm::ParameterSet& pset, edm::ConsumesCollector & iC):
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

  //parameters for _vs_ProductionVertexTransvPosition plots
  minVertpos  = pset.getParameter<double>("minVertpos");
  maxVertpos  = pset.getParameter<double>("maxVertpos");
  nintVertpos = pset.getParameter<int>("nintVertpos");

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
  auto initGPselector = [&](auto& sel, auto& name) {
    sel = std::make_unique<GenParticleCustomSelector>(ParameterAdapter<GenParticleCustomSelector>::make(pset.getParameter<ParameterSet>(name), iC));
  };

  initTPselector(generalTpSelector,             "generalTpSelector");
  initTPselector(TpSelectorForEfficiencyVsEta,  "TpSelectorForEfficiencyVsEta");
  initTPselector(TpSelectorForEfficiencyVsPhi,  "TpSelectorForEfficiencyVsPhi");
  initTPselector(TpSelectorForEfficiencyVsPt,   "TpSelectorForEfficiencyVsPt");
  initTPselector(TpSelectorForEfficiencyVsVTXR, "TpSelectorForEfficiencyVsVTXR");
  initTPselector(TpSelectorForEfficiencyVsVTXZ, "TpSelectorForEfficiencyVsVTXZ");

  initGPselector(generalGpSelector,             "generalGpSelector");
  initGPselector(GpSelectorForEfficiencyVsEta,  "GpSelectorForEfficiencyVsEta");
  initGPselector(GpSelectorForEfficiencyVsPhi,  "GpSelectorForEfficiencyVsPhi");
  initGPselector(GpSelectorForEfficiencyVsPt,   "GpSelectorForEfficiencyVsPt");
  initGPselector(GpSelectorForEfficiencyVsVTXR, "GpSelectorForEfficiencyVsVTXR");
  initGPselector(GpSelectorForEfficiencyVsVTXZ, "GpSelectorForEfficiencyVsVTXZ");

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

}

MTVHistoProducerAlgoForTracker::~MTVHistoProducerAlgoForTracker() {}

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

void MTVHistoProducerAlgoForTracker::bookSimTrackHistos(DQMStore::IBooker& ibook){
  h_assoceta.push_back( ibook.book1D("num_assoc(simToReco)_eta","N of associated tracks (simToReco) vs eta",nintEta,minEta,maxEta) );
  h_simuleta.push_back( ibook.book1D("num_simul_eta","N of simulated tracks vs eta",nintEta,minEta,maxEta) );

  h_assocpT.push_back( ibook.book1D("num_assoc(simToReco)_pT","N of associated tracks (simToReco) vs pT",nintPt,minPt,maxPt) );
  h_simulpT.push_back( ibook.book1D("num_simul_pT","N of simulated tracks vs pT",nintPt,minPt,maxPt) );

  h_assochit.push_back( ibook.book1D("num_assoc(simToReco)_hit","N of associated tracks (simToReco) vs hit",nintHit,minHit,maxHit) );
  h_simulhit.push_back( ibook.book1D("num_simul_hit","N of simulated tracks vs hit",nintHit,minHit,maxHit) );

  h_assoclayer.push_back( ibook.book1D("num_assoc(simToReco)_layer","N of associated tracks (simToReco) vs layer",nintHit,minHit,maxHit) );
  h_simullayer.push_back( ibook.book1D("num_simul_layer","N of simulated tracks vs layer",nintHit,minHit,maxHit) );

  h_assocpixellayer.push_back( ibook.book1D("num_assoc(simToReco)_pixellayer","N of associated tracks (simToReco) vs pixel layer",nintHit,minHit,maxHit) );
  h_simulpixellayer.push_back( ibook.book1D("num_simul_pixellayer","N of simulated tracks vs pixel layer",nintHit,minHit,maxHit) );

  h_assoc3Dlayer.push_back( ibook.book1D("num_assoc(simToReco)_3Dlayer","N of associated tracks (simToReco) vs 3D layer",nintHit,minHit,maxHit) );
  h_simul3Dlayer.push_back( ibook.book1D("num_simul_3Dlayer","N of simulated tracks vs 3D layer",nintHit,minHit,maxHit) );

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

  nrecHit_vs_nsimHit_sim2rec.push_back( ibook.book2D("nrecHit_vs_nsimHit_sim2rec","nrecHit vs nsimHit (Sim2RecAssoc)",
						     nintHit,minHit,maxHit, nintHit,minHit,maxHit ));

  if(useLogPt){
    BinLogX(h_assocpT.back()->getTH1F());
    BinLogX(h_simulpT.back()->getTH1F());
  }
}

void MTVHistoProducerAlgoForTracker::bookSimTrackPVAssociationHistos(DQMStore::IBooker& ibook) {
  h_assocdxypv.push_back( ibook.book1D("num_assoc(simToReco)_dxypv","N of associated tracks (simToReco) vs dxy(PV)",nintDxy,minDxy,maxDxy) );
  h_simuldxypv.push_back( ibook.book1D("num_simul_dxypv","N of simulated tracks vs dxy(PV)",nintDxy,minDxy,maxDxy) );

  h_assocdzpv.push_back( ibook.book1D("num_assoc(simToReco)_dzpv","N of associated tracks (simToReco) vs dz(PV)",nintDz,minDz,maxDz) );
  h_simuldzpv.push_back( ibook.book1D("num_simul_dzpv","N of simulated tracks vs dz(PV)",nintDz,minDz,maxDz) );

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

void MTVHistoProducerAlgoForTracker::bookRecoHistos(DQMStore::IBooker& ibook){
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
  h_assoc2eta.push_back( ibook.book1D("num_assoc(recoToSim)_eta","N of associated (recoToSim) tracks vs eta",nintEta,minEta,maxEta) );
  h_loopereta.push_back( ibook.book1D("num_duplicate_eta","N of associated (recoToSim) duplicate tracks vs eta",nintEta,minEta,maxEta) );
  h_misideta.push_back( ibook.book1D("num_chargemisid_eta","N of associated (recoToSim) charge misIDed tracks vs eta",nintEta,minEta,maxEta) );
  h_pileupeta.push_back( ibook.book1D("num_pileup_eta","N of associated (recoToSim) pileup tracks vs eta",nintEta,minEta,maxEta) );
  //
  h_recopT.push_back( ibook.book1D("num_reco_pT","N of reco track vs pT",nintPt,minPt,maxPt) );
  h_assoc2pT.push_back( ibook.book1D("num_assoc(recoToSim)_pT","N of associated (recoToSim) tracks vs pT",nintPt,minPt,maxPt) );
  h_looperpT.push_back( ibook.book1D("num_duplicate_pT","N of associated (recoToSim) duplicate tracks vs pT",nintPt,minPt,maxPt) );
  h_misidpT.push_back( ibook.book1D("num_chargemisid_pT","N of associated (recoToSim) charge misIDed tracks vs pT",nintPt,minPt,maxPt) );
  h_pileuppT.push_back( ibook.book1D("num_pileup_pT","N of associated (recoToSim) pileup tracks vs pT",nintPt,minPt,maxPt) );
  //
  h_recohit.push_back( ibook.book1D("num_reco_hit","N of reco track vs hit",nintHit,minHit,maxHit) );
  h_assoc2hit.push_back( ibook.book1D("num_assoc(recoToSim)_hit","N of associated (recoToSim) tracks vs hit",nintHit,minHit,maxHit) );
  h_looperhit.push_back( ibook.book1D("num_duplicate_hit","N of associated (recoToSim) duplicate tracks vs hit",nintHit,minHit,maxHit) );
  h_misidhit.push_back( ibook.book1D("num_chargemisid_hit","N of associated (recoToSim) charge misIDed tracks vs hit",nintHit,minHit,maxHit) );
  h_pileuphit.push_back( ibook.book1D("num_pileup_hit","N of associated (recoToSim) pileup tracks vs hit",nintHit,minHit,maxHit) );
  //
  h_recolayer.push_back( ibook.book1D("num_reco_layer","N of reco track vs layer",nintHit,minHit,maxHit) );
  h_assoc2layer.push_back( ibook.book1D("num_assoc(recoToSim)_layer","N of associated (recoToSim) tracks vs layer",nintHit,minHit,maxHit) );
  h_looperlayer.push_back( ibook.book1D("num_duplicate_layer","N of associated (recoToSim) duplicate tracks vs layer",nintHit,minHit,maxHit) );
  h_misidlayer.push_back( ibook.book1D("num_chargemisid_layer","N of associated (recoToSim) charge misIDed tracks vs layer",nintHit,minHit,maxHit) );
  h_pileuplayer.push_back( ibook.book1D("num_pileup_layer","N of associated (recoToSim) pileup tracks vs layer",nintHit,minHit,maxHit) );
  //
  h_recopixellayer.push_back( ibook.book1D("num_reco_pixellayer","N of reco track vs pixellayer",nintHit,minHit,maxHit) );
  h_assoc2pixellayer.push_back( ibook.book1D("num_assoc(recoToSim)_pixellayer","N of associated (recoToSim) tracks vs pixellayer",nintHit,minHit,maxHit) );
  h_looperpixellayer.push_back( ibook.book1D("num_duplicate_pixellayer","N of associated (recoToSim) duplicate tracks vs pixellayer",nintHit,minHit,maxHit) );
  h_misidpixellayer.push_back( ibook.book1D("num_chargemisid_pixellayer","N of associated (recoToSim) charge misIDed tracks vs pixellayer",nintHit,minHit,maxHit) );
  h_pileuppixellayer.push_back( ibook.book1D("num_pileup_pixellayer","N of associated (recoToSim) pileup tracks vs pixellayer",nintHit,minHit,maxHit) );
  //
  h_reco3Dlayer.push_back( ibook.book1D("num_reco_3Dlayer","N of reco track vs 3D layer",nintHit,minHit,maxHit) );
  h_assoc23Dlayer.push_back( ibook.book1D("num_assoc(recoToSim)_3Dlayer","N of associated (recoToSim) tracks vs 3D layer",nintHit,minHit,maxHit) );
  h_looper3Dlayer.push_back( ibook.book1D("num_duplicate_3Dlayer","N of associated (recoToSim) duplicate tracks vs 3D layer",nintHit,minHit,maxHit) );
  h_misid3Dlayer.push_back( ibook.book1D("num_chargemisid_3Dlayer","N of associated (recoToSim) charge misIDed tracks vs 3D layer",nintHit,minHit,maxHit) );
  h_pileup3Dlayer.push_back( ibook.book1D("num_pileup_3Dlayer","N of associated (recoToSim) pileup tracks vs 3D layer",nintHit,minHit,maxHit) );
  //
  h_recopu.push_back( ibook.book1D("num_reco_pu","N of reco track vs pu",nintPu,minPu,maxPu) );
  h_assoc2pu.push_back( ibook.book1D("num_assoc(recoToSim)_pu","N of associated (recoToSim) tracks vs pu",nintPu,minPu,maxPu) );
  h_looperpu.push_back( ibook.book1D("num_duplicate_pu","N of associated (recoToSim) duplicate tracks vs pu",nintPu,minPu,maxPu) );
  h_misidpu.push_back( ibook.book1D("num_chargemisid_pu","N of associated (recoToSim) charge misIDed tracks vs pu",nintPu,minPu,maxPu) );
  h_pileuppu.push_back( ibook.book1D("num_pileup_pu","N of associated (recoToSim) pileup tracks vs pu",nintPu,minPu,maxPu) );
  //
  h_recophi.push_back( ibook.book1D("num_reco_phi","N of reco track vs phi",nintPhi,minPhi,maxPhi) );
  h_assoc2phi.push_back( ibook.book1D("num_assoc(recoToSim)_phi","N of associated (recoToSim) tracks vs phi",nintPhi,minPhi,maxPhi) );
  h_looperphi.push_back( ibook.book1D("num_duplicate_phi","N of associated (recoToSim) duplicate tracks vs phi",nintPhi,minPhi,maxPhi) );
  h_misidphi.push_back( ibook.book1D("num_chargemisid_phi","N of associated (recoToSim) charge misIDed tracks vs phi",nintPhi,minPhi,maxPhi) );
  h_pileupphi.push_back( ibook.book1D("num_pileup_phi","N of associated (recoToSim) pileup tracks vs phi",nintPhi,minPhi,maxPhi) );

  h_recodxy.push_back( ibook.book1D("num_reco_dxy","N of reco track vs dxy",nintDxy,minDxy,maxDxy) );
  h_assoc2dxy.push_back( ibook.book1D("num_assoc(recoToSim)_dxy","N of associated (recoToSim) tracks vs dxy",nintDxy,minDxy,maxDxy) );
  h_looperdxy.push_back( ibook.book1D("num_duplicate_dxy","N of associated (recoToSim) looper tracks vs dxy",nintDxy,minDxy,maxDxy) );
  h_misiddxy.push_back( ibook.book1D("num_chargemisid_dxy","N of associated (recoToSim) charge misIDed tracks vs dxy",nintDxy,minDxy,maxDxy) );
  h_pileupdxy.push_back( ibook.book1D("num_pileup_dxy","N of associated (recoToSim) pileup tracks vs dxy",nintDxy,minDxy,maxDxy) );

  h_recodz.push_back( ibook.book1D("num_reco_dz","N of reco track vs dz",nintDz,minDz,maxDz) );
  h_assoc2dz.push_back( ibook.book1D("num_assoc(recoToSim)_dz","N of associated (recoToSim) tracks vs dz",nintDz,minDz,maxDz) );
  h_looperdz.push_back( ibook.book1D("num_duplicate_dz","N of associated (recoToSim) looper tracks vs dz",nintDz,minDz,maxDz) );
  h_misiddz.push_back( ibook.book1D("num_chargemisid_versus_dz","N of associated (recoToSim) charge misIDed tracks vs dz",nintDz,minDz,maxDz) );
  h_pileupdz.push_back( ibook.book1D("num_pileup_dz","N of associated (recoToSim) pileup tracks vs dz",nintDz,minDz,maxDz) );

  h_recodr.push_back( ibook.book1D("num_reco_dr","N of reconstructed tracks vs dR",nintdr,log10(mindr),log10(maxdr)) );
  h_assoc2dr.push_back( ibook.book1D("num_assoc(recoToSim)_dr","N of associated tracks (recoToSim) vs dR",nintdr,log10(mindr),log10(maxdr)) );
  h_pileupdr.push_back( ibook.book1D("num_pileup_dr","N of associated (recoToSim) pileup tracks vs dR",nintdr,log10(mindr),log10(maxdr)) );
  BinLogX(h_recodr.back()->getTH1F());
  BinLogX(h_assoc2dr.back()->getTH1F());
  BinLogX(h_pileupdr.back()->getTH1F());

  h_recochi2.push_back( ibook.book1D("num_reco_chi2","N of reco track vs normalized #chi^{2}",nintChi2,minChi2,maxChi2) );
  h_assoc2chi2.push_back( ibook.book1D("num_assoc(recoToSim)_chi2","N of associated (recoToSim) tracks vs normalized #chi^{2}",nintChi2,minChi2,maxChi2) );
  h_looperchi2.push_back( ibook.book1D("num_duplicate_chi2","N of associated (recoToSim) looper tracks vs normalized #chi^{2}",nintChi2,minChi2,maxChi2) );
  h_misidchi2.push_back( ibook.book1D("num_chargemisid_chi2","N of associated (recoToSim) charge misIDed tracks vs normalized #chi^{2}",nintChi2,minChi2,maxChi2) );
  h_pileupchi2.push_back( ibook.book1D("num_pileup_chi2","N of associated (recoToSim) pileup tracks vs normalized #chi^{2}",nintChi2,minChi2,maxChi2) );

  /////////////////////////////////

  h_eta.push_back( ibook.book1D("eta", "pseudorapidity residue", 1000, -0.1, 0.1 ) );
  h_pt.push_back( ibook.book1D("pullPt", "pull of p_{t}", 100, -10, 10 ) );
  h_pullTheta.push_back( ibook.book1D("pullTheta","pull of #theta parameter",250,-25,25) );
  h_pullPhi.push_back( ibook.book1D("pullPhi","pull of #phi parameter",250,-25,25) );
  h_pullDxy.push_back( ibook.book1D("pullDxy","pull of dxy parameter",250,-25,25) );
  h_pullDz.push_back( ibook.book1D("pullDz","pull of dz parameter",250,-25,25) );
  h_pullQoverp.push_back( ibook.book1D("pullQoverp","pull of qoverp parameter",250,-25,25) );

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

  etares_vs_eta.push_back( ibook.book2D("etares_vs_eta","etaresidue vs eta",nintEta,minEta,maxEta,200,-0.1,0.1) );
  nrec_vs_nsim.push_back( ibook.book2D("nrec_vs_nsim","nrec vs nsim", nintTracks,minTracks,maxTracks, nintTracks,minTracks,maxTracks) );

  chi2_vs_eta.push_back( ibook.bookProfile("chi2mean","mean #chi^{2} vs #eta",nintEta,minEta,maxEta, 200, 0, 20, " " ));
  chi2_vs_phi.push_back( ibook.bookProfile("chi2mean_vs_phi","mean #chi^{2} vs #phi",nintPhi,minPhi,maxPhi, 200, 0, 20, " " ) );

  nhits_vs_eta.push_back( ibook.bookProfile("hits_eta","mean hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nPXBhits_vs_eta.push_back( ibook.bookProfile("PXBhits_vs_eta","mean # PXB its vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nPXFhits_vs_eta.push_back( ibook.bookProfile("PXFhits_vs_eta","mean # PXF hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nTIBhits_vs_eta.push_back( ibook.bookProfile("TIBhits_vs_eta","mean # TIB hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nTIDhits_vs_eta.push_back( ibook.bookProfile("TIDhits_vs_eta","mean # TID hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nTOBhits_vs_eta.push_back( ibook.bookProfile("TOBhits_vs_eta","mean # TOB hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nTEChits_vs_eta.push_back( ibook.bookProfile("TEChits_vs_eta","mean # TEC hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );

  nLayersWithMeas_vs_eta.push_back( ibook.bookProfile("LayersWithMeas_eta","mean # Layers with measurement vs eta",
                                                      nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nPXLlayersWithMeas_vs_eta.push_back( ibook.bookProfile("PXLlayersWithMeas_vs_eta","mean # PXL Layers with measurement vs eta",
                                                         nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nSTRIPlayersWithMeas_vs_eta.push_back( ibook.bookProfile("STRIPlayersWithMeas_vs_eta","mean # STRIP Layers with measurement vs eta",
                                                           nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nSTRIPlayersWith1dMeas_vs_eta.push_back( ibook.bookProfile("STRIPlayersWith1dMeas_vs_eta","mean # STRIP Layers with 1D measurement vs eta",
                                                             nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );
  nSTRIPlayersWith2dMeas_vs_eta.push_back( ibook.bookProfile("STRIPlayersWith2dMeas_vs_eta","mean # STRIP Layers with 2D measurement vs eta",
                                                             nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );

  nhits_vs_phi.push_back( ibook.bookProfile("hits_phi","mean # hits vs #phi",nintPhi,minPhi,maxPhi,nintHit,minHit,maxHit, " ") );

  nlosthits_vs_eta.push_back( ibook.bookProfile("losthits_vs_eta","mean # lost hits vs eta",nintEta,minEta,maxEta,nintHit,minHit,maxHit, " ") );

  //resolution of track parameters
  //                       dPt/Pt    cotTheta        Phi            TIP            LIP
  // log10(pt)<0.5        100,0.1    240,0.08     100,0.015      100,0.1000    150,0.3000
  // 0.5<log10(pt)<1.5    100,0.1    120,0.01     100,0.003      100,0.0100    150,0.0500
  // >1.5                 100,0.3    100,0.005    100,0.0008     100,0.0060    120,0.0300

  ptres_vs_eta.push_back(ibook.book2D("ptres_vs_eta","ptres_vs_eta",
				      nintEta,minEta,maxEta, ptRes_nbin, ptRes_rangeMin, ptRes_rangeMax));

  ptres_vs_phi.push_back( ibook.book2D("ptres_vs_phi","p_{t} res vs #phi",
				       nintPhi,minPhi,maxPhi, ptRes_nbin, ptRes_rangeMin, ptRes_rangeMax));

  ptres_vs_pt.push_back(ibook.book2D("ptres_vs_pt","ptres_vs_pt",nintPt,minPt,maxPt, ptRes_nbin, ptRes_rangeMin, ptRes_rangeMax));

  cotThetares_vs_eta.push_back(ibook.book2D("cotThetares_vs_eta","cotThetares_vs_eta",
					    nintEta,minEta,maxEta,cotThetaRes_nbin, cotThetaRes_rangeMin, cotThetaRes_rangeMax));


  cotThetares_vs_pt.push_back(ibook.book2D("cotThetares_vs_pt","cotThetares_vs_pt",
					   nintPt,minPt,maxPt, cotThetaRes_nbin, cotThetaRes_rangeMin, cotThetaRes_rangeMax));


  phires_vs_eta.push_back(ibook.book2D("phires_vs_eta","phires_vs_eta",
				       nintEta,minEta,maxEta, phiRes_nbin, phiRes_rangeMin, phiRes_rangeMax));

  phires_vs_pt.push_back(ibook.book2D("phires_vs_pt","phires_vs_pt",
				      nintPt,minPt,maxPt, phiRes_nbin, phiRes_rangeMin, phiRes_rangeMax));

  phires_vs_phi.push_back(ibook.book2D("phires_vs_phi","#phi res vs #phi",
				       nintPhi,minPhi,maxPhi,phiRes_nbin, phiRes_rangeMin, phiRes_rangeMax));

  dxyres_vs_eta.push_back(ibook.book2D("dxyres_vs_eta","dxyres_vs_eta",
				       nintEta,minEta,maxEta,dxyRes_nbin, dxyRes_rangeMin, dxyRes_rangeMax));

  dxyres_vs_pt.push_back( ibook.book2D("dxyres_vs_pt","dxyres_vs_pt",
				       nintPt,minPt,maxPt,dxyRes_nbin, dxyRes_rangeMin, dxyRes_rangeMax));

  dzres_vs_eta.push_back(ibook.book2D("dzres_vs_eta","dzres_vs_eta",
				      nintEta,minEta,maxEta,dzRes_nbin, dzRes_rangeMin, dzRes_rangeMax));

  dzres_vs_pt.push_back(ibook.book2D("dzres_vs_pt","dzres_vs_pt",nintPt,minPt,maxPt,dzRes_nbin, dzRes_rangeMin, dzRes_rangeMax));

  ptmean_vs_eta_phi.push_back(ibook.bookProfile2D("ptmean_vs_eta_phi","mean p_{t} vs #eta and #phi",
						  nintPhi,minPhi,maxPhi,nintEta,minEta,maxEta,1000,0,1000));
  phimean_vs_eta_phi.push_back(ibook.bookProfile2D("phimean_vs_eta_phi","mean #phi vs #eta and #phi",
						   nintPhi,minPhi,maxPhi,nintEta,minEta,maxEta,nintPhi,minPhi,maxPhi));

  //pulls of track params vs eta: to be used with fitslicesytool
  dxypull_vs_eta.push_back(ibook.book2D("dxypull_vs_eta","dxypull_vs_eta",nintEta,minEta,maxEta,100,-10,10));
  ptpull_vs_eta.push_back(ibook.book2D("ptpull_vs_eta","ptpull_vs_eta",nintEta,minEta,maxEta,100,-10,10));
  dzpull_vs_eta.push_back(ibook.book2D("dzpull_vs_eta","dzpull_vs_eta",nintEta,minEta,maxEta,100,-10,10));
  phipull_vs_eta.push_back(ibook.book2D("phipull_vs_eta","phipull_vs_eta",nintEta,minEta,maxEta,100,-10,10));
  thetapull_vs_eta.push_back(ibook.book2D("thetapull_vs_eta","thetapull_vs_eta",nintEta,minEta,maxEta,100,-10,10));

  //      h_ptshiftetamean.push_back( ibook.book1D("h_ptshifteta_Mean","<#deltapT/pT>[%] vs #eta",nintEta,minEta,maxEta) );


  //pulls of track params vs phi
  ptpull_vs_phi.push_back(ibook.book2D("ptpull_vs_phi","p_{t} pull vs #phi",nintPhi,minPhi,maxPhi,100,-10,10));
  phipull_vs_phi.push_back(ibook.book2D("phipull_vs_phi","#phi pull vs #phi",nintPhi,minPhi,maxPhi,100,-10,10));
  thetapull_vs_phi.push_back(ibook.book2D("thetapull_vs_phi","#theta pull vs #phi",nintPhi,minPhi,maxPhi,100,-10,10));


  nrecHit_vs_nsimHit_rec2sim.push_back( ibook.book2D("nrecHit_vs_nsimHit_rec2sim","nrecHit vs nsimHit (Rec2simAssoc)",
						     nintHit,minHit,maxHit, nintHit,minHit,maxHit ));

  if(useLogPt){
    BinLogX(dzres_vs_pt.back()->getTH2F());
    BinLogX(dxyres_vs_pt.back()->getTH2F());
    BinLogX(phires_vs_pt.back()->getTH2F());
    BinLogX(cotThetares_vs_pt.back()->getTH2F());
    BinLogX(ptres_vs_pt.back()->getTH2F());
    BinLogX(h_looperpT.back()->getTH1F());
    BinLogX(h_misidpT.back()->getTH1F());
    BinLogX(h_recopT.back()->getTH1F());
    BinLogX(h_assoc2pT.back()->getTH1F());
    BinLogX(h_pileuppT.back()->getTH1F());
  }
}

void MTVHistoProducerAlgoForTracker::bookRecoPVAssociationHistos(DQMStore::IBooker& ibook){
  h_recodxypv.push_back( ibook.book1D("num_reco_dxypv","N of reco track vs dxy(PV)",nintDxy,minDxy,maxDxy) );
  h_assoc2dxypv.push_back( ibook.book1D("num_assoc(recoToSim)_dxypv","N of associated (recoToSim) tracks vs dxy(PV)",nintDxy,minDxy,maxDxy) );
  h_looperdxypv.push_back( ibook.book1D("num_duplicate_dxypv","N of associated (recoToSim) looper tracks vs dxy(PV)",nintDxy,minDxy,maxDxy) );
  h_misiddxypv.push_back( ibook.book1D("num_chargemisid_dxypv","N of associated (recoToSim) charge misIDed tracks vs dxy(PV)",nintDxy,minDxy,maxDxy) );
  h_pileupdxypv.push_back( ibook.book1D("num_pileup_dxypv","N of associated (recoToSim) pileup tracks vs dxy(PV)",nintDxy,minDxy,maxDxy) );

  h_recodzpv.push_back( ibook.book1D("num_reco_dzpv","N of reco track vs dz(PV)",nintDz,minDz,maxDz) );
  h_assoc2dzpv.push_back( ibook.book1D("num_assoc(recoToSim)_dzpv","N of associated (recoToSim) tracks vs dz(PV)",nintDz,minDz,maxDz) );
  h_looperdzpv.push_back( ibook.book1D("num_duplicate_dzpv","N of associated (recoToSim) looper tracks vs dz(PV)",nintDz,minDz,maxDz) );
  h_misiddzpv.push_back( ibook.book1D("num_chargemisid_versus_dzpv","N of associated (recoToSim) charge misIDed tracks vs dz(PV)",nintDz,minDz,maxDz) );
  h_pileupdzpv.push_back( ibook.book1D("num_pileup_dzpv","N of associated (recoToSim) pileup tracks vs dz(PV)",nintDz,minDz,maxDz) );

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
									 const math::XYZPoint *pvPosition){
  bool isMatched = track;
  const auto eta = getEta(momentumTP.eta());
  const auto phi = momentumTP.phi();
  const auto pt = getPt(sqrt(momentumTP.perp2()));
  const auto nSim3DLayers = nSimPixelLayers + nSimStripMonoAndStereoLayers;
  const auto vertxy = sqrt(vertexTP.perp2());
  const auto vertz = vertexTP.z();

  if((*TpSelectorForEfficiencyVsEta)(tp)){
    //effic vs eta
    fillPlotNoFlow(h_simuleta[count], eta);
    if (isMatched) fillPlotNoFlow(h_assoceta[count], eta);
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
      nrecHit_vs_nsimHit_sim2rec[count]->Fill( track->numberOfValidHits(),nSimHits);
    }
    //effic vs pu
    fillPlotNoFlow(h_simulpu[count], numVertices);
    if(isMatched) fillPlotNoFlow(h_assocpu[count],numVertices);
    //efficiency vs dR
    fillPlotNoFlow(h_simuldr[count],dR);
    if (isMatched) fillPlotNoFlow(h_assocdr[count],dR);
  }

  if((*TpSelectorForEfficiencyVsPhi)(tp)){
    fillPlotNoFlow(h_simulphi[count], phi);
    if (isMatched) fillPlotNoFlow(h_assocphi[count], phi);
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
      if (isMatched) fillPlotNoFlow(h_assocdxypv[count], dxyPVSim);
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

      h_simul_dzpvcut[count]->Fill(0);
      h_simul_dzpvsigcut[count]->Fill(0);
      h_simul_dzpvcut_pt[count]->Fill(0, pt);
      h_simul_dzpvsigcut_pt[count]->Fill(0, pt);
      if(isMatched) {
        fillPlotNoFlow(h_assocdzpv[count], dzPVSim);

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
  }

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
								   const math::XYZPoint& bsPosition,
								   const math::XYZPoint *pvPosition,
								   bool isMatched,
								   bool isSigMatched,
								   bool isChargeMatched,
								   int numAssocRecoTracks,
								   int numVertices,
								   int nSimHits,
								   double sharedFraction,
								   double dR){

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
  const auto deltar = min(max(dR,h_recodr[count]->getTH1()->GetXaxis()->GetXmin()),h_recodr[count]->getTH1()->GetXaxis()->GetXmax());
  const auto chi2 = track.normalizedChi2();

  fillPlotNoFlow(h_recoeta[count], eta);
  fillPlotNoFlow(h_recophi[count], phi);
  fillPlotNoFlow(h_recopT[count], pt);
  fillPlotNoFlow(h_recodxy[count], dxy);
  fillPlotNoFlow(h_recodz[count], dz);
  fillPlotNoFlow(h_recohit[count], nhits);
  fillPlotNoFlow(h_recolayer[count], nlayers);
  fillPlotNoFlow(h_recopixellayer[count], nPixelLayers);
  fillPlotNoFlow(h_reco3Dlayer[count], n3DLayers);
  fillPlotNoFlow(h_recopu[count],numVertices);
  fillPlotNoFlow(h_recochi2[count], chi2);
  h_recodr[count]->Fill(deltar);
  if(pvPosition) {
    fillPlotNoFlow(h_recodxypv[count], dxypv);
    fillPlotNoFlow(h_recodzpv[count], dzpv);

    h_reco_dzpvcut[count]->Fill(std::abs(dzpv));
    h_reco_dzpvsigcut[count]->Fill(std::abs(dzpvsig));
    h_reco_dzpvcut_pt[count]->Fill(std::abs(dzpv), pt);
    h_reco_dzpvsigcut_pt[count]->Fill(std::abs(dzpvsig), pt);
  }

  if (isMatched) {
    fillPlotNoFlow(h_assoc2eta[count], eta);
    fillPlotNoFlow(h_assoc2phi[count], phi);
    fillPlotNoFlow(h_assoc2pT[count], pt);
    fillPlotNoFlow(h_assoc2dxy[count], dxy);
    fillPlotNoFlow(h_assoc2dz[count], dz);
    fillPlotNoFlow(h_assoc2hit[count], nhits);
    fillPlotNoFlow(h_assoc2layer[count], nlayers);
    fillPlotNoFlow(h_assoc2pixellayer[count], nPixelLayers);
    fillPlotNoFlow(h_assoc23Dlayer[count], n3DLayers);
    fillPlotNoFlow(h_assoc2pu[count],numVertices);
    fillPlotNoFlow(h_assoc2chi2[count], chi2);
    h_assoc2dr[count]->Fill(deltar);
    if(pvPosition) {
      fillPlotNoFlow(h_assoc2dxypv[count], dxypv);
      fillPlotNoFlow(h_assoc2dzpv[count], dzpv);

      h_assoc2_dzpvcut[count]->Fill(std::abs(dzpv));
      h_assoc2_dzpvsigcut[count]->Fill(std::abs(dzpvsig));
      h_assoc2_dzpvcut_pt[count]->Fill(std::abs(dzpv), pt);
      h_assoc2_dzpvsigcut_pt[count]->Fill(std::abs(dzpvsig), pt);
    }

    nrecHit_vs_nsimHit_rec2sim[count]->Fill( track.numberOfValidHits(),nSimHits);
    h_assocFraction[count]->Fill( sharedFraction);
    h_assocSharedHit[count]->Fill( sharedHits);

    if (!isChargeMatched) {
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
      }
    }

    if (numAssocRecoTracks>1) {
      fillPlotNoFlow(h_loopereta[count], eta);
      fillPlotNoFlow(h_looperphi[count], phi);
      fillPlotNoFlow(h_looperpT[count], pt);
      fillPlotNoFlow(h_looperdxy[count], dxy);
      fillPlotNoFlow(h_looperdz[count], dz);
      fillPlotNoFlow(h_looperhit[count], nhits);
      fillPlotNoFlow(h_looperlayer[count], nlayers);
      fillPlotNoFlow(h_looperpixellayer[count], nPixelLayers);
      fillPlotNoFlow(h_looper3Dlayer[count], n3DLayers);
      fillPlotNoFlow(h_looperpu[count], numVertices);
      fillPlotNoFlow(h_looperchi2[count], chi2);
      if(pvPosition) {
        fillPlotNoFlow(h_looperdxypv[count], dxypv);
        fillPlotNoFlow(h_looperdzpv[count], dzpv);
      }
    }
    else if(!isSigMatched) {
      fillPlotNoFlow(h_pileupeta[count], eta);
      fillPlotNoFlow(h_pileupphi[count], phi);
      fillPlotNoFlow(h_pileuppT[count], pt);
      fillPlotNoFlow(h_pileupdxy[count], dxy);
      fillPlotNoFlow(h_pileupdz[count], dz);
      fillPlotNoFlow(h_pileuphit[count], nhits);
      fillPlotNoFlow(h_pileuplayer[count], nlayers);
      fillPlotNoFlow(h_pileuppixellayer[count], nPixelLayers);
      fillPlotNoFlow(h_pileup3Dlayer[count], n3DLayers);
      fillPlotNoFlow(h_pileuppu[count], numVertices);
      fillPlotNoFlow(h_pileupchi2[count], chi2);
      h_pileupdr[count]->Fill(deltar);
      if(pvPosition) {
        fillPlotNoFlow(h_pileupdxypv[count], dxypv);
        fillPlotNoFlow(h_pileupdzpv[count], dzpv);

        h_pileup_dzpvcut[count]->Fill(std::abs(dzpv));
        h_pileup_dzpvsigcut[count]->Fill(std::abs(dzpvsig));
        h_pileup_dzpvcut_pt[count]->Fill(std::abs(dzpv), pt);
        h_pileup_dzpvsigcut_pt[count]->Fill(std::abs(dzpvsig), pt);
      }
    }
  }
}


void MTVHistoProducerAlgoForTracker::fill_simAssociated_recoTrack_histos(int count,
									 const reco::Track& track){
    //nchi2 and hits global distributions
    h_nchi2[count]->Fill(track.normalizedChi2());
    h_nchi2_prob[count]->Fill(TMath::Prob(track.chi2(),(int)track.ndof()));
    h_hits[count]->Fill(track.numberOfValidHits());
    h_losthits[count]->Fill(track.numberOfLostHits());
    chi2_vs_nhits[count]->Fill(track.numberOfValidHits(),track.normalizedChi2());
    h_charge[count]->Fill( track.charge() );
    h_nmisslayers_inner[count]->Fill(track.hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS));
    h_nmisslayers_outer[count]->Fill(track.hitPattern().numberOfHits(reco::HitPattern::MISSING_OUTER_HITS));

    //chi2 and #hit vs eta: fill 2D histos
    const auto eta = getEta(track.eta());
    chi2_vs_eta[count]->Fill(eta, track.normalizedChi2());
    nhits_vs_eta[count]->Fill(eta, track.numberOfValidHits());
    nPXBhits_vs_eta[count]->Fill(eta, track.hitPattern().numberOfValidPixelBarrelHits());
    nPXFhits_vs_eta[count]->Fill(eta, track.hitPattern().numberOfValidPixelEndcapHits());
    nTIBhits_vs_eta[count]->Fill(eta, track.hitPattern().numberOfValidStripTIBHits());
    nTIDhits_vs_eta[count]->Fill(eta, track.hitPattern().numberOfValidStripTIDHits());
    nTOBhits_vs_eta[count]->Fill(eta, track.hitPattern().numberOfValidStripTOBHits());
    nTEChits_vs_eta[count]->Fill(eta, track.hitPattern().numberOfValidStripTECHits());
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


void MTVHistoProducerAlgoForTracker::fill_trackBased_histos(int count, int assTracks, int numRecoTracks, int numSimTracks){

   	h_tracks[count]->Fill(assTracks);
   	h_fakes[count]->Fill(numRecoTracks-assTracks);
  	nrec_vs_nsim[count]->Fill(numRecoTracks,numSimTracks);

}



void MTVHistoProducerAlgoForTracker::fill_ResoAndPull_recoTrack_histos(int count,
								       const TrackingParticle::Vector& momentumTP,
								       const TrackingParticle::Point& vertexTP,
								       int chargeTP,
								       const reco::Track& track,
								       const math::XYZPoint& bsPosition){

  // evaluation of TP parameters
  double qoverpSim = chargeTP/sqrt(momentumTP.x()*momentumTP.x()+momentumTP.y()*momentumTP.y()+momentumTP.z()*momentumTP.z());
  double lambdaSim = M_PI/2-momentumTP.theta();
  double phiSim    = momentumTP.phi();
  double dxySim    = (-vertexTP.x()*sin(momentumTP.phi())+vertexTP.y()*cos(momentumTP.phi()));
  double dzSim     = vertexTP.z() - (vertexTP.x()*momentumTP.x()+vertexTP.y()*momentumTP.y())/sqrt(momentumTP.perp2())
    * momentumTP.z()/sqrt(momentumTP.perp2());


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
    //effic vs hits
    fillPlotNoFlow(h_simulhit[count],(int)nSimHits);
    if(isMatched) {
      fillPlotNoFlow(h_assochit[count],(int)nSimHits);
      nrecHit_vs_nsimHit_sim2rec[count]->Fill( track->numberOfValidHits(),nSimHits);
    }
    //effic vs pu
    fillPlotNoFlow(h_simulpu[count],numVertices);
    if (isMatched) fillPlotNoFlow(h_assocpu[count],numVertices);
    //efficiency vs dR
    //not implemented for now
  }

  if((*GpSelectorForEfficiencyVsPhi)(tp)){
    fillPlotNoFlow(h_simulphi[count],momentumTP.phi());
    if (isMatched) fillPlotNoFlow(h_assocphi[count],momentumTP.phi());
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
