#include "Validation/RecoMuon/plugins/MuonTrackValidator.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/plugins/CosmicParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/interface/TrackingParticleIP.h"

#include "TMath.h"
#include <TF1.h>

using namespace std;
using namespace edm;

void MuonTrackValidator::bookHistograms(DQMEDAnalyzer::DQMStore::IBooker& ibooker,
                                        edm::Run const&,
                                        edm::EventSetup const& setup) {
  for (unsigned int ww = 0; ww < associators.size(); ww++) {
    for (unsigned int www = 0; www < label.size(); www++) {
      ibooker.cd();
      InputTag algo = label[www];
      string dirName = dirName_;

      auto setBinLogX = [this](TH1* th1) {
        if (this->useLogPt) {
          BinLogX(th1);
        }
      };

      if (!algo.process().empty())
        dirName += algo.process() + "_";
      if (!algo.label().empty())
        dirName += algo.label();
      if (!algo.instance().empty())
        dirName += ("_" + algo.instance());
      if (dirName.find("Tracks") < dirName.length()) {
        dirName.replace(dirName.find("Tracks"), 6, "Trks");
      }
      if (dirName.find("UpdatedAtVtx") < dirName.length()) {
        dirName.replace(dirName.find("UpdatedAtVtx"), 12, "UpdAtVtx");
      }
      string assoc = associators[ww];
      if (assoc.find("tpToTkmuTrackAssociation") < assoc.length()) {
        dirName += "_TkAsso";
      }
      std::replace(dirName.begin(), dirName.end(), ':', '_');
      ibooker.setCurrentFolder(dirName);

      h_tracks.push_back(
          ibooker.book1D("Ntracks", "Number of reconstructed tracks", nintNTracks, minNTracks, maxNTracks));
      h_fakes.push_back(ibooker.book1D("Nfakes", "Number of fake reco tracks", nintFTracks, minFTracks, maxFTracks));
      h_charge.push_back(ibooker.book1D("Ncharge", "track charge", 3, -1.5, 1.5));

      h_recoeta.push_back(ibooker.book1D("num_reco_eta", "N of reco track vs eta", nintEta, minEta, maxEta));
      h_assoceta.push_back(ibooker.book1D(
          "num_assoSimToReco_eta", "N of associated tracks (simToReco) vs eta", nintEta, minEta, maxEta));
      h_assoc2eta.push_back(ibooker.book1D(
          "num_assoRecoToSim_eta", "N of associated (recoToSim) tracks vs eta", nintEta, minEta, maxEta));
      h_simuleta.push_back(ibooker.book1D("num_simul_eta", "N of simulated tracks vs eta", nintEta, minEta, maxEta));
      h_misideta.push_back(ibooker.book1D("num_chargemisid_eta",
                                          "N of associated (simToReco) tracks with charge misID vs eta",
                                          nintEta,
                                          minEta,
                                          maxEta));

      h_recopT.push_back(ibooker.book1D("num_reco_pT", "N of reco track vs pT", nintPt, minPt, maxPt, setBinLogX));
      h_assocpT.push_back(ibooker.book1D(
          "num_assoSimToReco_pT", "N of associated tracks (simToReco) vs pT", nintPt, minPt, maxPt, setBinLogX));
      h_assoc2pT.push_back(ibooker.book1D(
          "num_assoRecoToSim_pT", "N of associated (recoToSim) tracks vs pT", nintPt, minPt, maxPt, setBinLogX));
      h_simulpT.push_back(
          ibooker.book1D("num_simul_pT", "N of simulated tracks vs pT", nintPt, minPt, maxPt, setBinLogX));
      h_misidpT.push_back(ibooker.book1D("num_chargemisid_pT",
                                         "N of associated (simToReco) tracks with charge misID vs pT",
                                         nintPt,
                                         minPt,
                                         maxPt,
                                         setBinLogX));

      h_recophi.push_back(ibooker.book1D("num_reco_phi", "N of reco track vs phi", nintPhi, minPhi, maxPhi));
      h_assocphi.push_back(ibooker.book1D(
          "num_assoSimToReco_phi", "N of associated tracks (simToReco) vs phi", nintPhi, minPhi, maxPhi));
      h_assoc2phi.push_back(ibooker.book1D(
          "num_assoRecoToSim_phi", "N of associated (recoToSim) tracks vs phi", nintPhi, minPhi, maxPhi));
      h_simulphi.push_back(ibooker.book1D("num_simul_phi", "N of simulated tracks vs phi", nintPhi, minPhi, maxPhi));
      h_misidphi.push_back(ibooker.book1D("num_chargemisid_phi",
                                          "N of associated (simToReco) tracks with charge misID vs phi",
                                          nintPhi,
                                          minPhi,
                                          maxPhi));

      h_recohit.push_back(ibooker.book1D("num_reco_hit", "N of reco tracks vs N SimHits", nintNHit, minNHit, maxNHit));
      h_assochit.push_back(ibooker.book1D(
          "num_assoSimToReco_hit", "N of associated tracks (simToReco) vs N SimHits", nintNHit, minNHit, maxNHit));
      h_assoc2hit.push_back(ibooker.book1D(
          "num_assoRecoToSim_hit", "N of associated (recoToSim) tracks vs N Rechits", nintNHit, minNHit, maxNHit));
      h_simulhit.push_back(
          ibooker.book1D("num_simul_hit", "N of simulated tracks vs N SimHits", nintNHit, minNHit, maxNHit));
      h_misidhit.push_back(ibooker.book1D("num_chargemisid_hit",
                                          "N of associated (recoToSim) tracks with charge misID vs N RecHits",
                                          nintNHit,
                                          minNHit,
                                          maxNHit));

      h_recodxy.push_back(ibooker.book1D("num_reco_dxy", "N of reco track vs dxy", nintDxy, minDxy, maxDxy));
      h_assocdxy.push_back(ibooker.book1D(
          "num_assoSimToReco_dxy", "N of associated tracks (simToReco) vs dxy", nintDxy, minDxy, maxDxy));
      h_assoc2dxy.push_back(ibooker.book1D(
          "num_assoRecoToSim_dxy", "N of associated (recoToSim) tracks vs dxy", nintDxy, minDxy, maxDxy));
      h_simuldxy.push_back(ibooker.book1D("num_simul_dxy", "N of simulated tracks vs dxy", nintDxy, minDxy, maxDxy));
      h_misiddxy.push_back(ibooker.book1D("num_chargemisid_dxy",
                                          "N of associated (simToReco) tracks with charge misID vs dxy",
                                          nintDxy,
                                          minDxy,
                                          maxDxy));
      h_recodz.push_back(ibooker.book1D("num_reco_dz", "N of reco track vs dz", nintDz, minDz, maxDz));
      h_assocdz.push_back(
          ibooker.book1D("num_assoSimToReco_dz", "N of associated tracks (simToReco) vs dz", nintDz, minDz, maxDz));
      h_assoc2dz.push_back(
          ibooker.book1D("num_assoRecoToSim_dz", "N of associated (recoToSim) tracks vs dz", nintDz, minDz, maxDz));
      h_simuldz.push_back(ibooker.book1D("num_simul_dz", "N of simulated tracks vs dz", nintDz, minDz, maxDz));
      h_misiddz.push_back(ibooker.book1D(
          "num_chargemisid_dz", "N of associated (simToReco) tracks with charge misID vs dz", nintDz, minDz, maxDz));

      h_assocRpos.push_back(ibooker.book1D(
          "num_assoSimToReco_Rpos", "N of associated tracks (simToReco) vs Radius", nintRpos, minRpos, maxRpos));
      h_simulRpos.push_back(
          ibooker.book1D("num_simul_Rpos", "N of simulated tracks vs Radius", nintRpos, minRpos, maxRpos));

      h_assocZpos.push_back(ibooker.book1D(
          "num_assoSimToReco_Zpos", "N of associated tracks (simToReco) vs Z", nintZpos, minZpos, maxZpos));
      h_simulZpos.push_back(ibooker.book1D("num_simul_Zpos", "N of simulated tracks vs Z", nintZpos, minZpos, maxZpos));

      h_recopu.push_back(ibooker.book1D("num_reco_pu", "N of reco track vs pu", nintPU, minPU, maxPU));
      h_assocpu.push_back(
          ibooker.book1D("num_assoSimToReco_pu", "N of associated tracks (simToReco) vs pu", nintPU, minPU, maxPU));
      h_assoc2pu.push_back(
          ibooker.book1D("num_assoRecoToSim_pu", "N of associated (recoToSim) tracks vs pu", nintPU, minPU, maxPU));
      h_simulpu.push_back(ibooker.book1D("num_simul_pu", "N of simulated tracks vs pu", nintPU, minPU, maxPU));
      h_misidpu.push_back(ibooker.book1D(
          "num_chargemisid_pu", "N of associated (simToReco) charge misIDed tracks vs pu", nintPU, minPU, maxPU));

      h_nchi2.push_back(ibooker.book1D("chi2", "Track normalized #chi^{2}", 80, 0., 20.));
      h_nchi2_prob.push_back(ibooker.book1D("chi2prob", "Probability of track normalized #chi^{2}", 100, 0., 1.));

      chi2_vs_nhits.push_back(
          ibooker.book2D("chi2_vs_nhits", "#chi^{2} vs nhits", nintNHit, minNHit, maxNHit, 20, 0., 10.));
      chi2_vs_eta.push_back(ibooker.book2D("chi2_vs_eta", "chi2_vs_eta", nintEta, minEta, maxEta, 40, 0., 20.));
      chi2_vs_phi.push_back(ibooker.book2D("chi2_vs_phi", "#chi^{2} vs #phi", nintPhi, minPhi, maxPhi, 40, 0., 20.));

      h_nhits.push_back(ibooker.book1D("nhits", "Number of hits per track", nintNHit, minNHit, maxNHit));
      nhits_vs_eta.push_back(
          ibooker.book2D("nhits_vs_eta", "Number of Hits vs eta", nintEta, minEta, maxEta, nintNHit, minNHit, maxNHit));
      nhits_vs_phi.push_back(
          ibooker.book2D("nhits_vs_phi", "#hits vs #phi", nintPhi, minPhi, maxPhi, nintNHit, minNHit, maxNHit));

      if (do_MUOhitsPlots) {
        nDThits_vs_eta.push_back(ibooker.book2D(
            "nDThits_vs_eta", "Number of DT hits vs eta", nintEta, minEta, maxEta, nintDTHit, minDTHit, maxDTHit));
        nCSChits_vs_eta.push_back(ibooker.book2D(
            "nCSChits_vs_eta", "Number of CSC hits vs eta", nintEta, minEta, maxEta, nintCSCHit, minCSCHit, maxCSCHit));
        nRPChits_vs_eta.push_back(ibooker.book2D(
            "nRPChits_vs_eta", "Number of RPC hits vs eta", nintEta, minEta, maxEta, nintRPCHit, minRPCHit, maxRPCHit));
        if (useGEMs_)
          nGEMhits_vs_eta.push_back(ibooker.book2D(
              "nGEMhits_vs_eta", "Number of GEM hits vs eta", nintEta, minEta, maxEta, nintNHit, minNHit, maxNHit));
        if (useME0_)
          nME0hits_vs_eta.push_back(ibooker.book2D(
              "nME0hits_vs_eta", "Number of ME0 hits vs eta", nintEta, minEta, maxEta, nintNHit, minNHit, maxNHit));
      }

      if (do_TRKhitsPlots) {
        nTRK_LayersWithMeas_vs_eta.push_back(ibooker.book2D("nTRK_LayersWithMeas_vs_eta",
                                                            "# TRK Layers with measurement vs eta",
                                                            nintEta,
                                                            minEta,
                                                            maxEta,
                                                            nintLayers,
                                                            minLayers,
                                                            maxLayers));
        nPixel_LayersWithMeas_vs_eta.push_back(ibooker.book2D("nPixel_LayersWithMeas_vs_eta",
                                                              "Number of Pixel Layers with measurement vs eta",
                                                              nintEta,
                                                              minEta,
                                                              maxEta,
                                                              nintPixels,
                                                              minPixels,
                                                              maxPixels));
        h_nmisslayers_inner.push_back(ibooker.book1D(
            "nTRK_misslayers_inner", "Number of missing inner TRK layers", nintLayers, minLayers, maxLayers));
        h_nmisslayers_outer.push_back(ibooker.book1D(
            "nTRK_misslayers_outer", "Number of missing outer TRK layers", nintLayers, minLayers, maxLayers));
        h_nlosthits.push_back(ibooker.book1D("nlosthits", "Number of lost hits per track", 6, -0.5, 5.5));
        nlosthits_vs_eta.push_back(ibooker.book2D(
            "nlosthits_vs_eta", "Number of lost hits per track vs eta", nintEta, minEta, maxEta, 6, -0.5, 5.5));
      }

      ptres_vs_eta.push_back(ibooker.book2D("ptres_vs_eta",
                                            "p_{T} Relative Residual vs #eta",
                                            nintEta,
                                            minEta,
                                            maxEta,
                                            ptRes_nbin,
                                            ptRes_rangeMin,
                                            ptRes_rangeMax));
      ptres_vs_phi.push_back(ibooker.book2D("ptres_vs_phi",
                                            "p_{T} Relative Residual vs #phi",
                                            nintPhi,
                                            minPhi,
                                            maxPhi,
                                            ptRes_nbin,
                                            ptRes_rangeMin,
                                            ptRes_rangeMax));
      ptres_vs_pt.push_back(ibooker.book2D("ptres_vs_pt",
                                           "p_{T} Relative Residual vs p_{T}",
                                           nintPt,
                                           minPt,
                                           maxPt,
                                           ptRes_nbin,
                                           ptRes_rangeMin,
                                           ptRes_rangeMax,
                                           setBinLogX));
      h_ptpull.push_back(ibooker.book1D("ptpull", "p_{T} Pull", 100, -10., 10.));
      ptpull_vs_eta.push_back(
          ibooker.book2D("ptpull_vs_eta", "p_{T} Pull vs #eta", nintEta, minEta, maxEta, 100, -10., 10.));
      ptpull_vs_phi.push_back(
          ibooker.book2D("ptpull_vs_phi", "p_{T} Pull vs #phi", nintPhi, minPhi, maxPhi, 100, -10., 10.));
      h_qoverppull.push_back(ibooker.book1D("qoverppull", "q/p Pull", 100, -10., 10.));

      h_etaRes.push_back(ibooker.book1D("etaRes", "#eta residual", etaRes_nbin, etaRes_rangeMin, etaRes_rangeMax));
      etares_vs_eta.push_back(ibooker.book2D("etares_vs_eta",
                                             "#eta Residual vs #eta",
                                             nintEta,
                                             minEta,
                                             maxEta,
                                             etaRes_nbin,
                                             etaRes_rangeMin,
                                             etaRes_rangeMax));

      thetaCotres_vs_eta.push_back(ibooker.book2D("thetaCotres_vs_eta",
                                                  "cot(#theta) Residual vs #eta",
                                                  nintEta,
                                                  minEta,
                                                  maxEta,
                                                  cotThetaRes_nbin,
                                                  cotThetaRes_rangeMin,
                                                  cotThetaRes_rangeMax));
      thetaCotres_vs_pt.push_back(ibooker.book2D("thetaCotres_vs_pt",
                                                 "cot(#theta) Residual vs p_{T}",
                                                 nintPt,
                                                 minPt,
                                                 maxPt,
                                                 cotThetaRes_nbin,
                                                 cotThetaRes_rangeMin,
                                                 cotThetaRes_rangeMax,
                                                 setBinLogX));
      h_thetapull.push_back(ibooker.book1D("thetapull", "#theta Pull", 100, -10., 10.));
      thetapull_vs_eta.push_back(
          ibooker.book2D("thetapull_vs_eta", "#theta Pull vs #eta", nintEta, minEta, maxEta, 100, -10, 10));
      thetapull_vs_phi.push_back(
          ibooker.book2D("thetapull_vs_phi", "#theta Pull vs #phi", nintPhi, minPhi, maxPhi, 100, -10, 10));

      phires_vs_eta.push_back(ibooker.book2D("phires_vs_eta",
                                             "#phi Residual vs #eta",
                                             nintEta,
                                             minEta,
                                             maxEta,
                                             phiRes_nbin,
                                             phiRes_rangeMin,
                                             phiRes_rangeMax));
      phires_vs_pt.push_back(ibooker.book2D("phires_vs_pt",
                                            "#phi Residual vs p_{T}",
                                            nintPt,
                                            minPt,
                                            maxPt,
                                            phiRes_nbin,
                                            phiRes_rangeMin,
                                            phiRes_rangeMax,
                                            setBinLogX));
      phires_vs_phi.push_back(ibooker.book2D("phires_vs_phi",
                                             "#phi Residual vs #phi",
                                             nintPhi,
                                             minPhi,
                                             maxPhi,
                                             phiRes_nbin,
                                             phiRes_rangeMin,
                                             phiRes_rangeMax));
      h_phipull.push_back(ibooker.book1D("phipull", "#phi Pull", 100, -10., 10.));
      phipull_vs_eta.push_back(
          ibooker.book2D("phipull_vs_eta", "#phi Pull vs #eta", nintEta, minEta, maxEta, 100, -10, 10));
      phipull_vs_phi.push_back(
          ibooker.book2D("phipull_vs_phi", "#phi Pull vs #phi", nintPhi, minPhi, maxPhi, 100, -10, 10));

      dxyres_vs_eta.push_back(ibooker.book2D("dxyres_vs_eta",
                                             "dxy Residual vs #eta",
                                             nintEta,
                                             minEta,
                                             maxEta,
                                             dxyRes_nbin,
                                             dxyRes_rangeMin,
                                             dxyRes_rangeMax));
      dxyres_vs_pt.push_back(ibooker.book2D("dxyres_vs_pt",
                                            "dxy Residual vs p_{T}",
                                            nintPt,
                                            minPt,
                                            maxPt,
                                            dxyRes_nbin,
                                            dxyRes_rangeMin,
                                            dxyRes_rangeMax,
                                            setBinLogX));
      h_dxypull.push_back(ibooker.book1D("dxypull", "dxy Pull", 100, -10., 10.));
      dxypull_vs_eta.push_back(
          ibooker.book2D("dxypull_vs_eta", "dxy Pull vs #eta", nintEta, minEta, maxEta, 100, -10, 10));

      dzres_vs_eta.push_back(ibooker.book2D(
          "dzres_vs_eta", "dz Residual vs #eta", nintEta, minEta, maxEta, dzRes_nbin, dzRes_rangeMin, dzRes_rangeMax));
      dzres_vs_pt.push_back(ibooker.book2D("dzres_vs_pt",
                                           "dz Residual vs p_{T}",
                                           nintPt,
                                           minPt,
                                           maxPt,
                                           dzRes_nbin,
                                           dzRes_rangeMin,
                                           dzRes_rangeMax,
                                           setBinLogX));
      h_dzpull.push_back(ibooker.book1D("dzpull", "dz Pull", 100, -10., 10.));
      dzpull_vs_eta.push_back(
          ibooker.book2D("dzpull_vs_eta", "dz Pull vs #eta", nintEta, minEta, maxEta, 100, -10, 10));

      nRecHits_vs_nSimHits.push_back(ibooker.book2D(
          "nRecHits_vs_nSimHits", "nRecHits vs nSimHits", nintNHit, minNHit, maxNHit, nintNHit, minNHit, maxNHit));

      if (MABH) {
        h_PurityVsQuality.push_back(
            ibooker.book2D("PurityVsQuality", "Purity vs Quality (MABH)", 20, 0.01, 1.01, 20, 0.01, 1.01));
      }

      if (associators[ww] == "trackAssociatorByChi2") {
        h_assochi2.push_back(ibooker.book1D("assocChi2", "track association #chi^{2}", 1000, 0., 100.));
        h_assochi2_prob.push_back(ibooker.book1D("assocChi2_prob", "probability of association #chi^{2}", 100, 0., 1.));
      } else if (associators[ww] == "trackAssociatorByHits") {
        h_assocFraction.push_back(ibooker.book1D("assocFraction", "fraction of shared hits", 22, 0., 1.1));
        h_assocSharedHit.push_back(ibooker.book1D("assocSharedHit", "number of shared hits", 41, -0.5, 40.5));
      }

    }  //for (unsigned int www=0;www<label.size();www++)
  }    //for (unsigned int ww=0;ww<associators.size();ww++)
}

void MuonTrackValidator::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  using namespace reco;

  edm::LogInfo("MuonTrackValidator") << "\n===================================================="
                                     << "\n"
                                     << "Analyzing new event"
                                     << "\n"
                                     << "====================================================\n"
                                     << "\n";

  edm::Handle<std::vector<PileupSummaryInfo> > puinfoH;
  int PU_NumInteractions(-1);

  edm::ESHandle<ParametersDefinerForTP> Lhc_parametersDefinerTP;
  edm::ESHandle<CosmicParametersDefinerForTP> _Cosmic_parametersDefinerTP;
  std::unique_ptr<ParametersDefinerForTP> Cosmic_parametersDefinerTP;

  if (parametersDefiner == "LhcParametersDefinerForTP") {
    Lhc_parametersDefinerTP = setup.getHandle(tpDefinerEsToken);

    // PileupSummaryInfo is contained only in collision events
    event.getByToken(pileupinfo_Token, puinfoH);
    for (std::vector<PileupSummaryInfo>::const_iterator puInfoIt = puinfoH->begin(); puInfoIt != puinfoH->end();
         ++puInfoIt) {
      if (puInfoIt->getBunchCrossing() == 0) {
        PU_NumInteractions = puInfoIt->getPU_NumInteractions();
        break;
      }
    }

  } else if (parametersDefiner == "CosmicParametersDefinerForTP") {
    //setup.get<TrackAssociatorRecord>().get(parametersDefiner, _Cosmic_parametersDefinerTP);
    _Cosmic_parametersDefinerTP = setup.getHandle(cosmictpDefinerEsToken);

    //Since we modify the object, we must clone it
    Cosmic_parametersDefinerTP = _Cosmic_parametersDefinerTP->clone();

    edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
    //warning: make sure the TP collection used in the map is the same used here
    event.getByToken(_simHitTpMapTag, simHitsTPAssoc);
    Cosmic_parametersDefinerTP->initEvent(simHitsTPAssoc);
    cosmictpSelector.initEvent(simHitsTPAssoc);
  } else {
    edm::LogError("MuonTrackValidator") << "Unexpected label: parametersDefiner = " << parametersDefiner.c_str()
                                        << "\n";
  }

  TrackingParticleRefVector TPrefV;
  const TrackingParticleRefVector* ptr_TPrefV = nullptr;
  edm::Handle<TrackingParticleCollection> TPCollection_H;
  edm::Handle<TrackingParticleRefVector> TPCollectionRefVector_H;

  if (label_tp_refvector) {
    event.getByToken(tp_refvector_Token, TPCollectionRefVector_H);
    ptr_TPrefV = TPCollectionRefVector_H.product();
  } else {
    event.getByToken(tp_Token, TPCollection_H);
    size_t nTP = TPCollection_H->size();
    for (size_t i = 0; i < nTP; ++i) {
      TPrefV.push_back(TrackingParticleRef(TPCollection_H, i));
    }
    ptr_TPrefV = &TPrefV;
  }
  TrackingParticleRefVector const& tPC = *ptr_TPrefV;

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  bool bs_Available = event.getByToken(bsSrc_Token, recoBeamSpotHandle);
  reco::BeamSpot bs;
  if (bs_Available)
    bs = *recoBeamSpotHandle;
  edm::LogVerbatim("MuonTrackValidator") << bs;

  std::vector<const reco::TrackToTrackingParticleAssociator*> associator;
  if (UseAssociators) {
    edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator;
    for (auto const& associatorName : associators) {
      event.getByLabel(associatorName, theAssociator);
      associator.push_back(theAssociator.product());
    }
  }

  int w = 0;
  for (unsigned int ww = 0; ww < associators.size(); ww++) {
    for (unsigned int www = 0; www < label.size(); www++) {
      //
      //get collections from the event
      //
      edm::Handle<edm::View<Track> > trackCollection;
      unsigned int trackCollectionSize = 0;

      reco::RecoToSimCollection recSimColl;
      reco::SimToRecoCollection simRecColl;

      // account for missing track collections (HLT case)
      if (!event.getByToken(track_Collection_Token[www], trackCollection) && ignoremissingtkcollection_) {
        recSimColl.post_insert();
        simRecColl.post_insert();
      }

      //associate tracks to TrackingParticles
      else {
        trackCollectionSize = trackCollection->size();

        if (UseAssociators) {
          edm::LogVerbatim("MuonTrackValidator")
              << "Analyzing " << label[www].process() << ":" << label[www].label() << ":" << label[www].instance()
              << " with " << associators[ww].c_str() << "\n";

          LogTrace("MuonTrackValidator") << "Calling associateRecoToSim method"
                                         << "\n";
          recSimColl = associator[ww]->associateRecoToSim(trackCollection, TPCollection_H);
          LogTrace("MuonTrackValidator") << "Calling associateSimToReco method"
                                         << "\n";
          simRecColl = associator[ww]->associateSimToReco(trackCollection, TPCollection_H);
        } else {
          edm::LogVerbatim("MuonTrackValidator")
              << "Analyzing " << label[www].process() << ":" << label[www].label() << ":" << label[www].instance()
              << " with " << associatormap.process() << ":" << associatormap.label() << ":" << associatormap.instance()
              << "\n";

          Handle<reco::SimToRecoCollection> simtorecoCollectionH;
          event.getByToken(simToRecoCollection_Token, simtorecoCollectionH);
          simRecColl = *simtorecoCollectionH.product();

          Handle<reco::RecoToSimCollection> recotosimCollectionH;
          event.getByToken(recoToSimCollection_Token, recotosimCollectionH);
          recSimColl = *recotosimCollectionH.product();
        }
      }

      //
      //fill simulation histograms
      //
      edm::LogVerbatim("MuonTrackValidator") << "\n# of TrackingParticles: " << tPC.size() << "\n";
      int ats = 0;
      int st = 0;

      for (size_t i = 0; i < tPC.size(); i++) {
        bool TP_is_matched = false;
        bool isChargeOK = true;
        double quality = 0.;

        const TrackingParticleRef& tpr = tPC[i];
        const TrackingParticle& tp = *tpr;

        TrackingParticle::Vector momentumTP;
        TrackingParticle::Point vertexTP;
        double dxySim = 0;
        double dzSim = 0;

        //If the TrackingParticle is collision-like, get the momentum and vertex at production state
        //and the impact parameters w.r.t. PCA
        if (parametersDefiner == "LhcParametersDefinerForTP") {
          LogTrace("MuonTrackValidator") << "TrackingParticle " << i;
          if (!tpSelector(tp))
            continue;
          momentumTP = tp.momentum();
          vertexTP = tp.vertex();
          TrackingParticle::Vector momentum = Lhc_parametersDefinerTP->momentum(event, setup, tpr);
          TrackingParticle::Point vertex = Lhc_parametersDefinerTP->vertex(event, setup, tpr);
          dxySim = TrackingParticleIP::dxy(vertex, momentum, bs.position());
          dzSim = TrackingParticleIP::dz(vertex, momentum, bs.position());
        }
        //for cosmics get the momentum and vertex at PCA
        else if (parametersDefiner == "CosmicParametersDefinerForTP") {
          edm::LogVerbatim("MuonTrackValidator") << "TrackingParticle " << i;
          if (!cosmictpSelector(tpr, &bs, event, setup))
            continue;
          momentumTP = Cosmic_parametersDefinerTP->momentum(event, setup, tpr);
          vertexTP = Cosmic_parametersDefinerTP->vertex(event, setup, tpr);
          dxySim = TrackingParticleIP::dxy(vertexTP, momentumTP, bs.position());
          dzSim = TrackingParticleIP::dz(vertexTP, momentumTP, bs.position());
        }

        // Number of counted SimHits depend on the selection of tracker and muon detectors (via cfg parameters)
        int nSimHits = 0;
        if (usetracker && usemuon) {
          nSimHits = tpr.get()->numberOfHits();
        } else if (!usetracker && usemuon) {
          nSimHits = tpr.get()->numberOfHits() - tpr.get()->numberOfTrackerHits();
        } else if (usetracker && !usemuon) {
          nSimHits = tpr.get()->numberOfTrackerHits();
        }

        edm::LogVerbatim("MuonTrackValidator") << "--------------------Selected TrackingParticle #" << tpr.key()
                                               << "  (N counted simhits = " << nSimHits << ")";
        edm::LogVerbatim("MuonTrackValidator")
            << "momentumTP: pt = " << sqrt(momentumTP.perp2()) << ", pz = " << momentumTP.z()
            << ", \t vertexTP: radius = " << sqrt(vertexTP.perp2()) << ",  z = " << vertexTP.z();
        st++;

        double TPeta = momentumTP.eta();
        double xTPeta = getEta(TPeta);  // may be |eta| in histos according to useFabsEta
        double TPpt = sqrt(momentumTP.perp2());
        double xTPpt = getPt(TPpt);  // may be 1/pt in histos according to useInvPt
        double TPphi = momentumTP.phi();
        double TPrpos = sqrt(vertexTP.perp2());
        double TPzpos = vertexTP.z();

        int assoc_recoTrack_NValidHits = 0;
        if (simRecColl.find(tpr) != simRecColl.end()) {
          auto const& rt = simRecColl[tpr];
          if (!rt.empty()) {
            RefToBase<Track> assoc_recoTrack = rt.begin()->first;
            TP_is_matched = true;
            ats++;
            if (assoc_recoTrack->charge() != tpr->charge())
              isChargeOK = false;
            quality = rt.begin()->second;
            assoc_recoTrack_NValidHits = assoc_recoTrack->numberOfValidHits();
            edm::LogVerbatim("MuonTrackValidator") << "-----------------------------associated to Track #"
                                                   << assoc_recoTrack.key() << " with quality:" << quality << "\n";
          }
        } else {
          edm::LogVerbatim("MuonTrackValidator")
              << "TrackingParticle #" << tpr.key() << " with pt,eta,phi: " << sqrt(momentumTP.perp2()) << " , "
              << momentumTP.eta() << " , " << momentumTP.phi() << " , "
              << " NOT associated to any reco::Track"
              << "\n";
        }

        // histos for efficiency vs eta
        fillPlotNoFlow(h_simuleta[w], xTPeta);
        if (TP_is_matched) {
          fillPlotNoFlow(h_assoceta[w], xTPeta);
          if (!isChargeOK)
            fillPlotNoFlow(h_misideta[w], xTPeta);
        }

        // histos for efficiency vs phi
        fillPlotNoFlow(h_simulphi[w], TPphi);
        if (TP_is_matched) {
          fillPlotNoFlow(h_assocphi[w], TPphi);
          if (!isChargeOK)
            fillPlotNoFlow(h_misidphi[w], TPphi);
        }

        // histos for efficiency vs pT
        fillPlotNoFlow(h_simulpT[w], xTPpt);
        if (TP_is_matched) {
          fillPlotNoFlow(h_assocpT[w], xTPpt);
          if (!isChargeOK)
            fillPlotNoFlow(h_misidpT[w], xTPpt);
        }

        // histos for efficiency vs dxy
        fillPlotNoFlow(h_simuldxy[w], dxySim);
        if (TP_is_matched) {
          fillPlotNoFlow(h_assocdxy[w], dxySim);
          if (!isChargeOK)
            fillPlotNoFlow(h_misiddxy[w], dxySim);
        }

        // histos for efficiency vs dz
        fillPlotNoFlow(h_simuldz[w], dzSim);
        if (TP_is_matched) {
          fillPlotNoFlow(h_assocdz[w], dzSim);
          if (!isChargeOK)
            fillPlotNoFlow(h_misiddz[w], dzSim);
        }

        // histos for efficiency vs Radius
        fillPlotNoFlow(h_simulRpos[w], TPrpos);
        if (TP_is_matched)
          fillPlotNoFlow(h_assocRpos[w], TPrpos);

        // histos for efficiency vs z position
        fillPlotNoFlow(h_simulZpos[w], TPzpos);
        if (TP_is_matched)
          fillPlotNoFlow(h_assocZpos[w], TPzpos);

        // histos for efficiency vs Number of Hits
        fillPlotNoFlow(h_simulhit[w], nSimHits);
        if (TP_is_matched) {
          fillPlotNoFlow(h_assochit[w], nSimHits);
          nRecHits_vs_nSimHits[w]->Fill(nSimHits, assoc_recoTrack_NValidHits);

          // charge misid is more useful w.r.t. nRecHits (filled after)
          //if (!isChargeOK) fillPlotNoFlow(h_misidhit[w], nSimHits);
        }

        // histos for efficiency vs PileUp
        fillPlotNoFlow(h_simulpu[w], PU_NumInteractions);
        if (TP_is_matched) {
          fillPlotNoFlow(h_assocpu[w], PU_NumInteractions);
          if (!isChargeOK)
            fillPlotNoFlow(h_misidpu[w], PU_NumInteractions);
        }

      }  // End for (size_t i = 0; i < tPCeff.size(); i++) {

      //
      //fill reconstructed track histograms
      //
      edm::LogVerbatim("MuonTrackValidator")
          << "\n# of reco::Tracks with " << label[www].process() << ":" << label[www].label() << ":"
          << label[www].instance() << ": " << trackCollectionSize << "\n";

      int at = 0;
      int rT = 0;
      for (edm::View<Track>::size_type i = 0; i < trackCollectionSize; ++i) {
        bool Track_is_matched = false;
        bool isChargeOK = true;
        RefToBase<Track> track(trackCollection, i);
        int nRecHits = track->numberOfValidHits();
        rT++;

        std::vector<std::pair<TrackingParticleRef, double> > tp;
        TrackingParticleRef tpr;

        // new logic (bidirectional)
        if (BiDirectional_RecoToSim_association) {
          edm::LogVerbatim("MuonTrackValidator") << "----------------------------------------Track #" << track.key()
                                                 << " (N valid rechits = " << nRecHits << ")";

          if (recSimColl.find(track) != recSimColl.end()) {
            tp = recSimColl[track];
            if (!tp.empty()) {
              tpr = tp.begin()->first;
              // RtS and StR must associate the same pair !
              if (simRecColl.find(tpr) != simRecColl.end()) {
                auto const& assoc_track_checkback = simRecColl[tpr].begin()->first;

                if (assoc_track_checkback.key() == track.key()) {
                  Track_is_matched = true;
                  at++;
                  if (track->charge() != tpr->charge())
                    isChargeOK = false;
                  double Purity = tp.begin()->second;
                  double Quality = simRecColl[tpr].begin()->second;
                  edm::LogVerbatim("MuonTrackValidator")
                      << "with pt=" << track->pt() << " associated with purity:" << Purity << " to TrackingParticle #"
                      << tpr.key() << "\n";
                  if (MABH)
                    h_PurityVsQuality[w]->Fill(Quality, Purity);
                }
              }
            }
          }

          if (!Track_is_matched)
            edm::LogVerbatim("MuonTrackValidator")
                << "with pt=" << track->pt() << " NOT associated to any TrackingParticle"
                << "\n";
        }
        // old logic, valid for cosmics 2-legs reco (bugged for collision scenario)
        else {
          if (recSimColl.find(track) != recSimColl.end()) {
            tp = recSimColl[track];
            if (!tp.empty()) {
              tpr = tp.begin()->first;
              Track_is_matched = true;
              at++;
              if (track->charge() != tpr->charge())
                isChargeOK = false;
              edm::LogVerbatim("MuonTrackValidator") << "reco::Track #" << track.key() << " with pt=" << track->pt()
                                                     << " associated with quality:" << tp.begin()->second << "\n";
            }
          } else {
            edm::LogVerbatim("MuonTrackValidator") << "reco::Track #" << track.key() << " with pt=" << track->pt()
                                                   << " NOT associated to any TrackingParticle"
                                                   << "\n";
          }
        }

        double etaRec = track->eta();
        double xetaRec = getEta(etaRec);

        double ptRec = track->pt();
        double xptRec = getPt(ptRec);

        double qoverpRec = track->qoverp();
        double phiRec = track->phi();
        double thetaRec = track->theta();
        double dxyRec = track->dxy(bs.position());
        double dzRec = track->dz(bs.position());

        double qoverpError = track->qoverpError();
        double ptError = track->ptError();
        double thetaError = track->thetaError();
        double phiError = track->phiError();
        double dxyError = track->dxyError();
        double dzError = track->dzError();

        // histos for fake rate vs eta
        fillPlotNoFlow(h_recoeta[w], xetaRec);
        if (Track_is_matched) {
          fillPlotNoFlow(h_assoc2eta[w], xetaRec);
        }

        // histos for fake rate vs phi
        fillPlotNoFlow(h_recophi[w], phiRec);
        if (Track_is_matched) {
          fillPlotNoFlow(h_assoc2phi[w], phiRec);
        }

        // histos for fake rate vs pT
        fillPlotNoFlow(h_recopT[w], xptRec);
        if (Track_is_matched) {
          fillPlotNoFlow(h_assoc2pT[w], xptRec);
        }

        // histos for fake rate vs dxy
        fillPlotNoFlow(h_recodxy[w], dxyRec);
        if (Track_is_matched) {
          fillPlotNoFlow(h_assoc2dxy[w], dxyRec);
        }

        // histos for fake rate vs dz
        fillPlotNoFlow(h_recodz[w], dzRec);
        if (Track_is_matched) {
          fillPlotNoFlow(h_assoc2dz[w], dzRec);
        }

        // histos for fake rate vs Number of RecHits in track
        fillPlotNoFlow(h_recohit[w], nRecHits);
        if (Track_is_matched) {
          fillPlotNoFlow(h_assoc2hit[w], nRecHits);
          // charge misid w.r.t. nRecHits
          if (!isChargeOK)
            fillPlotNoFlow(h_misidhit[w], nRecHits);
        }

        // histos for fake rate vs Number of PU interactions
        fillPlotNoFlow(h_recopu[w], PU_NumInteractions);
        if (Track_is_matched) {
          fillPlotNoFlow(h_assoc2pu[w], PU_NumInteractions);
        }

        // Fill other histos
        TrackingParticle* tpp = const_cast<TrackingParticle*>(tpr.get());
        // TrackingParticle parameters at point of closest approach to the beamline
        TrackingParticle::Vector momentumTP;
        TrackingParticle::Point vertexTP;

        if (parametersDefiner == "LhcParametersDefinerForTP") {
          // following reco plots are made only from tracks associated to selected signal TPs
          if (!(Track_is_matched && tpSelector(*tpp)))
            continue;
          else {
            momentumTP = Lhc_parametersDefinerTP->momentum(event, setup, tpr);
            vertexTP = Lhc_parametersDefinerTP->vertex(event, setup, tpr);
          }
        } else if (parametersDefiner == "CosmicParametersDefinerForTP") {
          // following reco plots are made only from tracks associated to selected signal TPs
          if (!(Track_is_matched && cosmictpSelector(tpr, &bs, event, setup)))
            continue;
          else {
            momentumTP = Cosmic_parametersDefinerTP->momentum(event, setup, tpr);
            vertexTP = Cosmic_parametersDefinerTP->vertex(event, setup, tpr);
          }
        }

        if (associators[ww] == "trackAssociatorByChi2") {
          //association chi2
          double assocChi2 = -tp.begin()->second;  //in association map is stored -chi2
          h_assochi2[www]->Fill(assocChi2);
          h_assochi2_prob[www]->Fill(TMath::Prob((assocChi2)*5, 5));
        } else if (associators[ww] == "trackAssociatorByHits") {
          double fraction = tp.begin()->second;
          h_assocFraction[www]->Fill(fraction);
          h_assocSharedHit[www]->Fill(fraction * nRecHits);
        }

        h_charge[w]->Fill(track->charge());

        // Hits
        h_nhits[w]->Fill(nRecHits);
        nhits_vs_eta[w]->Fill(xetaRec, nRecHits);
        nhits_vs_phi[w]->Fill(phiRec, nRecHits);

        if (do_MUOhitsPlots) {
          nDThits_vs_eta[w]->Fill(xetaRec, track->hitPattern().numberOfValidMuonDTHits());
          nCSChits_vs_eta[w]->Fill(xetaRec, track->hitPattern().numberOfValidMuonCSCHits());
          nRPChits_vs_eta[w]->Fill(xetaRec, track->hitPattern().numberOfValidMuonRPCHits());
          if (useGEMs_)
            nGEMhits_vs_eta[w]->Fill(xetaRec, track->hitPattern().numberOfValidMuonGEMHits());
          if (useME0_)
            nME0hits_vs_eta[w]->Fill(xetaRec, track->hitPattern().numberOfValidMuonME0Hits());
        }

        if (do_TRKhitsPlots) {
          nTRK_LayersWithMeas_vs_eta[w]->Fill(xetaRec, track->hitPattern().trackerLayersWithMeasurement());
          nPixel_LayersWithMeas_vs_eta[w]->Fill(xetaRec, track->hitPattern().pixelLayersWithMeasurement());
          h_nlosthits[w]->Fill(track->numberOfLostHits());
          h_nmisslayers_inner[w]->Fill(track->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS));
          h_nmisslayers_outer[w]->Fill(track->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS));
          nlosthits_vs_eta[w]->Fill(xetaRec, track->numberOfLostHits());
        }

        // normalized chi2
        h_nchi2[w]->Fill(track->normalizedChi2());
        h_nchi2_prob[w]->Fill(TMath::Prob(track->chi2(), (int)track->ndof()));
        chi2_vs_nhits[w]->Fill(nRecHits, track->normalizedChi2());
        chi2_vs_eta[w]->Fill(xetaRec, track->normalizedChi2());
        chi2_vs_phi[w]->Fill(phiRec, track->normalizedChi2());

        double ptSim = sqrt(momentumTP.perp2());
        double xptSim = getPt(ptSim);
        double qoverpSim = tpr->charge() / sqrt(momentumTP.x() * momentumTP.x() + momentumTP.y() * momentumTP.y() +
                                                momentumTP.z() * momentumTP.z());
        double etaSim = momentumTP.eta();
        double thetaSim = momentumTP.theta();
        double phiSim = momentumTP.phi();
        double dxySim = TrackingParticleIP::dxy(vertexTP, momentumTP, bs.position());
        double dzSim = TrackingParticleIP::dz(vertexTP, momentumTP, bs.position());

        double etares = etaRec - etaSim;
        double ptRelRes = (ptRec - ptSim) / ptSim;  // relative residual -> resolution
        double ptPull = (ptRec - ptSim) / ptError;
        double qoverpPull = (qoverpRec - qoverpSim) / qoverpError;
        double thetaPull = (thetaRec - thetaSim) / thetaError;
        double phiDiff = phiRec - phiSim;
        if (abs(phiDiff) > M_PI) {
          if (phiDiff > 0.)
            phiDiff = phiDiff - 2. * M_PI;
          else
            phiDiff = phiDiff + 2. * M_PI;
        }
        double phiPull = phiDiff / phiError;
        double dxyPull = (dxyRec - dxySim) / dxyError;
        double dzPull = (dzRec - dzSim) / dzError;

        h_etaRes[w]->Fill(etares);
        etares_vs_eta[w]->Fill(xetaRec, etares);

        ptres_vs_eta[w]->Fill(xetaRec, ptRelRes);
        ptres_vs_pt[w]->Fill(xptSim, ptRelRes);
        ptres_vs_phi[w]->Fill(phiRec, ptRelRes);
        h_ptpull[w]->Fill(ptPull);
        ptpull_vs_eta[w]->Fill(xetaRec, ptPull);
        ptpull_vs_phi[w]->Fill(phiRec, ptPull);
        h_qoverppull[w]->Fill(qoverpPull);

        thetaCotres_vs_eta[w]->Fill(xetaRec, cos(thetaRec) / sin(thetaRec) - cos(thetaSim) / sin(thetaSim));
        thetaCotres_vs_pt[w]->Fill(xptSim, cos(thetaRec) / sin(thetaRec) - cos(thetaSim) / sin(thetaSim));
        h_thetapull[w]->Fill(thetaPull);
        thetapull_vs_eta[w]->Fill(xetaRec, thetaPull);
        thetapull_vs_phi[w]->Fill(phiRec, thetaPull);

        phires_vs_eta[w]->Fill(xetaRec, phiDiff);
        phires_vs_pt[w]->Fill(xptSim, phiDiff);
        phires_vs_phi[w]->Fill(phiRec, phiDiff);
        h_phipull[w]->Fill(phiPull);
        phipull_vs_eta[w]->Fill(xetaRec, phiPull);
        phipull_vs_phi[w]->Fill(phiRec, phiPull);

        dxyres_vs_eta[w]->Fill(xetaRec, dxyRec - dxySim);
        dxyres_vs_pt[w]->Fill(xptSim, dxyRec - dxySim);
        h_dxypull[w]->Fill(dxyPull);
        dxypull_vs_eta[w]->Fill(xetaRec, dxyPull);

        dzres_vs_eta[w]->Fill(xetaRec, dzRec - dzSim);
        dzres_vs_pt[w]->Fill(xptSim, dzRec - dzSim);
        h_dzpull[w]->Fill(dzPull);
        dzpull_vs_eta[w]->Fill(xetaRec, dzPull);

        double contrib_Qoverp = qoverpPull * qoverpPull / 5;
        double contrib_dxy = dxyPull * dxyPull / 5;
        double contrib_dz = dzPull * dzPull / 5;
        double contrib_theta = thetaPull * thetaPull / 5;
        double contrib_phi = phiPull * phiPull / 5;
        double assoChi2 = contrib_Qoverp + contrib_dxy + contrib_dz + contrib_theta + contrib_phi;

        LogTrace("MuonTrackValidator") << "normalized Chi2 (track 5-dofs matching) = " << assoChi2 << "\n"
                                       << "\t contrib_Qoverp = " << contrib_Qoverp << "\n"
                                       << "\t contrib_theta = " << contrib_theta << "\n"
                                       << "\t contrib_phi = " << contrib_phi << "\n"
                                       << "\t contrib_dxy = " << contrib_dxy << "\n"
                                       << "\t contrib_dz = " << contrib_dz << "\n";

        LogTrace("MuonTrackValidator") << "ptRec = " << ptRec << "\n"
                                       << "etaRec = " << etaRec << "\n"
                                       << "qoverpRec = " << qoverpRec << "\n"
                                       << "thetaRec = " << thetaRec << "\n"
                                       << "phiRec = " << phiRec << "\n"
                                       << "dxyRec = " << dxyRec << "\n"
                                       << "dzRec = " << dzRec << "\n"
                                       << ""
                                       << "\n"
                                       << "qoverpError = " << qoverpError << "\n"
                                       << "thetaError = " << thetaError << "\n"
                                       << "phiError = " << phiError << "\n"
                                       << "dxyError = " << dxyError << "\n"
                                       << "dzError = " << dzError << "\n"
                                       << ""
                                       << "\n"
                                       << "ptSim = " << ptSim << "\n"
                                       << "etaSim = " << etaSim << "\n"
                                       << "qoverpSim = " << qoverpSim << "\n"
                                       << "thetaSim = " << thetaSim << "\n"
                                       << "phiSim = " << phiSim << "\n"
                                       << "dxySim = " << dxySim << "\n"
                                       << "dzSim = " << dzSim << "\n";
      }  // End of for(edm::View<Track>::size_type i=0; i<trackCollectionSize; ++i) {

      h_tracks[w]->Fill(at);
      h_fakes[w]->Fill(rT - at);
      edm::LogVerbatim("MuonTrackValidator") << "Total Simulated: " << st << "\n"
                                             << "Total Associated (simToReco): " << ats << "\n"
                                             << "Total Reconstructed: " << rT << "\n"
                                             << "Total Associated (recoToSim): " << at << "\n"
                                             << "Total Fakes: " << rT - at << "\n";
      w++;
    }  // End of for (unsigned int www=0;www<label.size();www++){
  }    //END of for (unsigned int ww=0;ww<associators.size();ww++){
}
