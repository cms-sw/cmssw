#ifndef Validation_RecoVertex_SVMonitoringBundle_h
#define Validation_RecoVertex_SVMonitoringBundle_h

// Package:    Validation/RecoVertex
// Class:      SVMonitoringBundle
//
/**\class SVMonitoringBundle Validation/RecoVertex/interface/SVMonitoringBundle.h

 Description: Bundle of monitoring elements for the SecondaryVertexAnalyzer,
              grouping all histograms for the same kinematic variable at one
              place (e.g. the sim, assocSimToReco, reco, assocRecoToSim, fake,
              duplicate and pileup histograms all vs decay length).

 Original Author: Jan Schulz
*/

#include "DQMServices/Core/interface/DQMBookingHelpers.h"
#include "SimTracker/TrackAssociation/interface/trackingVertexMotherPdgIdAndPt.h"

class SVMonitoringBundle {
public:
  using IBooker = dqm::reco::DQMStore::IBooker;

  SVMonitoringBundle() = default;
  virtual ~SVMonitoringBundle() = default;

  // ---------------------------------------------------------------------------
  // Filling — sim side
  //
  // isMatched:         this sim vertex was matched to at least one reco vertex
  // isReconstructable: this sim vertex passes the reconstructability criteria
  //                    (minimum decay length, minimum charged daughters, etc.)
  // isMerged:          this sim vertex was merged with another sim vertex into
  //                    a single reco vertex
  // args:              kinematic variable(s) to fill (e.g. decayLength, eta, nTracks)
  // ---------------------------------------------------------------------------
  template <typename... Args>
  void fillSimVertexHistos(const bool isMatched, const bool isReconstructable, const bool isMerged, Args... args) {
    h_sim->Fill(args...);
    if (isMatched)
      h_assocSimToReco->Fill(args...);
    if (isMerged)
      h_merged->Fill(args...);
    if (isReconstructable) {
      h_reconstructableSim->Fill(args...);
      if (isMatched)
        h_assocReconstructableSimToReco->Fill(args...);
    }
  }

  // ---------------------------------------------------------------------------
  // Filling — reco side
  //
  // isMatched:   this reco vertex was matched to a sim vertex
  // isDuplicate: this reco vertex is a duplicate (sim vertex already matched
  //              to another reco vertex with higher quality)
  // isFake:      this reco vertex has no sim match at all
  // isFromPileup: matched sim vertex originates from a pileup interaction
  // args:         kinematic variable(s) to fill
  // ---------------------------------------------------------------------------
  template <typename... Args>
  void fillRecoVertexHistos(
      const bool isMatched, const bool isDuplicate, const bool isFake, const bool isFromPileup, Args... args) {
    h_reco->Fill(args...);
    if (isMatched)
      h_assocRecoToSim->Fill(args...);
    if (isDuplicate)
      h_duplicate->Fill(args...);
    if (isFake)
      h_fake->Fill(args...);
    if (isFromPileup)
      h_pileup->Fill(args...);
  }

  // ---------------------------------------------------------------------------
  // Filling — per-PDG category
  //
  // Fills the b-hadron, c-hadron, or other-origin histogram based on the
  // absolute value of the mother PDG ID. Intended for use on the sim side
  // to provide b/c/other efficiency breakdowns for tagging studies.
  //
  // Histograms are only filled if they were booked (non-null); calling this
  // method when bookPerPdgHistos=false in booking is safe.
  // ---------------------------------------------------------------------------
  template <typename... Args>
  void fillSimVertexHistosByPdg(const int motherPdgId, const bool isMatched, Args... args) {
    if (sim::isBHadron(motherPdgId) && h_sim_b) {
      h_sim_b->Fill(args...);
      if (isMatched && h_assocSimToReco_b)
        h_assocSimToReco_b->Fill(args...);
    } else if (sim::isCHadron(motherPdgId) && h_sim_c) {
      h_sim_c->Fill(args...);
      if (isMatched && h_assocSimToReco_c)
        h_assocSimToReco_c->Fill(args...);
    } else if (sim::isSHadron(motherPdgId) && h_sim_s) {
      h_sim_s->Fill(args...);
      if (isMatched && h_assocSimToReco_s)
        h_assocSimToReco_s->Fill(args...);
    } else if (sim::isTau(motherPdgId) && h_sim_tau) {
      h_sim_tau->Fill(args...);
      if (isMatched && h_assocSimToReco_tau)
        h_assocSimToReco_tau->Fill(args...);
    } else if (h_sim_other) {
      h_sim_other->Fill(args...);
      if (isMatched && h_assocSimToReco_other)
        h_assocSimToReco_other->Fill(args...);
    }
  }

  // ---------------------------------------------------------------------------
  // Booking
  // ---------------------------------------------------------------------------

  /// Book a 1D bundle, optionally with log X axis and per-PDG histograms.
  template <typename... Args>
  void book1DIfLogX(IBooker &ibooker,
                    const bool logScale,
                    const bool bookSimHistos,
                    const bool bookRecoHistos,
                    const bool bookPerPdgHistos,
                    Args &&...args) {
    bookGeneric(
        [](auto &ib, bool log, auto &&...innerArgs) {
          return dqm::booking::book1DIfLogX(ib, log, std::forward<decltype(innerArgs)>(innerArgs)...);
        },
        ibooker,
        logScale,
        bookSimHistos,
        bookRecoHistos,
        bookPerPdgHistos,
        std::forward<Args>(args)...);
  }

  template <typename... Args>
  void book1D(IBooker &ibooker,
              const bool bookSimHistos,
              const bool bookRecoHistos,
              const bool bookPerPdgHistos,
              Args &&...args) {
    book1DIfLogX(ibooker, false, bookSimHistos, bookRecoHistos, bookPerPdgHistos, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void book1DLogX(IBooker &ibooker,
                  const bool bookSimHistos,
                  const bool bookRecoHistos,
                  const bool bookPerPdgHistos,
                  Args &&...args) {
    book1DIfLogX(ibooker, true, bookSimHistos, bookRecoHistos, bookPerPdgHistos, std::forward<Args>(args)...);
  }

  /// Book a 2D bundle, optionally with log X axis and per-PDG histograms.
  template <typename... Args>
  void book2DIfLogX(IBooker &ibooker,
                    const bool logScale,
                    const bool bookSimHistos,
                    const bool bookRecoHistos,
                    const bool bookPerPdgHistos,
                    Args &&...args) {
    bookGeneric(
        [](auto &ib, bool log, auto &&...innerArgs) {
          return dqm::booking::book2DIfLogX(ib, log, std::forward<decltype(innerArgs)>(innerArgs)...);
        },
        ibooker,
        logScale,
        bookSimHistos,
        bookRecoHistos,
        bookPerPdgHistos,
        std::forward<Args>(args)...);
  }

  template <typename... Args>
  void book2D(IBooker &ibooker,
              const bool bookSimHistos,
              const bool bookRecoHistos,
              const bool bookPerPdgHistos,
              Args &&...args) {
    book2DIfLogX(ibooker, false, bookSimHistos, bookRecoHistos, bookPerPdgHistos, std::forward<Args>(args)...);
  }

  // Allow bulk modification of all booked histograms, e.g. for setting axis
  // labels on collection summary plots.
  template <typename ModificationFunc, typename... Args>
  void modifyHistograms(ModificationFunc modify, Args &&...args) {
    for (auto h : {h_sim,
                   h_reconstructableSim,
                   h_assocSimToReco,
                   h_assocReconstructableSimToReco,
                   h_reco,
                   h_assocRecoToSim,
                   h_duplicate,
                   h_fake,
                   h_merged,
                   h_pileup,
                   h_sim_b,
                   h_sim_c,
                   h_sim_other,
                   h_assocSimToReco_b,
                   h_assocSimToReco_c,
                   h_assocSimToReco_other}) {
      if (h)
        modify(h, std::forward<Args>(args)...);
    }
  }

private:
  // ---------------------------------------------------------------------------
  // Generic booking implementation
  // ---------------------------------------------------------------------------
  template <typename BookFunc, typename... Args>
  void bookGeneric(BookFunc bookFunc,
                   IBooker &ibooker,
                   const bool logScale,
                   const bool bookSimHistos,
                   const bool bookRecoHistos,
                   const bool bookPerPdgHistos,
                   const std::string &name,
                   const std::string &xlabel,
                   const std::string &ylabel,
                   const int nBins,
                   const double valMin,
                   const double valMax,
                   Args... args) {
    const std::string xylabels = ";" + xlabel + ";" + ylabel;

    if (bookSimHistos) {
      h_sim = bookFunc(ibooker,
                       logScale,
                       ("num_sim_" + name).c_str(),
                       ("N of simulated SVs" + xylabels).c_str(),
                       nBins,
                       valMin,
                       valMax,
                       args...);
      h_assocSimToReco = bookFunc(ibooker,
                                  logScale,
                                  ("num_assoc(simToReco)_" + name).c_str(),
                                  ("N of simulated SVs matched to a reco SV" + xylabels).c_str(),
                                  nBins,
                                  valMin,
                                  valMax,
                                  args...);
      h_reconstructableSim = bookFunc(ibooker,
                                      logScale,
                                      ("num_reconstructableSim_" + name).c_str(),
                                      ("N of reconstructable simulated SVs" + xylabels).c_str(),
                                      nBins,
                                      valMin,
                                      valMax,
                                      args...);
      h_assocReconstructableSimToReco =
          bookFunc(ibooker,
                   logScale,
                   ("num_assoc(reconstructableSimToReco)_" + name).c_str(),
                   ("N of reconstructable simulated SVs matched to a reco SV" + xylabels).c_str(),
                   nBins,
                   valMin,
                   valMax,
                   args...);
      h_merged =
          bookFunc(ibooker,
                   logScale,
                   ("num_merged_" + name).c_str(),
                   ("N of merged simulated SVs (matched to RecoSV with multiple sim-matches)" + xylabels).c_str(),
                   nBins,
                   valMin,
                   valMax,
                   args...);

      if (bookPerPdgHistos) {
        // b-hadron origin
        h_sim_b = bookFunc(ibooker,
                           logScale,
                           ("num_sim_b_" + name).c_str(),
                           ("N of simulated B-hadron SVs" + xylabels).c_str(),
                           nBins,
                           valMin,
                           valMax,
                           args...);
        h_assocSimToReco_b = bookFunc(ibooker,
                                      logScale,
                                      ("num_assoc(simToReco)_b_" + name).c_str(),
                                      ("N of simulated B-hadron SVs matched to a reco SV" + xylabels).c_str(),
                                      nBins,
                                      valMin,
                                      valMax,
                                      args...);
        // c-hadron origin
        h_sim_c = bookFunc(ibooker,
                           logScale,
                           ("num_sim_c_" + name).c_str(),
                           ("N of simulated D-hadron SVs" + xylabels).c_str(),
                           nBins,
                           valMin,
                           valMax,
                           args...);
        h_assocSimToReco_c = bookFunc(ibooker,
                                      logScale,
                                      ("num_assoc(simToReco)_c_" + name).c_str(),
                                      ("N of simulated D-hadron SVs matched to a reco SV" + xylabels).c_str(),
                                      nBins,
                                      valMin,
                                      valMax,
                                      args...);
        // c-hadron origin
        h_sim_s = bookFunc(ibooker,
                           logScale,
                           ("num_sim_s_" + name).c_str(),
                           ("N of simulated K-hadron SVs" + xylabels).c_str(),
                           nBins,
                           valMin,
                           valMax,
                           args...);
        h_assocSimToReco_s = bookFunc(ibooker,
                                      logScale,
                                      ("num_assoc(simToReco)_s_" + name).c_str(),
                                      ("N of simulated K-hadron SVs matched to a reco SV" + xylabels).c_str(),
                                      nBins,
                                      valMin,
                                      valMax,
                                      args...);
        // tau-hadron origin
        h_sim_tau = bookFunc(ibooker,
                             logScale,
                             ("num_sim_tau_" + name).c_str(),
                             ("N of simulated tau SVs" + xylabels).c_str(),
                             nBins,
                             valMin,
                             valMax,
                             args...);
        h_assocSimToReco_tau = bookFunc(ibooker,
                                        logScale,
                                        ("num_assoc(simToReco)_tau_" + name).c_str(),
                                        ("N of simulated tau SVs matched to a reco SV" + xylabels).c_str(),
                                        nBins,
                                        valMin,
                                        valMax,
                                        args...);
        // other origin (light hadrons, nuclear interactions, etc.)
        h_sim_other = bookFunc(ibooker,
                               logScale,
                               ("num_sim_other_" + name).c_str(),
                               ("N of simulated SVs from other origins" + xylabels).c_str(),
                               nBins,
                               valMin,
                               valMax,
                               args...);
        h_assocSimToReco_other = bookFunc(ibooker,
                                          logScale,
                                          ("num_assoc(simToReco)_other_" + name).c_str(),
                                          ("N of other simulated SVs matched to a reco SV" + xylabels).c_str(),
                                          nBins,
                                          valMin,
                                          valMax,
                                          args...);
      }
    }

    if (bookRecoHistos) {
      h_reco = bookFunc(ibooker,
                        logScale,
                        ("num_reco_" + name).c_str(),
                        ("N of reconstructed SVs" + xylabels).c_str(),
                        nBins,
                        valMin,
                        valMax,
                        args...);
      h_assocRecoToSim = bookFunc(ibooker,
                                  logScale,
                                  ("num_assoc(recoToSim)_" + name).c_str(),
                                  ("N of reconstructed SVs matched to a simulated SV" + xylabels).c_str(),
                                  nBins,
                                  valMin,
                                  valMax,
                                  args...);
      h_duplicate = bookFunc(ibooker,
                             logScale,
                             ("num_duplicate_" + name).c_str(),
                             ("N of duplicate reconstructed SVs" + xylabels).c_str(),
                             nBins,
                             valMin,
                             valMax,
                             args...);
      h_fake = bookFunc(ibooker,
                        logScale,
                        ("num_fake_" + name).c_str(),
                        ("N of fake reconstructed SVs (no sim match)" + xylabels).c_str(),
                        nBins,
                        valMin,
                        valMax,
                        args...);
      h_pileup = bookFunc(ibooker,
                          logScale,
                          ("num_pileup_" + name).c_str(),
                          ("N of reconstructed SVs matched to a pileup sim SV" + xylabels).c_str(),
                          nBins,
                          valMin,
                          valMax,
                          args...);
    }
  }

  // ---------------------------------------------------------------------------
  // Histogram pointers
  // ---------------------------------------------------------------------------

  // Sim side — always booked when bookSimHistos=true
  dqm::reco::MonitorElement *h_sim = nullptr;
  dqm::reco::MonitorElement *h_reconstructableSim = nullptr;
  dqm::reco::MonitorElement *h_assocSimToReco = nullptr;
  dqm::reco::MonitorElement *h_assocReconstructableSimToReco = nullptr;
  dqm::reco::MonitorElement *h_merged = nullptr;

  // Reco side — always booked when bookRecoHistos=true
  dqm::reco::MonitorElement *h_reco = nullptr;
  dqm::reco::MonitorElement *h_assocRecoToSim = nullptr;
  dqm::reco::MonitorElement *h_duplicate = nullptr;
  dqm::reco::MonitorElement *h_fake = nullptr;
  dqm::reco::MonitorElement *h_pileup = nullptr;

  // Per-PDG breakdown — only booked when bookPerPdgHistos=true
  dqm::reco::MonitorElement *h_sim_b = nullptr;
  dqm::reco::MonitorElement *h_sim_c = nullptr;
  dqm::reco::MonitorElement *h_sim_s = nullptr;
  dqm::reco::MonitorElement *h_sim_tau = nullptr;
  dqm::reco::MonitorElement *h_sim_other = nullptr;
  dqm::reco::MonitorElement *h_assocSimToReco_b = nullptr;
  dqm::reco::MonitorElement *h_assocSimToReco_c = nullptr;
  dqm::reco::MonitorElement *h_assocSimToReco_s = nullptr;
  dqm::reco::MonitorElement *h_assocSimToReco_tau = nullptr;
  dqm::reco::MonitorElement *h_assocSimToReco_other = nullptr;
};

#endif  // Validation_RecoVertex_SVMonitoringBundle_h
