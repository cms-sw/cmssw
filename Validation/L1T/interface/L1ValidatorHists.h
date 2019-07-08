#ifndef L1ValidatorHists_h
#define L1ValidatorHists_h

#include "TH1F.h"
//#include <TH2F.h>

//#include <DataFormats/HepMCCandidate/interface/GenParticle.h>
#include <DataFormats/Candidate/interface/LeafCandidate.h>

#include <string>

#include "DQMServices/Core/interface/DQMStore.h"

/*#define DECLARE_HISTS(TYPE) \
TH2F * ## TYPE ## _N_Pt; \
TH2F * ## TYPE ## _N_Eta; \
TH2F * ## TYPE ## _Eff_Pt; \
TH2F * ## TYPE ## _Eff_Eta; \
TH2F * ## TYPE ## _dR; \
TH2F * ## TYPE ## _dPt;
*/

class L1ValidatorHists {
public:
  L1ValidatorHists();
  ~L1ValidatorHists();

  void Book(DQMStore::IBooker &, std::string dirname);
  void Write();

  int NEvents;

  // enum Type{ IsoEG, NonIsoEG, CenJet, ForJet, TauJet, Muon, Number };
  enum Type { Egamma, Jet, Tau, Muon, Number };
  std::string Name[4];

  // Each object has gen (pt, eta, phi, pdg), reco (pt, eta, phi, pdg)
  // split by pdg (CenJet, ETM, ETT, ForJet, HTM, HTT, IsoEG, Mu, NoIsoEG,
  // TauJet): 10 needs efficiency (by pt, eta), fake rate (by pt, eta), dR (by
  // pt, (eta?))
  /**TH2F *IsoEG_N_Pt_Eta, *IsoEG_Eff_Pt, *IsoEG_Eff_Eta, *IsoEG_dR_Pt,
   *IsoEG_dPt_Pt; TH2F *NonIsoEG_N_Pt_Eta, *NonIsoEG_Eff_Pt, *NonIsoEG_Eff_Eta,
   *NonIsoEG_dR_Pt, *NonIsoEG_dPt_Pt; TH2F *CenJet_N_Pt_Eta, *CenJet_Eff_Pt,
   *CenJet_Eff_Eta, *CenJet_dR_Pt, *CenJet_dPt_Pt; TH2F *Muon_N_Pt_Eta,
   *Muon_Eff_Pt, *Muon_Eff_Eta, *Muon_dR_Pt, *Muon_dPt_Pt;*/

  /*DECLARE_HISTS(IsoEG)
  DECLARE_HISTS(NonIsoEG)
  DECLARE_HISTS(CenJet)
  DECLARE_HISTS(ForJet)
  DECLARE_HISTS(TauJet)
  DECLARE_HISTS(Muon)*/

  MonitorElement *N[Type::Number];

  MonitorElement *Eff_Pt[Type::Number];
  MonitorElement *Eff_Pt_Denom[Type::Number];
  MonitorElement *Eff_Pt_Nomin[Type::Number];
  MonitorElement *Eff_Eta[Type::Number];
  MonitorElement *Eff_Eta_Denom[Type::Number];
  MonitorElement *Eff_Eta_Nomin[Type::Number];
  MonitorElement *TurnOn_15[Type::Number];
  MonitorElement *TurnOn_15_Denom[Type::Number];
  MonitorElement *TurnOn_15_Nomin[Type::Number];
  MonitorElement *TurnOn_30[Type::Number];
  MonitorElement *TurnOn_30_Denom[Type::Number];
  MonitorElement *TurnOn_30_Nomin[Type::Number];
  MonitorElement *dR[Type::Number];
  MonitorElement *dPt[Type::Number];
  MonitorElement *dR_vs_Pt[Type::Number];
  MonitorElement *dPt_vs_Pt[Type::Number];

  // add the rest...
  // TH2F *ETM_Delta, *ETT_Delta, *HTM_Delta, *HTT_Delta;

  void Fill(int, const reco::LeafCandidate *, const reco::LeafCandidate *);
  void FillNumber(int, int);
  //  private:
  //    void NormalizeSlices(TH2F *Hist);
};

#endif
