#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Validation/HcalHits/interface/HcalSimHitStudy.h"

#include "FWCore/Utilities/interface/Exception.h"

HcalSimHitStudy::HcalSimHitStudy(const edm::ParameterSet &ps) {
  g4Label = ps.getUntrackedParameter<std::string>("moduleLabel", "g4SimHits");
  hcalHits = ps.getUntrackedParameter<std::string>("HitCollection", "HcalHits");
  outFile_ = ps.getUntrackedParameter<std::string>("outputFile", "hcHit.root");
  verbose_ = ps.getUntrackedParameter<bool>("Verbose", false);
  testNumber_ = ps.getParameter<bool>("TestNumber");
  hep17_ = ps.getParameter<bool>("hep17");
  checkHit_ = true;

  tok_hits_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label, hcalHits));
  tok_HRNDC_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord, edm::Transition::BeginRun>();

  edm::LogInfo("HcalSim") << "Module Label: " << g4Label << "   Hits: " << hcalHits << " / " << checkHit_
                          << "   Output: " << outFile_;
}

HcalSimHitStudy::~HcalSimHitStudy() {}

void HcalSimHitStudy::bookHistograms(DQMStore::IBooker &ib, edm::Run const &run, edm::EventSetup const &es) {
  const auto &pHRNDC = es.getData(tok_HRNDC_);
  hcons = &pHRNDC;
  maxDepthHB_ = hcons->getMaxDepth(0);
  maxDepthHE_ = hcons->getMaxDepth(1);
  maxDepthHF_ = hcons->getMaxDepth(2);
  maxDepthHO_ = hcons->getMaxDepth(3);
  maxDepth_ = (maxDepthHB_ > maxDepthHE_ ? maxDepthHB_ : maxDepthHE_);
  maxDepth_ = (maxDepth_ > maxDepthHF_ ? maxDepth_ : maxDepthHF_);
  maxDepth_ = (maxDepth_ > maxDepthHO_ ? maxDepth_ : maxDepthHO_);

  // Get Phi segmentation from geometry, use the max phi number so that all iphi
  // values are included.

  int NphiMax = hcons->getNPhi(0);

  NphiMax = (hcons->getNPhi(1) > NphiMax ? hcons->getNPhi(1) : NphiMax);
  NphiMax = (hcons->getNPhi(2) > NphiMax ? hcons->getNPhi(2) : NphiMax);
  NphiMax = (hcons->getNPhi(3) > NphiMax ? hcons->getNPhi(3) : NphiMax);

  // Center the iphi bins on the integers
  iphi_min = 0.5;
  iphi_max = NphiMax + 0.5;
  iphi_bins = (int)(iphi_max - iphi_min);

  int iEtaHBMax = hcons->getEtaRange(0).second;
  int iEtaHEMax = std::max(hcons->getEtaRange(1).second, 1);
  int iEtaHFMax = hcons->getEtaRange(2).second;
  int iEtaHOMax = hcons->getEtaRange(3).second;

  // Retain classic behavior, all plots have same ieta range.
  // Comment out code to allow each subdetector to have its on range

  int iEtaMax = (iEtaHBMax > iEtaHEMax ? iEtaHBMax : iEtaHEMax);
  iEtaMax = (iEtaMax > iEtaHFMax ? iEtaMax : iEtaHFMax);
  iEtaMax = (iEtaMax > iEtaHOMax ? iEtaMax : iEtaHOMax);

  iEtaHBMax = iEtaMax;
  iEtaHEMax = iEtaMax;
  iEtaHFMax = iEtaMax;
  iEtaHOMax = iEtaMax;

  // Give an empty bin around the subdet ieta range to make it clear that all
  // ieta rings have been included
  ieta_min_HB = -iEtaHBMax - 1.5;
  ieta_max_HB = iEtaHBMax + 1.5;
  ieta_bins_HB = (int)(ieta_max_HB - ieta_min_HB);

  ieta_min_HE = -iEtaHEMax - 1.5;
  ieta_max_HE = iEtaHEMax + 1.5;
  ieta_bins_HE = (int)(ieta_max_HE - ieta_min_HE);

  ieta_min_HF = -iEtaHFMax - 1.5;
  ieta_max_HF = iEtaHFMax + 1.5;
  ieta_bins_HF = (int)(ieta_max_HF - ieta_min_HF);

  ieta_min_HO = -iEtaHOMax - 1.5;
  ieta_max_HO = iEtaHOMax + 1.5;
  ieta_bins_HO = (int)(ieta_max_HO - ieta_min_HO);

  Char_t hname[100];
  Char_t htitle[100];

  ib.setCurrentFolder("HcalHitsV/HcalSimHitsTask");

  // Histograms for Hits
  if (checkHit_) {
    meAllNHit_ = ib.book1D("Hit01", "Number of Hits in HCal", 20000, 0., 20000.);
    meBadDetHit_ = ib.book1D("Hit02", "Hits with wrong Det", 100, 0., 100.);
    meBadSubHit_ = ib.book1D("Hit03", "Hits with wrong Subdet", 100, 0., 100.);
    meBadIdHit_ = ib.book1D("Hit04", "Hits with wrong ID", 100, 0., 100.);
    meHBNHit_ = ib.book1D("Hit05", "Number of Hits in HB", 20000, 0., 20000.);
    meHENHit_ = ib.book1D("Hit06", "Number of Hits in HE", 10000, 0., 10000.);
    meHONHit_ = ib.book1D("Hit07", "Number of Hits in HO", 10000, 0., 10000.);
    meHFNHit_ = ib.book1D("Hit08", "Number of Hits in HF", 10000, 0., 10000.);
    meDetectHit_ = ib.book1D("Hit09", "Detector ID", 50, 0., 50.);
    meSubdetHit_ = ib.book1D("Hit10", "Subdetectors in HCal", 50, 0., 50.);
    meDepthHit_ = ib.book1D("Hit11", "Depths in HCal", 20, 0., 20.);
    meEtaHit_ = ib.book1D("Hit12", "Eta in HCal", ieta_bins_HF, ieta_min_HF, ieta_max_HF);
    meEtaPhiHit_ =
        ib.book2D("Hit12b", "Eta-phi in HCal", ieta_bins_HF, ieta_min_HF, ieta_max_HF, iphi_bins, iphi_min, iphi_max);
    for (int depth = 1; depth <= maxDepth_; depth++) {
      sprintf(hname, "Hit12bd%d", depth);
      sprintf(htitle, "Eta-phi in HCal d%d", depth);
      meEtaPhiHitDepth_.push_back(
          ib.book2D(hname, htitle, ieta_bins_HF, ieta_min_HF, ieta_max_HF, iphi_bins, iphi_min, iphi_max));
    }
    // KC: There are different phi segmentation schemes, this plot uses wider
    // bins to represent the most sparse segmentation
    mePhiHit_ = ib.book1D("Hit13", "Phi in HCal (HB,HO)", iphi_bins, iphi_min, iphi_max);
    mePhiHitb_ = ib.book1D("Hit13b", "Phi in HCal (HE,HF)", iphi_bins, iphi_min, iphi_max);
    meEnergyHit_ = ib.book1D("Hit14", "Energy in HCal", 2000, 0., 20.);
    meTimeHit_ = ib.book1D("Hit15", "Time in HCal", 528, 0., 528.);
    meTimeWHit_ = ib.book1D("Hit16", "Time in HCal (E wtd)", 528, 0., 528.);
    meHBDepHit_ = ib.book1D("Hit17", "Depths in HB", 20, 0., 20.);
    meHEDepHit_ = ib.book1D("Hit18", "Depths in HE", 20, 0., 20.);
    meHODepHit_ = ib.book1D("Hit19", "Depths in HO", 20, 0., 20.);
    meHFDepHit_ = ib.book1D("Hit20", "Depths in HF", 20, 0., 20.);
    meHFDepHitw_ = ib.book1D("Hit20b", "Depths in HF (p.e. weighted)", 20, 0., 20.);
    meHBEtaHit_ = ib.book1D("Hit21", "Eta in HB", ieta_bins_HB, ieta_min_HB, ieta_max_HB);
    meHEEtaHit_ = ib.book1D("Hit22", "Eta in HE", ieta_bins_HE, ieta_min_HE, ieta_max_HE);
    meHOEtaHit_ = ib.book1D("Hit23", "Eta in HO", ieta_bins_HO, ieta_min_HO, ieta_max_HO);
    meHFEtaHit_ = ib.book1D("Hit24", "Eta in HF", ieta_bins_HF, ieta_min_HF, ieta_max_HF);
    meHBPhiHit_ = ib.book1D("Hit25", "Phi in HB", iphi_bins, iphi_min, iphi_max);
    meHEPhiHit_ = ib.book1D("Hit26", "Phi in HE", iphi_bins, iphi_min, iphi_max);
    meHOPhiHit_ = ib.book1D("Hit27", "Phi in HO", iphi_bins, iphi_min, iphi_max);
    meHFPhiHit_ = ib.book1D("Hit28", "Phi in HF", iphi_bins, iphi_min, iphi_max);
    meHBEneHit_ = ib.book1D("Hit29", "Energy in HB", 2000, 0., 20.);
    meHEEneHit_ = ib.book1D("Hit30", "Energy in HE", 500, 0., 5.);
    meHEP17EneHit_ = ib.book1D("Hit30b", "Energy in HEP17", 500, 0., 5.);
    meHOEneHit_ = ib.book1D("Hit31", "Energy in HO", 500, 0., 5.);
    meHFEneHit_ = ib.book1D("Hit32", "Energy in HF", 1001, -0.5, 1000.5);

    // HxEneMap, HxEneSum, HxEneSum_vs_ieta plot the sum of the simhits energy
    // within a single ieta-iphi tower.

    meHBEneMap_ =
        ib.book2D("HBEneMap", "HBEneMap", ieta_bins_HB, ieta_min_HB, ieta_max_HB, iphi_bins, iphi_min, iphi_max);
    meHEEneMap_ =
        ib.book2D("HEEneMap", "HEEneMap", ieta_bins_HE, ieta_min_HE, ieta_max_HE, iphi_bins, iphi_min, iphi_max);
    meHOEneMap_ =
        ib.book2D("HOEneMap", "HOEneMap", ieta_bins_HO, ieta_min_HO, ieta_max_HO, iphi_bins, iphi_min, iphi_max);
    meHFEneMap_ =
        ib.book2D("HFEneMap", "HFEneMap", ieta_bins_HF, ieta_min_HF, ieta_max_HF, iphi_bins, iphi_min, iphi_max);

    meHBEneSum_ = ib.book1D("HBEneSum", "HBEneSum", 2000, 0., 20.);
    meHEEneSum_ = ib.book1D("HEEneSum", "HEEneSum", 500, 0., 5.);
    meHOEneSum_ = ib.book1D("HOEneSum", "HOEneSum", 500, 0., 5.);
    meHFEneSum_ = ib.book1D("HFEneSum", "HFEneSum", 1001, -0.5, 1000.5);

    meHBEneSum_vs_ieta_ = ib.bookProfile(
        "HBEneSum_vs_ieta", "HBEneSum_vs_ieta", ieta_bins_HB, ieta_min_HB, ieta_max_HB, 2011, -10.5, 2000.5, " ");
    meHEEneSum_vs_ieta_ = ib.bookProfile(
        "HEEneSum_vs_ieta", "HEEneSum_vs_ieta", ieta_bins_HE, ieta_min_HE, ieta_max_HE, 2011, -10.5, 2000.5, " ");
    meHOEneSum_vs_ieta_ = ib.bookProfile(
        "HOEneSum_vs_ieta", "HOEneSum_vs_ieta", ieta_bins_HO, ieta_min_HO, ieta_max_HO, 2011, -10.5, 2000.5, " ");
    meHFEneSum_vs_ieta_ = ib.bookProfile(
        "HFEneSum_vs_ieta", "HFEneSum_vs_ieta", ieta_bins_HF, ieta_min_HF, ieta_max_HF, 2011, -10.5, 2000.5, " ");

    meHBTimHit_ = ib.book1D("Hit33", "Time in HB", 528, 0., 528.);
    meHETimHit_ = ib.book1D("Hit34", "Time in HE", 528, 0., 528.);
    meHOTimHit_ = ib.book1D("Hit35", "Time in HO", 528, 0., 528.);
    meHFTimHit_ = ib.book1D("Hit36", "Time in HF", 528, 0., 528.);
    // These are the zoomed in energy ranges
    meHBEneHit2_ = ib.book1D("Hit37", "Energy in HB 2", 100, 0., 0.0001);
    meHEEneHit2_ = ib.book1D("Hit38", "Energy in HE 2", 100, 0., 0.0001);
    meHEP17EneHit2_ = ib.book1D("Hit38b", "Energy in HEP17 2", 100, 0., 0.0001);
    meHOEneHit2_ = ib.book1D("Hit39", "Energy in HO 2", 100, 0., 0.0001);
    meHFEneHit2_ = ib.book1D("Hit40", "Energy in HF 2", 100, 0.5, 100.5);
    meHBL10Ene_ = ib.book1D("Hit41", "Log10Energy in HB", 140, -10., 4.);
    meHEL10Ene_ = ib.book1D("Hit42", "Log10Energy in HE", 140, -10., 4.);
    meHFL10Ene_ = ib.book1D("Hit43", "Log10Energy in HF", 50, -1., 4.);
    meHOL10Ene_ = ib.book1D("Hit44", "Log10Energy in HO", 140, -10., 4.);
    meHBL10EneP_ = ib.bookProfile("Hit45", "Log10Energy in HB vs Hit contribution", 140, -10., 4., 100, 0., 1.);
    meHEL10EneP_ = ib.bookProfile("Hit46", "Log10Energy in HE vs Hit contribution", 140, -10., 4., 100, 0., 1.);
    meHFL10EneP_ = ib.bookProfile("Hit47", "Log10Energy in HF vs Hit contribution", 140, -10., 4., 100, 0., 1.);
    meHOL10EneP_ = ib.bookProfile("Hit48", "Log10Energy in HO vs Hit contribution", 140, -10., 4., 100, 0., 1.);
  }
}

/*void HcalSimHitStudy::endJob() {
  if (dbe_ && outFile_.size() > 0) dbe_->save(outFile_);
}*/

void HcalSimHitStudy::analyze(const edm::Event &e, const edm::EventSetup &) {
  LogDebug("HcalSim") << "Run = " << e.id().run() << " Event = " << e.id().event();

  std::vector<PCaloHit> caloHits;
  edm::Handle<edm::PCaloHitContainer> hitsHcal;

  bool getHits = false;
  if (checkHit_) {
    e.getByToken(tok_hits_, hitsHcal);
    if (hitsHcal.isValid())
      getHits = true;
  }

  LogDebug("HcalSim") << "HcalValidation: Input flags Hits " << getHits;

  if (getHits) {
    caloHits.insert(caloHits.end(), hitsHcal->begin(), hitsHcal->end());
    LogDebug("HcalSim") << "HcalValidation: Hit buffer " << caloHits.size();
    analyzeHits(caloHits);
  }
}

void HcalSimHitStudy::analyzeHits(std::vector<PCaloHit> &hits) {
  int nHit = hits.size();
  int nHB = 0, nHE = 0, nHO = 0, nHF = 0, nBad1 = 0, nBad2 = 0, nBad = 0;
  std::vector<double> encontHB(140, 0.);
  std::vector<double> encontHE(140, 0.);
  std::vector<double> encontHF(140, 0.);
  std::vector<double> encontHO(140, 0.);
  double entotHB = 0, entotHE = 0, entotHF = 0, entotHO = 0;

  double HBEneMap[ieta_bins_HB][iphi_bins];
  double HEEneMap[ieta_bins_HE][iphi_bins];
  double HOEneMap[ieta_bins_HO][iphi_bins];
  double HFEneMap[ieta_bins_HF][iphi_bins];

  // Works in ieta_min_Hx is < 0
  int eta_offset_HB = -(int)ieta_min_HB;
  int eta_offset_HE = -(int)ieta_min_HE;
  int eta_offset_HO = -(int)ieta_min_HO;
  int eta_offset_HF = -(int)ieta_min_HF;

  for (int i = 0; i < ieta_bins_HB; i++) {
    for (int j = 0; j < iphi_bins; j++) {
      HBEneMap[i][j] = 0.;
    }
  }

  for (int i = 0; i < ieta_bins_HE; i++) {
    for (int j = 0; j < iphi_bins; j++) {
      HEEneMap[i][j] = 0.;
    }
  }

  for (int i = 0; i < ieta_bins_HO; i++) {
    for (int j = 0; j < iphi_bins; j++) {
      HOEneMap[i][j] = 0.;
    }
  }

  for (int i = 0; i < ieta_bins_HF; i++) {
    for (int j = 0; j < iphi_bins; j++) {
      HFEneMap[i][j] = 0.;
    }
  }

  for (int i = 0; i < nHit; i++) {
    double energy = hits[i].energy();
    double log10en = log10(energy);
    int log10i = int((log10en + 10.) * 10.);
    double time = hits[i].time();
    unsigned int id_ = hits[i].id();
    int det, subdet, depth, eta, phi;
    HcalDetId hid;
    if (testNumber_)
      hid = HcalHitRelabeller::relabel(id_, hcons);
    else
      hid = HcalDetId(id_);
    det = hid.det();
    subdet = hid.subdet();
    depth = hid.depth();
    eta = hid.ieta();
    phi = hid.iphi();

    LogDebug("HcalSim") << "Hit[" << i << "] ID " << std::hex << id_ << std::dec << " Det " << det << " Sub " << subdet
                        << " depth " << depth << " Eta " << eta << " Phi " << phi << " E " << energy << " time "
                        << time;
    if (det == 4) {  // Check DetId.h
      if (subdet == static_cast<int>(HcalBarrel))
        nHB++;
      else if (subdet == static_cast<int>(HcalEndcap))
        nHE++;
      else if (subdet == static_cast<int>(HcalOuter))
        nHO++;
      else if (subdet == static_cast<int>(HcalForward))
        nHF++;
      else {
        nBad++;
        nBad2++;
      }
    } else {
      nBad++;
      nBad1++;
    }

    meDetectHit_->Fill(double(det));
    if (det == 4) {
      meSubdetHit_->Fill(double(subdet));
      meDepthHit_->Fill(double(depth));
      meEtaHit_->Fill(double(eta));
      meEtaPhiHit_->Fill(double(eta), double(phi));
      meEtaPhiHitDepth_[depth - 1]->Fill(double(eta), double(phi));

      // We will group the phi plots by HB,HO and HE,HF since these groups share
      // similar segmentation schemes
      if (subdet == static_cast<int>(HcalBarrel))
        mePhiHit_->Fill(double(phi));
      else if (subdet == static_cast<int>(HcalEndcap))
        mePhiHitb_->Fill(double(phi));
      else if (subdet == static_cast<int>(HcalOuter))
        mePhiHit_->Fill(double(phi));
      else if (subdet == static_cast<int>(HcalForward))
        mePhiHitb_->Fill(double(phi));

      // KC: HF energy is in photoelectrons rather than eV, so it will not be
      // included in total HCal energy
      if (subdet != static_cast<int>(HcalForward)) {
        meEnergyHit_->Fill(energy);

        // Since the HF energy is a different scale it does not make sense to
        // include it in the Energy Weighted Plot
        meTimeWHit_->Fill(double(time), energy);
      }
      meTimeHit_->Fill(time);

      if (subdet == static_cast<int>(HcalBarrel)) {
        meHBDepHit_->Fill(double(depth));
        meHBEtaHit_->Fill(double(eta));
        meHBPhiHit_->Fill(double(phi));
        meHBEneHit_->Fill(energy);
        meHBEneHit2_->Fill(energy);
        meHBTimHit_->Fill(time);
        meHBL10Ene_->Fill(log10en);
        if (log10i >= 0 && log10i < 140)
          encontHB[log10i] += energy;
        entotHB += energy;

        HBEneMap[eta + eta_offset_HB][phi - 1] += energy;

      } else if (subdet == static_cast<int>(HcalEndcap)) {
        meHEDepHit_->Fill(double(depth));
        meHEEtaHit_->Fill(double(eta));
        meHEPhiHit_->Fill(double(phi));

        bool isHEP17 = (phi >= 63) && (phi <= 66) && (eta > 0);
        if (hep17_) {
          if (!isHEP17) {
            meHEEneHit_->Fill(energy);
            meHEEneHit2_->Fill(energy);
          } else {
            meHEP17EneHit_->Fill(energy);
            meHEP17EneHit2_->Fill(energy);
          }
        } else {
          meHEEneHit_->Fill(energy);
          meHEEneHit2_->Fill(energy);
        }

        meHETimHit_->Fill(time);
        meHEL10Ene_->Fill(log10en);
        if (log10i >= 0 && log10i < 140)
          encontHE[log10i] += energy;
        entotHE += energy;

        HEEneMap[eta + eta_offset_HE][phi - 1] += energy;

      } else if (subdet == static_cast<int>(HcalOuter)) {
        meHODepHit_->Fill(double(depth));
        meHOEtaHit_->Fill(double(eta));
        meHOPhiHit_->Fill(double(phi));
        meHOEneHit_->Fill(energy);
        meHOEneHit2_->Fill(energy);
        meHOTimHit_->Fill(time);
        meHOL10Ene_->Fill(log10en);
        if (log10i >= 0 && log10i < 140)
          encontHO[log10i] += energy;
        entotHO += energy;

        HOEneMap[eta + eta_offset_HO][phi - 1] += energy;

      } else if (subdet == static_cast<int>(HcalForward)) {
        meHFDepHit_->Fill(double(depth));
        meHFDepHitw_->Fill(double(depth), energy);
        meHFEtaHit_->Fill(double(eta));
        meHFPhiHit_->Fill(double(phi));
        meHFEneHit_->Fill(energy);
        meHFEneHit2_->Fill(energy);
        meHFTimHit_->Fill(time);
        meHFL10Ene_->Fill(log10en);
        if (log10i >= 0 && log10i < 140)
          encontHF[log10i] += energy;
        entotHF += energy;

        HFEneMap[eta + eta_offset_HF][phi - 1] += energy;
      }
    }
  }
  if (entotHB != 0)
    for (int i = 0; i < 140; i++)
      meHBL10EneP_->Fill(-10. + (float(i) + 0.5) / 10., encontHB[i] / entotHB);
  if (entotHE != 0)
    for (int i = 0; i < 140; i++)
      meHEL10EneP_->Fill(-10. + (float(i) + 0.5) / 10., encontHE[i] / entotHE);
  if (entotHF != 0)
    for (int i = 0; i < 140; i++)
      meHFL10EneP_->Fill(-10. + (float(i) + 0.5) / 10., encontHF[i] / entotHF);
  if (entotHO != 0)
    for (int i = 0; i < 140; i++)
      meHOL10EneP_->Fill(-10. + (float(i) + 0.5) / 10., encontHO[i] / entotHO);

  meAllNHit_->Fill(double(nHit));
  meBadDetHit_->Fill(double(nBad1));
  meBadSubHit_->Fill(double(nBad2));
  meBadIdHit_->Fill(double(nBad));
  meHBNHit_->Fill(double(nHB));
  meHENHit_->Fill(double(nHE));
  meHONHit_->Fill(double(nHO));
  meHFNHit_->Fill(double(nHF));

  for (int i = 0; i < ieta_bins_HB; i++) {
    for (int j = 0; j < iphi_bins; j++) {
      if (HBEneMap[i][j] != 0) {
        meHBEneSum_->Fill(HBEneMap[i][j]);
        meHBEneSum_vs_ieta_->Fill((i - eta_offset_HB), HBEneMap[i][j]);
        meHBEneMap_->Fill((i - eta_offset_HB), j + 1, HBEneMap[i][j]);
      }
    }
  }

  for (int i = 0; i < ieta_bins_HE; i++) {
    for (int j = 0; j < iphi_bins; j++) {
      if (HEEneMap[i][j] != 0) {
        meHEEneSum_->Fill(HEEneMap[i][j]);
        meHEEneSum_vs_ieta_->Fill((i - eta_offset_HE), HEEneMap[i][j]);
        meHEEneMap_->Fill((i - eta_offset_HE), j + 1, HEEneMap[i][j]);
      }
    }
  }

  for (int i = 0; i < ieta_bins_HO; i++) {
    for (int j = 0; j < iphi_bins; j++) {
      if (HOEneMap[i][j] != 0) {
        meHOEneSum_->Fill(HOEneMap[i][j]);
        meHOEneSum_vs_ieta_->Fill((i - eta_offset_HO), HOEneMap[i][j]);
        meHOEneMap_->Fill((i - eta_offset_HO), j + 1, HOEneMap[i][j]);
      }
    }
  }

  for (int i = 0; i < ieta_bins_HF; i++) {
    for (int j = 0; j < iphi_bins; j++) {
      if (HFEneMap[i][j] != 0) {
        meHFEneSum_->Fill(HFEneMap[i][j]);
        meHFEneSum_vs_ieta_->Fill((i - eta_offset_HF), HFEneMap[i][j]);
        meHFEneMap_->Fill((i - eta_offset_HF), j + 1, HFEneMap[i][j]);
      }
    }
  }

  LogDebug("HcalSim") << "HcalSimHitStudy::analyzeHits: HB " << nHB << " HE " << nHE << " HO " << nHO << " HF " << nHF
                      << " Bad " << nBad << " All " << nHit;
}
