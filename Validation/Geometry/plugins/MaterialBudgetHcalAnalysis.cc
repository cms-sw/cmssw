#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingCalo.h"

#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TProfile2D.h>

#include <string>
#include <vector>

using namespace geant_units::operators;

class MaterialBudgetHcalAnalysis : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  MaterialBudgetHcalAnalysis(const edm::ParameterSet &p);
  ~MaterialBudgetHcalAnalysis() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void beginJob() override;

  static const uint32_t maxSet_ = 25, maxSet2_ = 9;
  const int binEta_, binPhi_;
  const double maxEta_, etaLow_, etaHigh_, etaLowMin_, etaLowMax_, etaMidMin_;
  const double etaMidMax_, etaHighMin_, etaHighMax_, etaMinP_, etaMaxP_;
  const edm::InputTag labelMBCalo_;
  const edm::EDGetTokenT<MaterialAccountingCaloCollection> tokMBCalo_;
  TH1F *me400_[maxSet_], *me800_[maxSet_], *me1300_[maxSet2_];
  TH2F *me1200_[maxSet_], *me1400_[maxSet2_];
  TProfile *me100_[maxSet_], *me200_[maxSet_], *me300_[maxSet_];
  TProfile *me500_[maxSet_], *me600_[maxSet_], *me700_[maxSet_];
  TProfile *me1500_[maxSet2_];
  TProfile *me1600_[maxSet_], *me1700_[maxSet_], *me1800_[maxSet_];
  TProfile *me1900_[maxSet_], *me2000_[maxSet_], *me2100_[maxSet_];
  TProfile *me2200_[maxSet_], *me2300_[maxSet_], *me2400_[maxSet_];
  TProfile2D *me900_[maxSet_], *me1000_[maxSet_], *me1100_[maxSet_];
};

MaterialBudgetHcalAnalysis::MaterialBudgetHcalAnalysis(const edm::ParameterSet &p)
    : binEta_(p.getParameter<int>("nBinEta")),
      binPhi_(p.getParameter<int>("nBinPhi")),
      maxEta_(p.getParameter<double>("maxEta")),
      etaLow_(p.getParameter<double>("etaLow")),
      etaHigh_(p.getParameter<double>("etaHigh")),
      etaLowMin_(p.getParameter<double>("etaLowMin")),
      etaLowMax_(p.getParameter<double>("etaLowMax")),
      etaMidMin_(p.getParameter<double>("etaMidMin")),
      etaMidMax_(p.getParameter<double>("etaMidMax")),
      etaHighMin_(p.getParameter<double>("etaHighMin")),
      etaHighMax_(p.getParameter<double>("etaHighMax")),
      etaMinP_(p.getParameter<double>("etaMinP")),
      etaMaxP_(p.getParameter<double>("etaMaxP")),
      labelMBCalo_(p.getParameter<edm::InputTag>("labelMBCaloLabel")),
      tokMBCalo_(consumes<MaterialAccountingCaloCollection>(labelMBCalo_)) {
  usesResource(TFileService::kSharedResource);
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalAnalysis: == Eta plot: NX " << binEta_ << " Range "
                                     << -maxEta_ << ":" << maxEta_ << " Phi plot: NX " << binPhi_ << " Range " << -1._pi
                                     << ":" << 1._pi << " (Eta limit " << etaLow_ << ":" << etaHigh_ << ")"
                                     << " Eta range (" << etaLowMin_ << ":" << etaLowMax_ << "), (" << etaMidMin_ << ":"
                                     << etaMidMax_ << "), (" << etaHighMin_ << ":" << etaHighMax_
                                     << ") Debug for eta range " << etaMinP_ << ":" << etaMaxP_;
}

void MaterialBudgetHcalAnalysis::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("nBinEta", 260);
  desc.add<int>("nBinPhi", 180);
  desc.add<double>("maxEta", 5.2);
  desc.add<double>("etaLow", -5.2);
  desc.add<double>("etaHigh", 5.2);
  desc.add<double>("etaMinP", 5.2);
  desc.add<double>("etaMaxP", 0.0);
  desc.add<double>("etaLowMin", 0.783);
  desc.add<double>("etaLowMax", 0.870);
  desc.add<double>("etaMidMin", 2.650);
  desc.add<double>("etaMidMax", 2.868);
  desc.add<double>("etaHighMin", 2.868);
  desc.add<double>("etaHighMax", 3.000);
  desc.add<edm::InputTag>("labelMBCaloLabel", edm::InputTag("g4SimHits", "HcalMatBCalo"));
  descriptions.add("materialBudgetHcalAnalysis", desc);
}

void MaterialBudgetHcalAnalysis::beginJob() {
  // Book histograms
  edm::Service<TFileService> tfile;

  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";

  double maxPhi = 1._pi;
  edm::LogVerbatim("MaterialBudgetFull") << "MaterialBudgetHcalAnalysis: Booking user histos === with " << binEta_
                                         << " bins in eta from " << -maxEta_ << " to " << maxEta_ << " and " << binPhi_
                                         << " bins in phi from " << -maxPhi << " to " << maxPhi;

  std::string iter;
  std::string range0 = "(" + std::to_string(etaMidMin_) + ":" + std::to_string(etaMidMax_) + ") ";
  std::string range1 = "(" + std::to_string(etaHighMin_) + ":" + std::to_string(etaHighMax_) + ") ";
  std::string range2 = "(" + std::to_string(etaLowMin_) + ":" + std::to_string(etaLowMax_) + ") ";
  // total X0
  for (uint32_t i = 0; i < maxSet_; i++) {
    iter = std::to_string(i);
    me100_[i] = tfile->make<TProfile>(
        std::to_string(i + 100).c_str(), ("MB(X0) prof Eta in region " + iter).c_str(), binEta_, -maxEta_, maxEta_);
    me200_[i] = tfile->make<TProfile>(
        std::to_string(i + 200).c_str(), ("MB(L0) prof Eta in region " + iter).c_str(), binEta_, -maxEta_, maxEta_);
    me300_[i] = tfile->make<TProfile>(
        std::to_string(i + 300).c_str(), ("MB(Step) prof Eta in region " + iter).c_str(), binEta_, -maxEta_, maxEta_);
    me400_[i] = tfile->make<TH1F>(
        std::to_string(i + 400).c_str(), ("Eta in region " + iter).c_str(), binEta_, -maxEta_, maxEta_);
    me500_[i] = tfile->make<TProfile>(
        std::to_string(i + 500).c_str(), ("MB(X0) prof Ph in region " + iter).c_str(), binPhi_, -maxPhi, maxPhi);
    me600_[i] = tfile->make<TProfile>(
        std::to_string(i + 600).c_str(), ("MB(L0) prof Ph in region " + iter).c_str(), binPhi_, -maxPhi, maxPhi);
    me700_[i] = tfile->make<TProfile>(
        std::to_string(i + 700).c_str(), ("MB(Step) prof Ph in region " + iter).c_str(), binPhi_, -maxPhi, maxPhi);
    me800_[i] =
        tfile->make<TH1F>(std::to_string(i + 800).c_str(), ("Phi in region " + iter).c_str(), binPhi_, -maxPhi, maxPhi);
    me900_[i] = tfile->make<TProfile2D>(std::to_string(i + 900).c_str(),
                                        ("MB(X0) prof Eta Phi in region " + iter).c_str(),
                                        binEta_ / 2,
                                        -maxEta_,
                                        maxEta_,
                                        binPhi_ / 2,
                                        -maxPhi,
                                        maxPhi);
    me1000_[i] = tfile->make<TProfile2D>(std::to_string(i + 1000).c_str(),
                                         ("MB(L0) prof Eta Phi in region " + iter).c_str(),
                                         binEta_ / 2,
                                         -maxEta_,
                                         maxEta_,
                                         binPhi_ / 2,
                                         -maxPhi,
                                         maxPhi);
    me1100_[i] = tfile->make<TProfile2D>(std::to_string(i + 1100).c_str(),
                                         ("MB(Step) prof Eta Phi in region " + iter).c_str(),
                                         binEta_ / 2,
                                         -maxEta_,
                                         maxEta_,
                                         binPhi_ / 2,
                                         -maxPhi,
                                         maxPhi);
    me1200_[i] = tfile->make<TH2F>(std::to_string(i + 1200).c_str(),
                                   ("Eta vs Phi in region " + iter).c_str(),
                                   binEta_ / 2,
                                   -maxEta_,
                                   maxEta_,
                                   binPhi_ / 2,
                                   -maxPhi,
                                   maxPhi);
    me1600_[i] = tfile->make<TProfile>(std::to_string(i + 1600).c_str(),
                                       ("MB(X0) prof Ph in region " + range0 + iter).c_str(),
                                       binPhi_,
                                       -maxPhi,
                                       maxPhi);
    me1700_[i] = tfile->make<TProfile>(std::to_string(i + 1700).c_str(),
                                       ("MB(L0) prof Ph in region " + range0 + iter).c_str(),
                                       binPhi_,
                                       -maxPhi,
                                       maxPhi);
    me1800_[i] = tfile->make<TProfile>(std::to_string(i + 1800).c_str(),
                                       ("MB(Step) prof Ph in region " + range0 + iter).c_str(),
                                       binPhi_,
                                       -maxPhi,
                                       maxPhi);
    me1900_[i] = tfile->make<TProfile>(std::to_string(i + 1900).c_str(),
                                       ("MB(X0) prof Ph in region " + range1 + iter).c_str(),
                                       binPhi_,
                                       -maxPhi,
                                       maxPhi);
    me2000_[i] = tfile->make<TProfile>(std::to_string(i + 2000).c_str(),
                                       ("MB(L0) prof Ph in region " + range1 + iter).c_str(),
                                       binPhi_,
                                       -maxPhi,
                                       maxPhi);
    me2100_[i] = tfile->make<TProfile>(std::to_string(i + 2100).c_str(),
                                       ("MB(Step) prof Ph in region " + range1 + iter).c_str(),
                                       binPhi_,
                                       -maxPhi,
                                       maxPhi);
    me2200_[i] = tfile->make<TProfile>(std::to_string(i + 2200).c_str(),
                                       ("MB(X0) prof Ph in region " + range2 + iter).c_str(),
                                       binPhi_,
                                       -maxPhi,
                                       maxPhi);
    me2300_[i] = tfile->make<TProfile>(std::to_string(i + 2300).c_str(),
                                       ("MB(L0) prof Ph in region " + range2 + iter).c_str(),
                                       binPhi_,
                                       -maxPhi,
                                       maxPhi);
    me2400_[i] = tfile->make<TProfile>(std::to_string(i + 2400).c_str(),
                                       ("MB(Step) prof Ph in region " + range2 + iter).c_str(),
                                       binPhi_,
                                       -maxPhi,
                                       maxPhi);
  }
  for (uint32_t i = 0; i < maxSet2_; i++) {
    iter = std::to_string(i);
    me1300_[i] = tfile->make<TH1F>(std::to_string(i + 1300).c_str(),
                                   ("Events with layers Hit (0 all, 1 HB, ..) for " + iter).c_str(),
                                   binEta_,
                                   -maxEta_,
                                   maxEta_);
    me1400_[i] = tfile->make<TH2F>(std::to_string(i + 1400).c_str(),
                                   ("Eta vs Phi for layers hit in " + iter).c_str(),
                                   binEta_ / 2,
                                   -maxEta_,
                                   maxEta_,
                                   binPhi_ / 2,
                                   -maxPhi,
                                   maxPhi);
    me1500_[i] = tfile->make<TProfile>(std::to_string(i + 1500).c_str(),
                                       ("Number of layers crossed (0 all, 1 HB, ..) for " + iter).c_str(),
                                       binEta_,
                                       -maxEta_,
                                       maxEta_);
  }

  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetHcalAnalysis: Booking user histos done ===";
}

void MaterialBudgetHcalAnalysis::analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MaterialBudgetFull") << "Run " << iEvent.id().run() << " Event " << iEvent.id().event()
                                         << " Luminosity " << iEvent.luminosityBlock() << " Bunch "
                                         << iEvent.bunchCrossing();
#endif

  // Fill from the MB collection
  auto const &hcalMBColl = iEvent.getHandle(tokMBCalo_);
  if (hcalMBColl.isValid()) {
    auto hcalMB = hcalMBColl.product();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MaterialBudgetFull")
        << "Finds HcalMaterialBudgetCollection with " << hcalMB->size() << " entries";
#endif

    for (auto itr = hcalMB->begin(); itr != hcalMB->end(); ++itr) {
      for (uint32_t ii = 0; ii < itr->m_stepLen.size(); ++ii) {
#ifdef EDM_ML_DEBUG
        if ((std::abs(itr->m_eta) >= etaMinP_) && (std::abs(itr->m_eta) <= etaMaxP_))
          edm::LogVerbatim("MaterialBudget")
              << "MaterialBudgetHcalAnalysis:FillHisto called with index " << ii << " integrated  step "
              << itr->m_stepLen[ii] << " X0 " << itr->m_radLen[ii] << " Lamda " << itr->m_intLen[ii];
#endif
        if (ii < maxSet_) {
          me100_[ii]->Fill(itr->m_eta, itr->m_radLen[ii]);
          me200_[ii]->Fill(itr->m_eta, itr->m_intLen[ii]);
          me300_[ii]->Fill(itr->m_eta, itr->m_stepLen[ii]);
          me400_[ii]->Fill(itr->m_eta);

          if (itr->m_eta >= etaLow_ && itr->m_eta <= etaHigh_) {
            me500_[ii]->Fill(itr->m_phi, itr->m_radLen[ii]);
            me600_[ii]->Fill(itr->m_phi, itr->m_intLen[ii]);
            me700_[ii]->Fill(itr->m_phi, itr->m_stepLen[ii]);
            me800_[ii]->Fill(itr->m_phi);
          }

          me900_[ii]->Fill(itr->m_eta, itr->m_phi, itr->m_radLen[ii]);
          me1000_[ii]->Fill(itr->m_eta, itr->m_phi, itr->m_intLen[ii]);
          me1100_[ii]->Fill(itr->m_eta, itr->m_phi, itr->m_stepLen[ii]);
          me1200_[ii]->Fill(itr->m_eta, itr->m_phi);

          if ((std::abs(itr->m_eta) >= etaMidMin_) && (std::abs(itr->m_eta) <= etaMidMax_)) {
            me1600_[ii]->Fill(itr->m_phi, itr->m_radLen[ii]);
            me1700_[ii]->Fill(itr->m_phi, itr->m_intLen[ii]);
            me1800_[ii]->Fill(itr->m_phi, itr->m_stepLen[ii]);
          }

          if ((std::abs(itr->m_eta) >= etaHighMin_) && (std::abs(itr->m_eta) <= etaHighMax_)) {
            me1900_[ii]->Fill(itr->m_phi, itr->m_radLen[ii]);
            me2000_[ii]->Fill(itr->m_phi, itr->m_intLen[ii]);
            me2100_[ii]->Fill(itr->m_phi, itr->m_stepLen[ii]);
          }

          if ((std::abs(itr->m_eta) >= etaLowMin_) && (std::abs(itr->m_eta) <= etaLowMax_)) {
            me2200_[ii]->Fill(itr->m_phi, itr->m_radLen[ii]);
            me2300_[ii]->Fill(itr->m_phi, itr->m_intLen[ii]);
            me2400_[ii]->Fill(itr->m_phi, itr->m_stepLen[ii]);
          }
        }
      }

      me1300_[0]->Fill(itr->m_eta);
      me1400_[0]->Fill(itr->m_eta, itr->m_phi);
      if (itr->m_layers[0] > 0) {
        me1300_[1]->Fill(itr->m_eta);
        me1400_[1]->Fill(itr->m_eta, itr->m_phi);
      }
      if (itr->m_layers[0] >= 16) {
        me1300_[2]->Fill(itr->m_eta);
        me1400_[2]->Fill(itr->m_eta, itr->m_phi);
      }
      if (itr->m_layers[1] > 0) {
        me1300_[3]->Fill(itr->m_eta);
        me1400_[3]->Fill(itr->m_eta, itr->m_phi);
      }
      if (itr->m_layers[1] >= 16) {
        me1300_[4]->Fill(itr->m_eta);
        me1400_[4]->Fill(itr->m_eta, itr->m_phi);
      }
      if (itr->m_layers[2] > 0) {
        me1300_[5]->Fill(itr->m_eta);
        me1400_[5]->Fill(itr->m_eta, itr->m_phi);
      }
      if (itr->m_layers[2] >= 2) {
        me1300_[6]->Fill(itr->m_eta);
        me1400_[6]->Fill(itr->m_eta, itr->m_phi);
      }
      if (itr->m_layers[3] > 0) {
        me1300_[7]->Fill(itr->m_eta);
        me1400_[7]->Fill(itr->m_eta, itr->m_phi);
      }
      if (itr->m_layers[0] > 0 || itr->m_layers[1] > 0 || (itr->m_layers[3] > 0 && std::abs(itr->m_eta) > 3.0)) {
        me1300_[8]->Fill(itr->m_eta);
        me1400_[8]->Fill(itr->m_eta, itr->m_phi);
      }
      me1500_[0]->Fill(itr->m_eta, (double)(itr->m_layers[0] + itr->m_layers[1] + itr->m_layers[2] + itr->m_layers[3]));
      me1500_[1]->Fill(itr->m_eta, (double)(itr->m_layers[0]));
      me1500_[2]->Fill(itr->m_eta, (double)(itr->m_layers[1]));
      me1500_[4]->Fill(itr->m_eta, (double)(itr->m_layers[3]));
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(MaterialBudgetHcalAnalysis);
