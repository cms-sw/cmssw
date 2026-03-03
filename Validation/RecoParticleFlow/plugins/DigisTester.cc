#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

class DigisTester : public DQMEDAnalyzer {
public:
  explicit DigisTester(const edm::ParameterSet&);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometry_token_;
  edm::EDGetTokenT<EBDigiCollection> ecalEBDigisToken_;
  edm::EDGetTokenT<EEDigiCollection> ecalEEDigisToken_;

  std::string outFolder_;

  using U2Map = std::unordered_map<std::string, MonitorElement*>;
  U2Map h2d_ebdigis_, h2d_eedigis_;
};

DigisTester::DigisTester(const edm::ParameterSet& iConfig)
    : geometry_token_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      ecalEBDigisToken_(consumes<EBDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalEBDigis"))),
      ecalEEDigisToken_(consumes<EEDigiCollection>(iConfig.getParameter<edm::InputTag>("ecalEEDigis"))),
      outFolder_(iConfig.getParameter<std::string>("outFolder")) {}

void DigisTester::bookHistograms(DQMStore::IBooker& ibook, edm::Run const&, edm::EventSetup const&) {
  ibook.setCurrentFolder(outFolder_ + "/Digis");

  unsigned mNBinsADC = 100;
  float mADCMin = 180.;
  float mADCMax = 500.;
  unsigned mNBinsPhi = 100;
  float mPhiMax = 3.2;
  unsigned mNBinsEta = 100;
  float mEtaMax = 3.3;

  std::unordered_map<std::string, std::tuple<unsigned, float, float, unsigned, float, float>> ebVars = {
      {"ADC_Eta", std::make_tuple(mNBinsADC, mADCMin, mADCMax, mNBinsEta, -mEtaMax, mEtaMax)},
      {"ADC_Phi", std::make_tuple(mNBinsADC, mADCMin, mADCMax, mNBinsPhi, -mPhiMax, mPhiMax)},
      {"Eta_Phi", std::make_tuple(mNBinsEta, -mEtaMax, mEtaMax, mNBinsPhi, -mPhiMax, mPhiMax)},
  };

  for (auto& ebVar : ebVars) {
    auto [nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY] = ebVar.second;
    auto x_title = ebVar.first.substr(0, ebVar.first.find("_"));
    auto y_title = ebVar.first.substr(ebVar.first.find("_") + 1);
    h2d_ebdigis_[ebVar.first] = ibook.book2D("EcalEBDigis" + ebVar.first,
                                             "EcalEBDigis;" + x_title + ";" + y_title,
                                             nBinsX,
                                             hMinX,
                                             hMaxX,
                                             nBinsY,
                                             hMinY,
                                             hMaxY);
  }

  std::unordered_map<std::string, std::tuple<unsigned, float, float, unsigned, float, float>> eeVars = {
      {"ADC_Eta", std::make_tuple(mNBinsADC, mADCMin, mADCMax, mNBinsEta, -mEtaMax, mEtaMax)},
      {"ADC_Phi", std::make_tuple(mNBinsADC, mADCMin, mADCMax, mNBinsPhi, -mPhiMax, mPhiMax)},
      {"Eta_Phi", std::make_tuple(mNBinsEta, -mEtaMax, mEtaMax, mNBinsPhi, -mPhiMax, mPhiMax)},
  };

  for (auto& eeVar : eeVars) {
    auto [nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY] = eeVar.second;
    auto x_title = eeVar.first.substr(0, eeVar.first.find("_"));
    auto y_title = eeVar.first.substr(eeVar.first.find("_") + 1);
    h2d_eedigis_[eeVar.first] = ibook.book2D("EcalEEDigis" + eeVar.first,
                                             "EcalEEDigis;" + x_title + ";" + y_title,
                                             nBinsX,
                                             hMinX,
                                             hMaxX,
                                             nBinsY,
                                             hMinY,
                                             hMaxY);
  }
}

void DigisTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const CaloGeometry& caloGeom = iSetup.getData(geometry_token_);
  // const auto& barrelGeom = caloGeom.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  // const auto& endcapGeom = caloGeom.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

  edm::Handle<EBDigiCollection> ebDigisHandle;
  iEvent.getByToken(ecalEBDigisToken_, ebDigisHandle);
  if (!ebDigisHandle.isValid()) {
    edm::LogPrint("DigisTester") << "Input EB Digis collection not found.";
    return;
  }

  edm::Handle<EEDigiCollection> eeDigisHandle;
  iEvent.getByToken(ecalEEDigisToken_, eeDigisHandle);
  if (!eeDigisHandle.isValid()) {
    edm::LogPrint("DigisTester") << "Input EE Digis collection not found.";
    return;
  }

  for (const edm::DataFrame& digi : *ebDigisHandle) {
    float eta = caloGeom.getPosition(digi.id()).eta();
    float phi = caloGeom.getPosition(digi.id()).phi();

    // Get the number of ADC samples (usually 10)
    int nADCSamples = digi.size();

    float adcMax = 0;
    for (int i = 0; i < nADCSamples; ++i) {
      // compute max of adc counts of all channels
      float adcVal = static_cast<EBDataFrame>(digi).sample(i).adc();
      if (adcVal > adcMax)
        adcMax = adcVal;
    }

    h2d_ebdigis_["ADC_Eta"]->Fill(adcMax, eta);
    h2d_ebdigis_["ADC_Phi"]->Fill(adcMax, phi);
    h2d_ebdigis_["Eta_Phi"]->Fill(eta, phi);
  }

  for (const edm::DataFrame& digi : *eeDigisHandle) {
    float eta = caloGeom.getPosition(digi.id()).eta();
    float phi = caloGeom.getPosition(digi.id()).phi();

    // Get the number of ADC samples (usually 10)
    int nADCSamples = digi.size();

    float adcMax = 0;
    for (int i = 0; i < nADCSamples; ++i) {
      // compute max of adc counts of all channels
      float adcVal = static_cast<EEDataFrame>(digi).sample(i).adc();
      if (adcVal > adcMax)
        adcMax = adcVal;
    }

    h2d_eedigis_["ADC_Eta"]->Fill(adcMax, eta);
    h2d_eedigis_["ADC_Phi"]->Fill(adcMax, phi);
    h2d_eedigis_["Eta_Phi"]->Fill(eta, phi);
  }
}

DEFINE_FWK_MODULE(DigisTester);
