// system include files
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "TH2D.h"
#include "TH1D.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

class HGCalDigiStudy : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCalDigiStudy(const edm::ParameterSet&);
  ~HGCalDigiStudy() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override;
  void endJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  struct digiInfo {
    digiInfo() {
      phi = eta = r = z = charge = 0.0;
      layer = adc = 0;
    }
    double phi, eta, r, z, charge;
    int layer, adc;
  };

  template <class T1, class T2>
  void digiValidation(const T1& detId, const T2* geom, int layer, uint16_t adc, double charge);

  // ----------member data ---------------------------
  const std::string nameDetector_;
  const edm::InputTag source_;
  const bool ifNose_, ifLayer_;
  const int verbosity_, SampleIndx_, nbinR_, nbinZ_, nbinEta_, nLayers_;
  const double rmin_, rmax_, zmin_, zmax_, etamin_, etamax_;
  const edm::EDGetTokenT<HGCalDigiCollection> digiSource_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcGeom_;
  const HGCalGeometry* hgcGeom_;
  int layers_, layerFront_, geomType_;
  TH1D *h_Charge_, *h_ADC_, *h_LayZp_, *h_LayZm_;
  TH2D *h_RZ_, *h_EtaPhi_;
  std::vector<TH2D*> h_XY_;
  TH1D *h_W1_, *h_W2_, *h_C1_, *h_C2_, *h_Ly_;
};

HGCalDigiStudy::HGCalDigiStudy(const edm::ParameterSet& iConfig)
    : nameDetector_(iConfig.getParameter<std::string>("detectorName")),
      source_(iConfig.getParameter<edm::InputTag>("digiSource")),
      ifNose_(iConfig.getUntrackedParameter<bool>("ifNose", false)),
      ifLayer_(iConfig.getUntrackedParameter<bool>("ifLayer", false)),
      verbosity_(iConfig.getUntrackedParameter<int>("verbosity", 0)),
      SampleIndx_(iConfig.getUntrackedParameter<int>("sampleIndex", 5)),
      nbinR_(iConfig.getUntrackedParameter<int>("nBinR", 300)),
      nbinZ_(iConfig.getUntrackedParameter<int>("nBinZ", 300)),
      nbinEta_(iConfig.getUntrackedParameter<int>("nBinEta", 200)),
      nLayers_(iConfig.getUntrackedParameter<int>("layers", 28)),
      rmin_(iConfig.getUntrackedParameter<double>("rMin", 0.0)),
      rmax_(iConfig.getUntrackedParameter<double>("rMax", 300.0)),
      zmin_(iConfig.getUntrackedParameter<double>("zMin", 300.0)),
      zmax_(iConfig.getUntrackedParameter<double>("zMax", 600.0)),
      etamin_(iConfig.getUntrackedParameter<double>("etaMin", 1.0)),
      etamax_(iConfig.getUntrackedParameter<double>("etaMax", 3.0)),
      digiSource_(consumes<HGCalDigiCollection>(source_)),
      tok_hgcGeom_(esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})) {
  usesResource(TFileService::kSharedResource);
  edm::LogVerbatim("HGCalValidation") << "HGCalDigiStudy: request for Digi collection " << source_ << " for "
                                      << nameDetector_;
}

void HGCalDigiStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detectorName", "HGCalEESensitive");
  desc.add<edm::InputTag>("digiSource", edm::InputTag("simHGCalUnsuppressedDigis", "EE"));
  desc.addUntracked<bool>("ifNose", false);
  desc.addUntracked<bool>("ifLayer", false);
  desc.addUntracked<int>("verbosity", 0);
  desc.addUntracked<int>("sampleIndex", 0);
  desc.addUntracked<double>("rMin", 0.0);
  desc.addUntracked<double>("rMax", 300.0);
  desc.addUntracked<double>("zMin", 300.0);
  desc.addUntracked<double>("zMax", 600.0);
  desc.addUntracked<double>("etaMin", 1.0);
  desc.addUntracked<double>("etaMax", 3.0);
  desc.addUntracked<int>("nBinR", 300);
  desc.addUntracked<int>("nBinZ", 300);
  desc.addUntracked<int>("nBinEta", 200);
  desc.addUntracked<int>("layers", 28);
  descriptions.add("hgcalDigiStudyEE", desc);
}

void HGCalDigiStudy::beginJob() {
  edm::Service<TFileService> fs;
  std::ostringstream hname, title;
  hname.str("");
  title.str("");
  hname << "RZ_" << nameDetector_;
  title << "R vs Z for " << nameDetector_;
  h_RZ_ = fs->make<TH2D>(hname.str().c_str(), title.str().c_str(), nbinZ_, zmin_, zmax_, nbinR_, rmin_, rmax_);
  if (ifLayer_) {
    for (int ly = 0; ly < nLayers_; ++ly) {
      hname.str("");
      title.str("");
      hname << "XY_L" << (ly + 1);
      title << "Y vs X at Layer " << (ly + 1);
      h_XY_.emplace_back(
          fs->make<TH2D>(hname.str().c_str(), title.str().c_str(), nbinR_, -rmax_, rmax_, nbinR_, -rmax_, rmax_));
    }
  } else {
    hname.str("");
    title.str("");
    hname << "EtaPhi_" << nameDetector_;
    title << "#phi vs #eta for " << nameDetector_;
    h_EtaPhi_ = fs->make<TH2D>(hname.str().c_str(), title.str().c_str(), nbinEta_, etamin_, etamax_, 200, -M_PI, M_PI);
  }
  hname.str("");
  title.str("");
  hname << "Charge_" << nameDetector_;
  title << "Charge for " << nameDetector_;
  h_Charge_ = fs->make<TH1D>(hname.str().c_str(), title.str().c_str(), 100, -25, 25);
  hname.str("");
  title.str("");
  hname << "ADC_" << nameDetector_;
  title << "ADC for " << nameDetector_;
  h_ADC_ = fs->make<TH1D>(hname.str().c_str(), title.str().c_str(), 200, 0, 50);
  hname.str("");
  title.str("");
  hname << "LayerZp_" << nameDetector_;
  title << "Charge vs Layer (+z) for " << nameDetector_;
  h_LayZp_ = fs->make<TH1D>(hname.str().c_str(), title.str().c_str(), 60, 0.0, 60.0);
  hname.str("");
  title.str("");
  hname << "LayerZm_" << nameDetector_;
  title << "Charge vs Layer (-z) for " << nameDetector_;
  h_LayZm_ = fs->make<TH1D>(hname.str().c_str(), title.str().c_str(), 60, 0.0, 60.0);
  hname.str("");
  title.str("");
  hname << "LY_" << nameDetector_;
  title << "Layer number for " << nameDetector_;
  h_Ly_ = fs->make<TH1D>(hname.str().c_str(), title.str().c_str(), 200, 0, 100);
  if (nameDetector_ == "HGCalHEScintillatorSensitive") {
    hname.str("");
    title.str("");
    hname << "IR_" << nameDetector_;
    title << "Radius index for " << nameDetector_;
    h_W1_ = fs->make<TH1D>(hname.str().c_str(), title.str().c_str(), 200, -50, 50);
    hname.str("");
    title.str("");
    hname << "FI_" << nameDetector_;
    title << "#phi index for " << nameDetector_;
    h_C1_ = fs->make<TH1D>(hname.str().c_str(), title.str().c_str(), 720, 0, 360);
  } else {
    hname.str("");
    title.str("");
    hname << "WU_" << nameDetector_;
    title << "u index of wafers for " << nameDetector_;
    h_W1_ = fs->make<TH1D>(hname.str().c_str(), title.str().c_str(), 200, -50, 50);
    hname.str("");
    title.str("");
    hname << "WV_" << nameDetector_;
    title << "v index of wafers for " << nameDetector_;
    h_W2_ = fs->make<TH1D>(hname.str().c_str(), title.str().c_str(), 100, -50, 50);
    hname.str("");
    title.str("");
    hname << "CU_" << nameDetector_;
    title << "u index of cells for " << nameDetector_;
    h_C1_ = fs->make<TH1D>(hname.str().c_str(), title.str().c_str(), 100, 0, 50);
    hname.str("");
    title.str("");
    hname << "CV_" << nameDetector_;
    title << "v index of cells for " << nameDetector_;
    h_C2_ = fs->make<TH1D>(hname.str().c_str(), title.str().c_str(), 100, 0, 50);
  }
}

void HGCalDigiStudy::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  edm::ESHandle<HGCalGeometry> geom = iSetup.getHandle(tok_hgcGeom_);
  if (!geom.isValid())
    edm::LogVerbatim("HGCalValidation") << "HGCalDigiStudy: Cannot get "
                                        << "valid Geometry Object for " << nameDetector_;
  hgcGeom_ = geom.product();
  layerFront_ = hgcGeom_->topology().dddConstants().firstLayer();
  layers_ = hgcGeom_->topology().dddConstants().layers(true);
  if (hgcGeom_->topology().waferHexagon8())
    geomType_ = 1;
  else
    geomType_ = 2;
  if (nameDetector_ == "HGCalHFNoseSensitive")
    geomType_ = 3;
  edm::LogVerbatim("HGCalValidation") << "HGCalDigiStudy: gets Geometry for " << nameDetector_ << " of type "
                                      << geomType_ << " with " << layers_ << " layers and front layer " << layerFront_;
}

void HGCalDigiStudy::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  unsigned int ntot(0), nused(0);

  if ((nameDetector_ == "HGCalEESensitive") || (nameDetector_ == "HGCalHFNoseSensitive")) {
    //HGCalEE
    const edm::Handle<HGCalDigiCollection>& theHGCEEDigiContainer = iEvent.getHandle(digiSource_);
    if (theHGCEEDigiContainer.isValid()) {
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation")
            << nameDetector_ << " with " << theHGCEEDigiContainer->size() << " element(s)";
      for (const auto& it : *(theHGCEEDigiContainer.product())) {
        ntot++;
        nused++;
        DetId detId = it.id();
        int layer =
            ((geomType_ == 1) ? HGCSiliconDetId(detId).layer()
                              : ((geomType_ == 2) ? HGCScintillatorDetId(detId).layer() : HFNoseDetId(detId).layer()));
        const HGCSample& hgcSample = it.sample(SampleIndx_);
        uint16_t adc = hgcSample.data();
        double charge = adc;
        //      uint16_t   gain      = hgcSample.toa();
        //      double     charge    = adc*gain;
        digiValidation(detId, hgcGeom_, layer, adc, charge);
        if (geomType_ == 1) {
          HGCSiliconDetId id = HGCSiliconDetId(detId);
          h_Ly_->Fill(id.layer());
          h_W1_->Fill(id.waferU());
          h_W2_->Fill(id.waferV());
          h_C1_->Fill(id.cellU());
          h_C2_->Fill(id.cellV());
        } else if (geomType_ == 2) {
          HGCScintillatorDetId id = HGCScintillatorDetId(detId);
          h_Ly_->Fill(id.layer());
          h_W1_->Fill(id.ieta());
          h_C1_->Fill(id.iphi());
        } else {
          HFNoseDetId id = HFNoseDetId(detId);
          h_Ly_->Fill(id.layer());
          h_W1_->Fill(id.waferU());
          h_W2_->Fill(id.waferV());
          h_C1_->Fill(id.cellU());
          h_C2_->Fill(id.cellV());
        }
      }
    } else {
      edm::LogVerbatim("HGCalValidation")
          << "DigiCollection handle " << source_ << " does not exist for " << nameDetector_ << " !!!";
    }
  } else if ((nameDetector_ == "HGCalHESiliconSensitive") || (nameDetector_ == "HGCalHEScintillatorSensitive")) {
    //HGCalHE
    const edm::Handle<HGCalDigiCollection>& theHGCHEDigiContainer = iEvent.getHandle(digiSource_);
    if (theHGCHEDigiContainer.isValid()) {
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation")
            << nameDetector_ << " with " << theHGCHEDigiContainer->size() << " element(s)";
      for (const auto& it : *(theHGCHEDigiContainer.product())) {
        ntot++;
        nused++;
        DetId detId = it.id();
        int layer =
            ((geomType_ == 1) ? HGCSiliconDetId(detId).layer()
                              : ((geomType_ == 2) ? HGCScintillatorDetId(detId).layer() : HFNoseDetId(detId).layer()));
        const HGCSample& hgcSample = it.sample(SampleIndx_);
        uint16_t adc = hgcSample.data();
        double charge = adc;
        //      uint16_t   gain      = hgcSample.toa();
        //      double     charge    = adc*gain;
        digiValidation(detId, hgcGeom_, layer, adc, charge);
        if (geomType_ == 1) {
          HGCSiliconDetId id = HGCSiliconDetId(detId);
          h_Ly_->Fill(id.layer());
          h_W1_->Fill(id.waferU());
          h_W2_->Fill(id.waferV());
          h_C1_->Fill(id.cellU());
          h_C2_->Fill(id.cellV());
        } else if (geomType_ == 2) {
          HGCScintillatorDetId id = HGCScintillatorDetId(detId);
          h_Ly_->Fill(id.layer());
          h_W1_->Fill(id.ieta());
          h_C1_->Fill(id.iphi());
        } else {
          HFNoseDetId id = HFNoseDetId(detId);
          h_Ly_->Fill(id.layer());
          h_W1_->Fill(id.waferU());
          h_W2_->Fill(id.waferV());
          h_C1_->Fill(id.cellU());
          h_C2_->Fill(id.cellV());
        }
      }
    } else {
      edm::LogVerbatim("HGCalValidation")
          << "DigiCollection handle " << source_ << " does not exist for " << nameDetector_ << " !!!";
    }
  } else {
    edm::LogWarning("HGCalValidation") << "invalid detector name !! " << nameDetector_ << " source " << source_;
  }
  edm::LogVerbatim("HGCalValidation") << "Event " << iEvent.id().event() << ":" << nameDetector_ << " with " << ntot
                                      << " total and " << nused << " used digis";
}

template <class T1, class T2>
void HGCalDigiStudy::digiValidation(const T1& detId, const T2* geom, int layer, uint16_t adc, double charge) {
  if (verbosity_ > 1)
    edm::LogVerbatim("HGCalValidation") << std::hex << detId.rawId() << std::dec << " adc = " << adc
                                        << " charge = " << charge;

  DetId id1 = DetId(detId.rawId());
  const GlobalPoint& gcoord = geom->getPosition(id1);

  digiInfo hinfo;
  hinfo.r = gcoord.perp();
  hinfo.z = gcoord.z();
  hinfo.eta = std::abs(gcoord.eta());
  hinfo.phi = gcoord.phi();
  hinfo.adc = adc;
  hinfo.charge = charge;
  hinfo.layer = layer + layerFront_;
  if (verbosity_ > 1)
    edm::LogVerbatim("HGCalValidation") << "R =  " << hinfo.r << " z = " << hinfo.z << " eta = " << hinfo.eta
                                        << " phi = " << hinfo.phi;

  h_Charge_->Fill(hinfo.charge);
  h_ADC_->Fill(hinfo.adc);
  h_RZ_->Fill(std::abs(hinfo.z), hinfo.r);
  if (ifLayer_) {
    if (layer <= static_cast<int>(h_XY_.size()))
      h_XY_[layer - 1]->Fill(gcoord.x(), gcoord.y());
  } else {
    h_EtaPhi_->Fill(hinfo.eta, hinfo.phi);
  }
  if (hinfo.z > 0)
    h_LayZp_->Fill(hinfo.layer, hinfo.charge);
  else
    h_LayZm_->Fill(hinfo.layer, hinfo.charge);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalDigiStudy);
