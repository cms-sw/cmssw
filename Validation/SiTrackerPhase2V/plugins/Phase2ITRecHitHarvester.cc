#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2HarvestingUtil.h"
class Phase2ITRecHitHarvester : public DQMEDHarvester {
public:
  explicit Phase2ITRecHitHarvester(const edm::ParameterSet&);
  ~Phase2ITRecHitHarvester() override;
  void dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void gausFitslices(MonitorElement* srcME, MonitorElement* meanME, MonitorElement* sigmaME);
  void dofitsForLayer(const std::string& iFolder, DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter);

  // ----------member data ---------------------------
  const edm::ParameterSet config_;
  const std::string topFolder_;
  const unsigned int nbarrelLayers_;
  const unsigned int ndisk1Rings_;  //FOR IT epix//FOR OT TEDD1 rings
  const unsigned int ndisk2Rings_;  //FOR IT epix//FOR OT TEDD2 rings
  const unsigned int fitThreshold_;
  const std::string ecapdisk1Name_;  //FOR IT Epix//FOR OT TEDD_1
  const std::string ecapdisk2Name_;  //FOR IT Fpix//FOR OT TEDD_2
  const std::string histoPhiname_;
  const std::string deltaXvsEtaname_;
  const std::string deltaXvsPhiname_;
  const std::string deltaYvsEtaname_;
  const std::string deltaYvsPhiname_;
};

Phase2ITRecHitHarvester::Phase2ITRecHitHarvester(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      topFolder_(iConfig.getParameter<std::string>("TopFolder")),
      nbarrelLayers_(iConfig.getParameter<uint32_t>("NbarrelLayers")),
      ndisk1Rings_(iConfig.getParameter<uint32_t>("NDisk1Rings")),
      ndisk2Rings_(iConfig.getParameter<uint32_t>("NDisk2Rings")),
      fitThreshold_(iConfig.getParameter<uint32_t>("NFitThreshold")),
      ecapdisk1Name_(iConfig.getParameter<std::string>("EcapDisk1Name")),
      ecapdisk2Name_(iConfig.getParameter<std::string>("EcapDisk2Name")),
      deltaXvsEtaname_(iConfig.getParameter<std::string>("ResidualXvsEta")),
      deltaXvsPhiname_(iConfig.getParameter<std::string>("ResidualXvsPhi")),
      deltaYvsEtaname_(iConfig.getParameter<std::string>("ResidualYvsEta")),
      deltaYvsPhiname_(iConfig.getParameter<std::string>("ResidualYvsPhi")) {}

Phase2ITRecHitHarvester::~Phase2ITRecHitHarvester() {}

void Phase2ITRecHitHarvester::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  for (unsigned int i = 1; i <= nbarrelLayers_; i++) {
    std::string iFolder = topFolder_ + "/Barrel/Layer" + std::to_string(i);
    dofitsForLayer(iFolder, ibooker, igetter);
  }
  for (unsigned int iSide = 1; iSide <= 2; iSide++) {
    const std::string ecapbasedisk1 =
        topFolder_ + "/EndCap_Side" + std::to_string(iSide) + "/" + ecapdisk1Name_ + "/Ring";
    const std::string ecapbasedisk2 =
        topFolder_ + "/EndCap_Side" + std::to_string(iSide) + "/" + ecapdisk2Name_ + "/Ring";
    //EPix or TEDD_1
    for (unsigned int epixr = 1; epixr <= ndisk1Rings_; epixr++) {
      std::string iFolder = ecapbasedisk1 + std::to_string(epixr);
      dofitsForLayer(iFolder, ibooker, igetter);
    }
    //FPix or TEDD_2
    for (unsigned int fpixr = 1; fpixr <= ndisk2Rings_; fpixr++) {
      std::string iFolder = ecapbasedisk2 + std::to_string(fpixr);
      dofitsForLayer(iFolder, ibooker, igetter);
    }
  }
}
//Function for Layer/Ring
void Phase2ITRecHitHarvester::dofitsForLayer(const std::string& iFolder,
                                             DQMStore::IBooker& ibooker,
                                             DQMStore::IGetter& igetter) {
  MonitorElement* deltaX_eta = igetter.get(iFolder + "/" + deltaXvsEtaname_);
  MonitorElement* deltaX_phi = igetter.get(iFolder + "/" + deltaXvsPhiname_);
  MonitorElement* deltaY_eta = igetter.get(iFolder + "/" + deltaYvsEtaname_);
  MonitorElement* deltaY_phi = igetter.get(iFolder + "/" + deltaYvsPhiname_);

  std::string resFolder = iFolder + "/ResolutionFromFit/";

  ibooker.cd();
  ibooker.setCurrentFolder(resFolder);
  MonitorElement* sigmaX_eta =
      phase2tkharvestutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("resXvseta"), ibooker);
  MonitorElement* meanX_eta =
      phase2tkharvestutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("meanXvseta"), ibooker);

  gausFitslices(deltaX_eta, meanX_eta, sigmaX_eta);

  MonitorElement* sigmaY_eta =
      phase2tkharvestutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("resYvseta"), ibooker);
  MonitorElement* meanY_eta =
      phase2tkharvestutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("meanYvseta"), ibooker);
  gausFitslices(deltaY_eta, meanY_eta, sigmaY_eta);

  MonitorElement* sigmaX_phi =
      phase2tkharvestutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("resXvsphi"), ibooker);
  MonitorElement* meanX_phi =
      phase2tkharvestutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("meanXvsphi"), ibooker);
  gausFitslices(deltaX_phi, meanX_phi, sigmaX_phi);

  MonitorElement* sigmaY_phi =
      phase2tkharvestutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("resYvsphi"), ibooker);
  MonitorElement* meanY_phi =
      phase2tkharvestutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("meanYvsphi"), ibooker);
  gausFitslices(deltaY_phi, meanY_phi, sigmaY_phi);
}
void Phase2ITRecHitHarvester::gausFitslices(MonitorElement* srcME, MonitorElement* meanME, MonitorElement* sigmaME) {
  TH2F* histo = srcME->getTH2F();

  // Fit slices projected along Y from bins in X
  const unsigned int cont_min = fitThreshold_;  //Minimum number of entries
  for (int i = 1; i <= srcME->getNbinsX(); i++) {
    TString iString(i);
    TH1* histoY = histo->ProjectionY(" ", i, i);
    const unsigned int cont = histoY->GetEntries();

    if (cont >= cont_min) {
      float minfit = histoY->GetMean() - histoY->GetRMS();
      float maxfit = histoY->GetMean() + histoY->GetRMS();

      std::unique_ptr<TF1> fitFcn{new TF1(TString("g") + histo->GetName() + iString, "gaus", minfit, maxfit)};
      histoY->Fit(fitFcn.get(), "QR0 SERIAL", "", minfit, maxfit);

      double* par = fitFcn->GetParameters();
      const double* err = fitFcn->GetParErrors();

      meanME->setBinContent(i, par[1]);
      meanME->setBinError(i, err[1]);

      sigmaME->setBinContent(i, par[2]);
      sigmaME->setBinError(i, err[2]);

      if (histoY)
        delete histoY;
    } else {
      if (histoY)
        delete histoY;
      continue;
    }
  }
}

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
void Phase2ITRecHitHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // phase2ITrechitHarvester
  edm::ParameterSetDescription desc;
  desc.add<std::string>("TopFolder", "TrackerPhase2ITRecHitV");
  desc.add<unsigned int>("NbarrelLayers", 4);
  desc.add<unsigned int>("NDisk1Rings", 5);
  desc.add<unsigned int>("NDisk2Rings", 4);
  desc.add<std::string>("EcapDisk1Name", "EPix");
  desc.add<std::string>("EcapDisk2Name", "FPix");
  desc.add<std::string>("ResidualXvsEta", "Delta_X_vs_Eta");
  desc.add<std::string>("ResidualXvsPhi", "Delta_X_vs_Phi");
  desc.add<std::string>("ResidualYvsEta", "Delta_Y_vs_Eta");
  desc.add<std::string>("ResidualYvsPhi", "Delta_Y_vs_Phi");

  edm::ParameterSetDescription psd0;
  psd0.add<std::string>("name", "resolutionXFitvseta");
  psd0.add<std::string>("title", ";|#eta|; X-Resolution from fit [#mum]");
  psd0.add<int>("NxBins", 41);
  psd0.add<double>("xmax", 4.1);
  psd0.add<double>("xmin", 0.);
  psd0.add<bool>("switch", true);
  desc.add<edm::ParameterSetDescription>("resXvseta", psd0);

  edm::ParameterSetDescription psd1;
  psd1.add<std::string>("name", "resolutionYFitvseta");
  psd1.add<std::string>("title", ";|#eta|; Y-Resolution from fit [#mum]");
  psd1.add<int>("NxBins", 41);
  psd1.add<double>("xmax", 4.1);
  psd1.add<double>("xmin", 0.);
  psd1.add<bool>("switch", true);
  desc.add<edm::ParameterSetDescription>("resYvseta", psd1);

  edm::ParameterSetDescription psd2;
  psd2.add<std::string>("name", "resolutionXFitvsphi");
  psd2.add<std::string>("title", ";#phi; X-Resolution from fit [#mum]");
  psd2.add<int>("NxBins", 36);
  psd2.add<double>("xmax", M_PI);
  psd2.add<double>("xmin", -M_PI);
  psd2.add<bool>("switch", true);
  desc.add<edm::ParameterSetDescription>("resXvsphi", psd2);

  edm::ParameterSetDescription psd3;
  psd3.add<std::string>("name", "resolutionYFitvsphi");
  psd3.add<std::string>("title", ";#phi; Y-Resolution from fit [#mum]");
  psd3.add<int>("NxBins", 36);
  psd3.add<double>("xmax", M_PI);
  psd3.add<double>("xmin", -M_PI);
  psd3.add<bool>("switch", true);
  desc.add<edm::ParameterSetDescription>("resYvsphi", psd3);

  edm::ParameterSetDescription psd4;
  psd4.add<std::string>("name", "meanXFitvseta");
  psd4.add<std::string>("title", ";|#eta|; Mean residual X from fit [#mum]");
  psd4.add<int>("NxBins", 41);
  psd4.add<double>("xmax", 4.1);
  psd4.add<double>("xmin", 0.);
  psd4.add<bool>("switch", true);
  desc.add<edm::ParameterSetDescription>("meanXvseta", psd4);

  edm::ParameterSetDescription psd5;
  psd5.add<std::string>("name", "meanYFitvseta");
  psd5.add<std::string>("title", ";|#eta|; Mean residual Y from fit [#mum]");
  psd5.add<int>("NxBins", 41);
  psd5.add<double>("xmax", 4.1);
  psd5.add<double>("xmin", 0.);
  psd5.add<bool>("switch", true);
  desc.add<edm::ParameterSetDescription>("meanYvseta", psd5);

  edm::ParameterSetDescription psd6;
  psd6.add<std::string>("name", "meanXFitvsphi");
  psd6.add<std::string>("title", ";#phi; Mean residual X from fit [#mum]");
  psd6.add<int>("NxBins", 36);
  psd6.add<double>("xmax", M_PI);
  psd6.add<double>("xmin", -M_PI);
  psd6.add<bool>("switch", true);
  desc.add<edm::ParameterSetDescription>("meanXvsphi", psd6);

  edm::ParameterSetDescription psd7;
  psd7.add<std::string>("name", "meanYFitvsphi");
  psd7.add<std::string>("title", ";#phi; Mean residual Y from fit [#mum]");
  psd7.add<int>("NxBins", 36);
  psd7.add<double>("xmax", M_PI);
  psd7.add<double>("xmin", -M_PI);
  psd7.add<bool>("switch", true);
  desc.add<edm::ParameterSetDescription>("meanYvsphi", psd7);

  desc.add<unsigned int>("NFitThreshold", 100);

  descriptions.add("Phase2ITRechitHarvester", desc);
  // or use the following to generate the label from the module's C++ type
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2ITRecHitHarvester);
