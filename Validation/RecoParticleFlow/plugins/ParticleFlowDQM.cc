#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DQMOffline/PFTau/interface/Matchers.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <algorithm>
#include <numeric>
#include <regex>
#include <sstream>
#include <vector>

class ParticleFlowDQM : public DQMEDAnalyzer {
public:
    ParticleFlowDQM(const edm::ParameterSet&);
    void analyze(const edm::Event&, const edm::EventSetup&) override;

protected:
    //Book histograms
    void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
    void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override {}
    void endRun(const edm::Run&, const edm::EventSetup&) override {}

private:

    class Plot1DInBin {
    public:
        const std::string name, title;
        const uint32_t nbins;
        const double min, max;
        const double ptbin_low, ptbin_high, etabin_low, etabin_high;
        MonitorElement* plot_;

        Plot1DInBin(
            const std::string _name,
            const std::string _title,
            const uint32_t _nbins, const double _min, const double _max,
            double _ptbin_low, double _ptbin_high, double _etabin_low, double _etabin_high
            )
            : name(_name),
            title(_title),
            nbins(_nbins),
            min(_min),
            max(_max),
            ptbin_low(_ptbin_low),
            ptbin_high(_ptbin_high),
            etabin_low(_etabin_low),
            etabin_high(_etabin_high)
        {
        }

        void book(DQMStore::IBooker& booker) {
            plot_ = booker.book1D(name, title, nbins, min, max);
        }

        void fill(double value) {
            assert(plot_ != nullptr);
            plot_->Fill(value);
        }

        //Check if a jet with a value v would be in the bin that applies to this plot
        bool isInBin(double v, double low, double high) {
            return v >= low && v < high;
        }

        bool isInPtBin(double pt) {
            return isInBin(pt, ptbin_low, ptbin_high);
        }

        bool isInEtaBin(double eta) {
            return isInBin(eta, etabin_low, etabin_high);
        }

        bool isInPtEtaBin(double pt, double eta) {
            return isInPtBin(pt) && isInEtaBin(eta);
        }
    };

    std::vector<Plot1DInBin> jetResponsePlots;

    double jetDeltaR;

    edm::InputTag recoJetsLabel;
    edm::InputTag genJetsLabel;
    edm::EDGetTokenT<edm::View<reco::Jet>> recoJetsToken;
    edm::EDGetTokenT<edm::View<reco::Jet>> genJetsToken;
    edm::EDGetTokenT<reco::CandViewMatchMap> srcRefToJetMap;

    void fillJetResponse(edm::View<reco::Jet>& recoJetCollection, edm::View<reco::Jet>& genJetCollection);
};

ParticleFlowDQM::ParticleFlowDQM(const edm::ParameterSet& iConfig)
{
    recoJetsLabel = iConfig.getParameter<edm::InputTag>("recoJetCollection");
    genJetsLabel = iConfig.getParameter<edm::InputTag>("genJetCollection");

    //DeltaR for reco to gen jet matching
    jetDeltaR = iConfig.getParameter<double>("jetDeltaR");

    //Create all jet response plots in bins of genjet pt and eta
    const auto& response_plots = iConfig.getParameter<std::vector<edm::ParameterSet>>("responsePlots");
    for (auto& pset : response_plots) {
        //Low and high edges of the pt and eta bins for jets to pass to be filled into this histogram
        const auto ptbin_low = pset.getParameter<double>("ptBinLow");
        const auto ptbin_high = pset.getParameter<double>("ptBinHigh");
        const auto etabin_low = pset.getParameter<double>("etaBinLow");
        const auto etabin_high = pset.getParameter<double>("etaBinHigh");

        const auto response_nbins = pset.getParameter<uint32_t>("responseNbins");
        const auto response_low = pset.getParameter<double>("responseLow");
        const auto response_high = pset.getParameter<double>("responseHigh");

        const auto name = pset.getParameter<std::string>("name");
        const auto title = pset.getParameter<std::string>("title");

        jetResponsePlots.push_back(Plot1DInBin(
            name, title,
            response_nbins, response_low, response_high,
            ptbin_low, ptbin_high,
            etabin_low, etabin_high
            ));
    }
    if (jetResponsePlots.size() > 200) {
        throw std::runtime_error("Requested too many jet response plots, aborting as this seems unusual.");
    }

    recoJetsToken = consumes<edm::View<reco::Jet>>(recoJetsLabel);
    genJetsToken = consumes<edm::View<reco::Jet>>(genJetsLabel);

}

void ParticleFlowDQM::fillJetResponse(
    edm::View<reco::Jet>& recoJetCollection,
    edm::View<reco::Jet>& genJetCollection)
{

    //match gen jets to reco jets, require minimum jetDeltaR, choose closest, do not try to match charge
    std::vector<int> matchIndices;
    PFB::match(genJetCollection, recoJetCollection, matchIndices, false, jetDeltaR);

    for (unsigned int i = 0; i < genJetCollection.size(); i++) {

        const auto& genJet = genJetCollection.at(i);
        int iMatch = matchIndices[i];
        
        //If gen jet had a matched reco jet
        if (iMatch != -1) {
            const auto& recoJet = recoJetCollection[iMatch];
            const auto pt_reco = recoJet.pt();
            const auto pt_gen = genJet.pt();
            const auto eta_gen = abs(genJet.eta());
            const auto response = pt_reco / pt_gen;

            //Loop linearly through all plots and check if they match the pt and eta bin
            //this is not algorithmically optimal but we don't expect to more than a few hundred plots
            //If this turns out to be a problem, can easily make a 2D-map for indices
            for (auto& plot : jetResponsePlots) {
                if (plot.isInPtEtaBin(pt_gen, eta_gen)) {
                    plot.fill(response);
                }
            }
        }
    }
}

void ParticleFlowDQM::bookHistograms(DQMStore::IBooker & booker, edm::Run const &, edm::EventSetup const &) {
    std::cout << "ParticleFlowDQM booking response histograms" << std::endl;
    booker.setCurrentFolder("Physics/JetResponse/");
    for (auto& plot : jetResponsePlots) {
        plot.book(booker);
    }
}

void ParticleFlowDQM::analyze(const edm::Event& iEvent, const edm::EventSetup&)
{

    edm::Handle<edm::View<reco::Jet>> recoJetCollectionHandle;
    iEvent.getByToken(recoJetsToken, recoJetCollectionHandle);

    edm::Handle<edm::View<reco::Jet>> genJetCollectionHandle;
    iEvent.getByToken(genJetsToken, genJetCollectionHandle);
    
    auto recoJetCollection = *recoJetCollectionHandle;
    auto genJetCollection = *genJetCollectionHandle;

    fillJetResponse(recoJetCollection, genJetCollection);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ParticleFlowDQM);
