#include "DQMOffline/PFTau/interface/Matchers.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include <SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h>

#include <vector>
#include <map>

class OffsetAnalyzerDQM : public DQMEDAnalyzer {
public:
    OffsetAnalyzerDQM(const edm::ParameterSet&);
    void analyze(const edm::Event&, const edm::EventSetup&) override;

protected:
    //Book histograms
    void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
    void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override {}
    void endRun(const edm::Run&, const edm::EventSetup&) override {}

private:
    struct Plot1D {
        std::string name, title, dir;
        int nxbins;
        double xlow, xhigh;
        MonitorElement* plot;

        Plot1D() : name(""), title(""), dir(""), nxbins(0), xlow(0), xhigh(0), plot(nullptr) {}
        Plot1D( const std::string& n, const std::string& t, const std::string& d,
                int nx, double x0, double x1 ) :
                name(n), title(t), dir(d), nxbins(nx), xlow(x0), xhigh(x1), plot(nullptr) {}

        virtual void book( DQMStore::IBooker& booker ) {
            booker.setCurrentFolder( dir );
            plot = booker.book1D( name, title, nxbins, xlow, xhigh );
        }

        virtual void fill( float value ) {
            assert( plot != nullptr );
            plot->Fill( value );
        }

        virtual ~Plot1D() {}
    };

    struct PlotProfile : public Plot1D {
        std::vector<double> xbins;
        int nybins;
        double ylow, yhigh;
        PlotProfile() : Plot1D(), xbins(0), nybins(0), ylow(0), yhigh(0) {}
        PlotProfile( const std::string& n, const std::string& t, const std::string& d, int nx, double x0, double x1,
                     const std::vector<double>& vx, int ny, double y0, double y1 ) :
                     Plot1D(n,t,d,nx,x0,x1), xbins(vx), nybins(ny), ylow(y0), yhigh(y1) {}

        void book( DQMStore::IBooker& booker ) {
            booker.setCurrentFolder( dir );
            plot = booker.bookProfile( name, title, xbins.size()-1, &xbins[0], nybins, ylow, yhigh );
        }
        //make other booker methods for uniform binning

        void fill( double value1, double value2 ) {
            assert( plot != nullptr );
            plot->Fill( value1, value2 );
        }
    };

    std::map<std::string, Plot1D> th1dPlots;
    std::map<std::string, PlotProfile> offsetPlots;
    std::map<int, std::string> pdgMap;

    std::string offsetPlotBaseName;

    edm::EDGetTokenT< edm::View<reco::Vertex> > pvToken;
    edm::EDGetTokenT< edm::View<PileupSummaryInfo> > muToken;
    edm::EDGetTokenT< edm::View<pat::PackedCandidate> > pfToken;
};

OffsetAnalyzerDQM::OffsetAnalyzerDQM(const edm::ParameterSet& iConfig)
{
    offsetPlotBaseName = iConfig.getParameter<std::string>("offsetPlotBaseName");

    pvToken = consumes< edm::View<reco::Vertex> >( iConfig.getParameter<edm::InputTag>("pvTag") );
    muToken = consumes< edm::View<PileupSummaryInfo> >( iConfig.getParameter<edm::InputTag>("muTag") );
    pfToken = consumes< edm::View<pat::PackedCandidate> >( iConfig.getParameter<edm::InputTag>("pfTag") );

    //initialize offset plots
    const auto& offset_psets = iConfig.getParameter<std::vector<edm::ParameterSet>>("offsetPlots");
    for (auto& pset : offset_psets) {
        std::string name = pset.getParameter<std::string>("name");
        std::string title = pset.getParameter<std::string>("title");
        std::string dir = pset.getParameter<std::string>("dir");
        std::vector<double> vx = pset.getParameter< std::vector<double> >("vx");
        int ny = pset.getParameter<uint32_t>("ny");
        double y0 = pset.getParameter<double>("y0");
        double y1 = pset.getParameter<double>("y1");

        offsetPlots[name] = PlotProfile(name, title, dir, 0, 0, 0, vx, ny, y0, y1);
    }

    //initialize th1d
    const auto& th1d_psets = iConfig.getParameter<std::vector<edm::ParameterSet>>("th1dPlots");
    for (auto& pset : th1d_psets) {
        std::string name = pset.getParameter<std::string>("name");
        std::string title = pset.getParameter<std::string>("title");
        std::string dir = pset.getParameter<std::string>("dir");
        int nx = pset.getParameter<uint32_t>("nx");
        double x0 = pset.getParameter<double>("x0");
        double x1 = pset.getParameter<double>("x1");

        th1dPlots[name] = Plot1D(name, title, dir, nx, x0, x1);
    }

    //create pdg map
    std::vector<uint32_t> pdgKeys = iConfig.getParameter< std::vector<uint32_t> >("pdgKeys");
    std::vector<std::string> pdgStrs = iConfig.getParameter< std::vector<std::string> >("pdgStrs");
    for (int i=0, n=pdgKeys.size(); i<n; i++) pdgMap[pdgKeys[i]] = pdgStrs[i];
}

void OffsetAnalyzerDQM::bookHistograms(DQMStore::IBooker & booker, edm::Run const &, edm::EventSetup const &) {
    std::cout << "OffsetAnalyzerDQM booking offset histograms" << std::endl;
    for (auto& pair : offsetPlots) {
        pair.second.book(booker);
    }
    for (auto& pair : th1dPlots) {
        pair.second.book(booker);
    }
}

void OffsetAnalyzerDQM::analyze(const edm::Event& iEvent, const edm::EventSetup&)
{
    //npv//
    edm::Handle< edm::View<reco::Vertex> > vertexHandle;
    iEvent.getByToken(pvToken, vertexHandle);

    int npv = 0;
    for (unsigned int i=0, n=vertexHandle->size(); i<n; i++) {
        const auto& pv = vertexHandle->at(i);

        if( !pv.isFake() && pv.ndof() > 4 && fabs(pv.z()) <= 24 && fabs(pv.position().rho()) <= 2 )
            npv++;
    }
    th1dPlots["npv"].fill( npv );

    //mu//
    int int_mu = -1;
    edm::Handle< edm::View<PileupSummaryInfo> > muHandle;
    if ( iEvent.getByToken(muToken, muHandle) ) {
      float mu = muHandle->at(1).getTrueNumInteractions();
      th1dPlots["mu"].fill( mu );
      int_mu = mu + 0.5;
    }

    //pf particles//
    edm::Handle< edm::View<pat::PackedCandidate> > pfHandle;
    iEvent.getByToken(pfToken, pfHandle);

    for (unsigned int i=0, n=pfHandle->size(); i<n ; i++) {
        const auto& cand = pfHandle->at(i);

        std::string pftype = pdgMap[ abs(cand.pdgId()) ];
        if ( pftype == "" ) continue;

        if ( pftype == "chm" ) { //check charged hadrons ONLY
            bool attached = false;

            for (unsigned int ipv=0, endpv=vertexHandle->size(); ipv<endpv && !attached; ipv++) {
                const auto& pv = vertexHandle->at(ipv);
                if ( !pv.isFake() && pv.ndof() >= 4 && fabs(pv.z()) < 24 ) { //must be attached to a good pv
                    if ( cand.fromPV(ipv) == 3 ) attached = true; //pv used in fit
                }
            }
            if (!attached) pftype = "chu"; //unmatched charged hadron
        }
////AOD////
//        bool attached = false;
//        reco::TrackRef candTrkRef( cand.trackRef() );
//
//        if ( pftype == "chm" && !candTrkRef.isNull() ) { //check charged hadrons ONLY
//
//            for (auto ipv=vertexHandle->begin(), endpv=vertexHandle->end(); ipv != endpv && !attached; ++ipv) {
//                if ( !ipv->isFake() && ipv->ndof() >= 4 && fabs(ipv->z()) < 24 ) { //must be attached to a good pv
//
//                    for(auto ivtrk=ipv->tracks_begin(), endvtrk=ipv->tracks_end(); ivtrk != endvtrk && !attached; ++ivtrk) {
//                        reco::TrackRef pvTrkRef(ivtrk->castTo<reco::TrackRef>());
//                        if (pvTrkRef == candTrkRef) attached = true;
//                    }
//                }
//            }
//            if (!attached) pftype = "chu"; //unmatched charged hadron
//        }
///////////
        std::string offset_name_npv = offsetPlotBaseName + "_npv" + std::to_string(npv) + "_" + pftype;
        offsetPlots[offset_name_npv].fill( cand.eta(), cand.et() );

        if (int_mu != -1) {
            std::string offset_name_mu = offsetPlotBaseName + "_mu" + std::to_string(int_mu) + "_" + pftype;
            offsetPlots[offset_name_mu].fill( cand.eta(), cand.et() );
        }
    }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OffsetAnalyzerDQM);
