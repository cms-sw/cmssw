#include <algorithm>
#include <cctype>
#include <iomanip>
#include <set>
#include <sstream>
#include <vector>

#include "TFile.h"
#include "TH1F.h"

#include "HepPDT/ParticleID.hh"

// user include files
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/IPTools/interface/IPTools.h"

#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexSorter.h"

#include "SimTracker/TrackHistory/interface/TrackClassifier.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"

//
// class decleration
//

class QualityCutsAnalyzer : public edm::EDAnalyzer
{

public:

    explicit QualityCutsAnalyzer(const edm::ParameterSet&);
    ~QualityCutsAnalyzer();

private:

    virtual void beginJob() override ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;

    // Member data

    typedef std::vector<int> vint;
    typedef std::vector<std::string> vstring;

    edm::InputTag trackProducer_;
    edm::InputTag primaryVertexProducer_;
    edm::InputTag jetTracksAssociation_;

    std::string rootFile_;

    int minimumNumberOfHits_, minimumNumberOfPixelHits_;
    double minimumTransverseMomentum_, maximumChiSquared_;

    bool useAllQualities_;
    reco::TrackBase::TrackQuality trackQuality_;

    void
    LoopOverJetTracksAssociation(
        const edm::ESHandle<TransientTrackBuilder> &,
        const edm::Handle<reco::VertexCollection> &,
        const edm::Handle<reco::JetTracksAssociationCollection> &
    );

    // Histograms for optimization

    struct histogram_element_t
    {
        double sdl;  // Signed decay length
        double dta;  // Distance to jet axis
        double tip;  // Transverse impact parameter
        double lip;  // Longitudinal impact parameter
        double ips;  // Impact parameter significance.
        double pt;   // Transverse momentum
        double chi2; // Chi^2
        std::size_t hits;      // Number of hits
        std::size_t pixelhits; // Number of hits

        histogram_element_t(double d, double a, double t, double l, double i, double p, double c, std::size_t h, std::size_t x)
        {
            sdl = d;
            dta = a;
            tip = t;
            lip = l;
            ips = i;
            pt = p;
            chi2 = c;
            hits = h;
            pixelhits = x;
        }

        histogram_element_t(const histogram_element_t & orig)
        {
            sdl = orig.sdl;
            dta = orig.dta;
            tip = orig.tip;
            lip = orig.lip;
            ips = orig.ips;
            pt = orig.pt;
            chi2 = orig.chi2;
            hits = orig.hits;
            pixelhits = orig.pixelhits;
        }
    };

    typedef std::vector<std::vector<histogram_element_t> > histogram_data_t;
    histogram_data_t histogram_data_;

    class histogram_t
    {

        TH1F* sdl;
        TH1F* dta;
        TH1F* tip;
        TH1F* lip;
        TH1F* ips;
        TH1F* pixelhits;
        TH1F* pt_1gev;
        TH1F* chi2;
        TH1F* hits;

    public:

        histogram_t(const std::string & particleType)
        {
            std::string name, title;
            name = std::string("hits_") + particleType;
            title = std::string("Hit distribution for ") + particleType;
            hits = new TH1F(name.c_str(), title.c_str(), 19, -0.5, 18.5);

            name = std::string("chi2_") + particleType;
            title = std::string("Chi2 distribution for ") + particleType;
            chi2 = new TH1F(name.c_str(), title.c_str(), 100, 0., 30.);

            name = std::string("pixelhits_") + particleType;
            title = std::string("Pixel hits distribution for ") + particleType;
            pixelhits = new TH1F(name.c_str(), title.c_str(), 21, -0.5, 20.5);

            name = std::string("pt_1Gev_") + particleType;
            title = std::string("Pt distribution close 1Gev for ") + particleType;
            pt_1gev = new TH1F(name.c_str(), title.c_str(), 100, 0., 2.);

            name = std::string("tip_") + particleType;
            title = std::string("Transverse impact parameter distribution for ") + particleType;
            tip = new TH1F(name.c_str(), title.c_str(), 100, -0.3, 0.3);

            name = std::string("lip_") + particleType;
            title = std::string("Longitudinal impact parameter distribution for ") + particleType;
            lip = new TH1F(name.c_str(), title.c_str(), 100, -1., 1.);

            name = std::string("ips_") + particleType;
            title = std::string("IPS distribution for ") + particleType;
            ips = new TH1F(name.c_str(), title.c_str(), 100, -25.0, 25.0);

            name = std::string("sdl_") + particleType;
            title = std::string("Decay length distribution for ") + particleType;
            sdl = new TH1F(name.c_str(), title.c_str(), 100, -5., 5.);

            name = std::string("dta_") + particleType;
            title = std::string("Distance to jet distribution for ") + particleType;
            dta = new TH1F(name.c_str(), title.c_str(), 100, 0.0, 0.2);
        }

        ~histogram_t()
        {
            delete hits;
            delete chi2;
            delete pixelhits;
            delete pt_1gev;
            delete tip;
            delete lip;
            delete ips;
            delete sdl;
            delete dta;
        }

        void Fill(const histogram_element_t & data)
        {
            hits->Fill(data.hits);
            chi2->Fill(data.chi2);
            pixelhits->Fill(data.pt);
            pt_1gev->Fill(data.pt);
            ips->Fill(data.ips);
            tip->Fill(data.tip);
            lip->Fill(data.lip);
            sdl->Fill(data.sdl);
            dta->Fill(data.dta);
        }

        void Write()
        {
            hits->Write();
            chi2->Write();
            pixelhits->Write();
            pt_1gev->Write();
            ips->Write();
            tip->Write();
            lip->Write();
            sdl->Write();
            dta->Write();
        }
    };

    // Track classification.
    TrackClassifier classifier_;

};


//
// constructors and destructor
//
QualityCutsAnalyzer::QualityCutsAnalyzer(const edm::ParameterSet& config) : classifier_(config,consumesCollector())
{
    trackProducer_         = config.getUntrackedParameter<edm::InputTag> ( "trackProducer" );
    consumes<edm::View<reco::Track>>(trackProducer_);
    primaryVertexProducer_ = config.getUntrackedParameter<edm::InputTag> ( "primaryVertexProducer" );
    consumes<reco::VertexCollection>(primaryVertexProducer_);
    jetTracksAssociation_  = config.getUntrackedParameter<edm::InputTag> ( "jetTracksAssociation" );
    consumes<reco::JetTracksAssociationCollection>(jetTracksAssociation_);

    rootFile_ = config.getUntrackedParameter<std::string> ( "rootFile" );

    minimumNumberOfHits_       = config.getUntrackedParameter<int> ( "minimumNumberOfHits" );
    minimumNumberOfPixelHits_  = config.getUntrackedParameter<int> ( "minimumNumberOfPixelHits" );
    minimumTransverseMomentum_ = config.getUntrackedParameter<double> ( "minimumTransverseMomentum" );
    maximumChiSquared_         = config.getUntrackedParameter<double> ( "maximumChiSquared" );

    std::string trackQualityType = config.getUntrackedParameter<std::string>("trackQualityClass"); //used
    trackQuality_ =  reco::TrackBase::qualityByName(trackQualityType);
    useAllQualities_ = false;

    std::transform(trackQualityType.begin(), trackQualityType.end(), trackQualityType.begin(), (int(*)(int)) std::tolower);
    if (trackQualityType == "any")
    {
        std::cout << "Using any" << std::endl;
        useAllQualities_ = true;
    }
}

QualityCutsAnalyzer::~QualityCutsAnalyzer() {}

//
// member functions
//

// ------------ method called to for each event  ------------
void
QualityCutsAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
    // Track collection
    edm::Handle<edm::View<reco::Track> > trackCollection;
    event.getByLabel(trackProducer_,trackCollection);
    // Primary vertex
    edm::Handle<reco::VertexCollection> primaryVertexCollection;
    event.getByLabel(primaryVertexProducer_, primaryVertexCollection);
    // Jet to tracks associator
    edm::Handle<reco::JetTracksAssociationCollection> jetTracks;
    event.getByLabel(jetTracksAssociation_, jetTracks);
    // Trasient track builder
    edm::ESHandle<TransientTrackBuilder> TTbuilder;
    setup.get<TransientTrackRecord>().get("TransientTrackBuilder", TTbuilder);

    // Setting up event information for the track categories.
    classifier_.newEvent(event, setup);

    LoopOverJetTracksAssociation(
        TTbuilder,
        primaryVertexCollection,
        jetTracks
    );
}


// ------------ method called once each job just before starting event loop  ------------
void
QualityCutsAnalyzer::beginJob()
{
    histogram_data_.resize(6);
}


// ------------ method called once each job just after ending the event loop  ------------
void
QualityCutsAnalyzer::endJob()
{
    TFile file(rootFile_.c_str(), "RECREATE");
    file.cd();

    // saving the histograms
    for (std::size_t i=0; i<6; i++)
    {
        std::string particle;
        if (i == 0)
            particle = std::string("B_tracks");
        else if (i == 1)
            particle = std::string("C_tracks");
        else if (i == 2)
            particle = std::string("nonB_tracks");
        else if (i == 3)
            particle = std::string("displaced_tracks");
        else if (i == 4)
            particle = std::string("bad_tracks");
        else
            particle = std::string("fake_tracks");

        histogram_t histogram(particle);

        for (std::size_t j=0; j<histogram_data_[i].size(); j++)
            histogram.Fill(histogram_data_[i][j]);

        histogram.Write();
    }

    file.Flush();
}


void
QualityCutsAnalyzer::LoopOverJetTracksAssociation(
    const edm::ESHandle<TransientTrackBuilder> & TTbuilder,
    const edm::Handle<reco::VertexCollection> & primaryVertexProducer_,
    const edm::Handle<reco::JetTracksAssociationCollection> & jetTracksAssociation
)
{
    const TransientTrackBuilder * bproduct = TTbuilder.product();

    // getting the primary vertex
    // use first pv of the collection
    reco::Vertex pv;

    if (primaryVertexProducer_->size() != 0)
    {
        PrimaryVertexSorter pvs;
        std::vector<reco::Vertex> sortedList = pvs.sortedList(*(primaryVertexProducer_.product()));
        pv = (sortedList.front());
    }
    else
    { // create a dummy PV
        // cout << "NO PV FOUND" << endl;
        reco::Vertex::Error e;
        e(0,0)=0.0015*0.0015;
        e(1,1)=0.0015*0.0015;
        e(2,2)=15.*15.;
        reco::Vertex::Point p(0,0,0);
        pv = reco::Vertex(p,e,1,1,1);
    }

    reco::JetTracksAssociationCollection::const_iterator it = jetTracksAssociation->begin();

    int i=0;

    for (; it != jetTracksAssociation->end(); it++, i++)
    {
        // get jetTracks object
        reco::JetTracksAssociationRef jetTracks(jetTracksAssociation, i);

        double pvZ = pv.z();
        GlobalVector direction(jetTracks->first->px(), jetTracks->first->py(), jetTracks->first->pz());

        // get the tracks associated to the jet
        reco::TrackRefVector tracks = jetTracks->second;
        for (std::size_t index = 0; index < tracks.size(); index++)
        {
            edm::RefToBase<reco::Track> track(tracks[index]);

            double pt = tracks[index]->pt();
            double chi2 = tracks[index]->normalizedChi2();
            int hits = tracks[index]->hitPattern().numberOfValidHits();
            int pixelHits = tracks[index]->hitPattern().numberOfValidPixelHits();

            if (
                hits < minimumNumberOfHits_ ||
                pixelHits < minimumNumberOfPixelHits_ ||
                pt < minimumTransverseMomentum_ ||
                chi2 >  maximumChiSquared_ ||
                (!useAllQualities_ && !tracks[index]->quality(trackQuality_))
            ) continue;

            const reco::TransientTrack transientTrack = bproduct->build(&(*tracks[index]));
            double dta = - IPTools::jetTrackDistance(transientTrack, direction, pv).second.value();
            double sdl = IPTools::signedDecayLength3D(transientTrack, direction, pv).second.value();
            double ips = IPTools::signedImpactParameter3D(transientTrack, direction, pv).second.value();
            double d0 = IPTools::signedTransverseImpactParameter(transientTrack, direction, pv).second.value();
            double dz = tracks[index]->dz() - pvZ;

            // Classify the reco track;
            classifier_.evaluate( edm::RefToBase<reco::Track>(tracks[index]) );

            // Check for the different categories
            if ( classifier_.is(TrackClassifier::Fake) )
                histogram_data_[5].push_back(histogram_element_t(sdl, dta, d0, dz, ips, pt, chi2, hits, pixelHits));
            else if ( classifier_.is(TrackClassifier::BWeakDecay) )
                histogram_data_[0].push_back(histogram_element_t(sdl, dta, d0, dz, ips, pt, chi2, hits, pixelHits));
            else if ( classifier_.is(TrackClassifier::Bad) )
                histogram_data_[4].push_back(histogram_element_t(sdl, dta, d0, dz, ips, pt, chi2, hits, pixelHits));
            else if (
                !classifier_.is(TrackClassifier::CWeakDecay) &&
                !classifier_.is(TrackClassifier::PrimaryVertex)
            )
                histogram_data_[3].push_back(histogram_element_t(sdl, dta, d0, dz, ips, pt, chi2, hits, pixelHits));
            else if ( classifier_.is(TrackClassifier::CWeakDecay) )
                histogram_data_[1].push_back(histogram_element_t(sdl, dta, d0, dz, ips, pt, chi2, hits, pixelHits));
            else
                histogram_data_[2].push_back(histogram_element_t(sdl, dta, d0, dz, ips, pt, chi2, hits, pixelHits));

        }
    }
}

DEFINE_FWK_MODULE(QualityCutsAnalyzer);
