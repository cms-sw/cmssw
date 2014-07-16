/*
 *  TrackCategoriesAnalyzer.C
 *
 *  Created by Victor Eduardo Bazterra on 06/17/08.
 *
 */

#include "TH1F.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimTracker/TrackHistory/interface/TrackClassifier.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"   /// FIXME: added code

//
// class decleration
//

class TrackCategoriesAnalyzer : public edm::EDAnalyzer
{
public:

    explicit TrackCategoriesAnalyzer(const edm::ParameterSet&);
    ~TrackCategoriesAnalyzer();

private:

    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

    // Member data

    edm::InputTag trackProducer_;

    std::size_t totalTracks_;

    TrackClassifier classifier_;

    TH1F * trackCategories_;
    TH1F * processTypesClassifier_;       /// FIXME: added code
    TH1F * processTypesST_;       /// FIXME: added code
    TH1F * processTypesSV_;       /// FIXME: added code
    TH1F * processTypesTP_;       /// FIXME: added code
    TH1F * processTypesTV_;       /// FIXME: added code
    TH1F * NSimVertTV_;       /// FIXME: added code
    TH1F * NSimTrackTP_;       /// FIXME: added code
    TH1F * NSimVertNoTV_;       /// FIXME: added code
    TH1F * NTVperSV_;       /// FIXME: added code
//     TH1F * NSimTrackNoTP_;       /// FIXME: added code
//     TH1F * NSimTrackNoTPEvent_;       /// FIXME: added code
    
//     TH1F * HepMCLabels_;       /// FIXME: added code
    TH1F * HepMCGenVertexs_;       /// FIXME: added code
    TH1F * NGenVertexs_;       /// FIXME: added code
    
    edm::InputTag trackingTruth_;

    Int_t numberTrackCategories_;
};


TrackCategoriesAnalyzer::TrackCategoriesAnalyzer(const edm::ParameterSet& config) : classifier_(config)
{
    // Get the track collection
    trackProducer_ = config.getUntrackedParameter<edm::InputTag> ( "trackProducer" );

    // Get the file service
    edm::Service<TFileService> fs;

    // Create a sub directory associated to the analyzer
    TFileDirectory directory = fs->mkdir( "TrackCategoriesAnalyzer" );

    // Number of track categories
    numberTrackCategories_ = TrackCategories::Unknown+1;

    // Define a new histograms
    trackCategories_ = fs->make<TH1F>(
                           "Frequency",
                           "Frequency for the different track categories",
                           numberTrackCategories_,
                           -0.5,
                           numberTrackCategories_ - 0.5
                       );
    
    processTypesClassifier_ = fs->make<TH1F>("ProcessTypesClassifier", "Process types of the classified tracks", 250, 0, 250); /// FIXME: added code
    processTypesST_ = fs->make<TH1F>("ProcessTypesST", "Process types of all SimTracks in event", 250, 0, 250); /// FIXME: added code
    processTypesSV_ = fs->make<TH1F>("ProcessTypesSV", "Process types of all SimVertexs in event", 250, 0, 250); /// FIXME: added code
    processTypesTP_ = fs->make<TH1F>("ProcessTypesTP", "Process types of all TrackingParticles in event", 250, 0, 250); /// FIXME: added code
    processTypesTV_ = fs->make<TH1F>("ProcessTypesTV", "Process types of all TrackingVertexs in event", 250, 0, 250); /// FIXME: added code
    NSimVertTV_ = fs->make<TH1F>("NSimVertTV", "Number of SimVertexs per TrackingVertex in event", 20, 0, 20); /// FIXME: added code
    NSimTrackTP_ = fs->make<TH1F>("NSimTrackTP", "Number of SimTracks per TrackingParticle in event", 20, 0, 20); /// FIXME: added code
    NSimVertNoTV_ = fs->make<TH1F>("NSimVertNoTV", "Overall number of SimVertexs w/o TrackingVertexs", 10000, 0, 10000); /// FIXME: added code
    NTVperSV_ = fs->make<TH1F>("NTVperSV", "Number of TrackingVertexs per SimVertex", 1000, 0, 1000); /// FIXME: added code
//     NSimTrackNoTP_ = fs->make<TH1F>("NSimTrackNoTP", "Overall number of SimTracks w/o TrackingParticles", 10000, 0, 10000); /// FIXME: added code
//     NSimTrackNoTPEvent_ = fs->make<TH1F>("NSimTrackNoTPEvent", "Number of SimTracks w/o TrackingParticle per event", 1000, 0, 1000); /// FIXME: added code
    
//     HepMCLabels_ = fs->make<TH1F>("HepMCLabels", "Number of HepMCLabels in event", 20, 0, 20); /// FIXME: added code
    HepMCGenVertexs_ = fs->make<TH1F>("HepMCGenVertexs", "Number of HepMC GenVertexs in event", 1000, 0, 1000); /// FIXME: added code
    NGenVertexs_ = fs->make<TH1F>("NGenVertexs", "Number of HepMC GenVertexs per vertex", 30, 0, 30); /// FIXME: added code
    
    trackingTruth_ = config.getUntrackedParameter<edm::InputTag>("trackingTruth");

    // Set the proper categories names
    for (Int_t i = 0; i < numberTrackCategories_; ++i)
        trackCategories_->GetXaxis()->SetBinLabel(i+1, TrackCategories::Names[i]);
}


TrackCategoriesAnalyzer::~TrackCategoriesAnalyzer() { }


void TrackCategoriesAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
    // Track collection
    edm::Handle<edm::View<reco::Track> > trackCollection;
    event.getByLabel(trackProducer_, trackCollection);

    // Set the classifier for a new event
    classifier_.newEvent(event, setup);
    
    /** FIXME: added code below */
    
    typedef math::XYZTLorentzVectorD LorentzVector;
    
    edm::Handle<edm::View<SimVertex> > simVertCollection;
    event.getByLabel("g4SimHits", simVertCollection);
    
    edm::Handle<edm::View<SimTrack> > simTrackCollection;
    event.getByLabel("g4SimHits", simTrackCollection);
    
    edm::Handle<TrackingVertexCollection> TVCollection;
    event.getByLabel(trackingTruth_, TVCollection);
    
    edm::Handle<TrackingParticleCollection> TPCollection;
    event.getByLabel(trackingTruth_, TPCollection);
    
    std::string genProd = "generator";
    edm::Handle<edm::HepMCProduct> hepMCprod;
    
    if (event.getByLabel(genProd, hepMCprod)) {
        const HepMC::GenEvent* genEvent = hepMCprod->GetEvent();
        HepMCGenVertexs_->Fill(genEvent->vertices_size());
    } else 
    {
        std::cout << "No GenVertices found!" << std::endl;
        HepMCGenVertexs_->Fill(0);
    }
    
    int noTVclose = 0;
    
    for (std::size_t i = 0; i < simVertCollection->size(); ++i)
    {
        unsigned int processType = simVertCollection->at(i).processType();
        processTypesSV_->Fill(processType);
        
        int closeTVcount = 0;
        for (std::size_t iTV = 0; iTV < TVCollection->size(); ++iTV)
        {
            LorentzVector diff = simVertCollection->at(i).position() - TVCollection->at(iTV).position();
            if (diff.R() < 0.003) closeTVcount++;
        }
        
        if (closeTVcount ==  0) noTVclose++;
        NTVperSV_->Fill(closeTVcount);
        
    }
    
    NSimVertNoTV_->Fill(noTVclose);
    
    for (std::size_t i = 0; i < simTrackCollection->size(); ++i)
    {
        unsigned int simVertInd = simTrackCollection->at(i).vertIndex();
        unsigned int processType = simVertCollection->at(simVertInd).processType();
        processTypesST_->Fill(processType);
    }
    
    for (std::size_t i = 0; i < TVCollection->size(); ++i)
    {
      NSimVertTV_->Fill(TVCollection->at(i).nG4Vertices());
      NGenVertexs_->Fill(TVCollection->at(i).nGenVertices());
      if (TVCollection->at(i).nG4Vertices() == 0) continue;
      unsigned int processType = TVCollection->at(i).g4Vertices_begin()->processType();
	processTypesTV_->Fill(processType);
    }
    
    for (std::size_t i = 0; i < TPCollection->size(); ++i)
    {
      NSimTrackTP_->Fill(TPCollection->at(i).g4Tracks().size());
      if (TPCollection->at(i).parentVertex()->nG4Vertices() == 0) continue;
      unsigned int processType = TPCollection->at(i).parentVertex()->g4Vertices_begin()->processType();
	processTypesTP_->Fill(processType);
    }
    
    /// FIXME: until here */

    // Loop over the track collection.
    for (std::size_t index = 0; index < trackCollection->size(); index++)
    {
        edm::RefToBase<reco::Track> track(trackCollection, index);

        // Classify the tracks
        classifier_.evaluate(track);
        
        /** FIXME: added code below */
        
        unsigned int processType = 0;
        
        if (classifier_.history().simVertex()->nG4Vertices() > 0)
            processType = classifier_.history().simVertex()->g4Vertices_begin()->processType();
        
        processTypesClassifier_->Fill(processType);
        
        /// FIXME: until here */

        // Fill the histogram with the categories
        for (Int_t i = 0; i != numberTrackCategories_; ++i)
            if (
                classifier_.is( (TrackCategories::Category) i )
            )
                trackCategories_->Fill(i);
    }
}


DEFINE_FWK_MODULE(TrackCategoriesAnalyzer);

