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

    Int_t numberTrackCategories_;
};


TrackCategoriesAnalyzer::TrackCategoriesAnalyzer(const edm::ParameterSet& config) : classifier_(config,consumesCollector())
{
    // Get the track collection
    trackProducer_ = config.getUntrackedParameter<edm::InputTag> ( "trackProducer" );
    consumes<edm::View<reco::Track>>(trackProducer_);

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

    // Loop over the track collection.
    for (std::size_t index = 0; index < trackCollection->size(); index++)
    {
        edm::RefToBase<reco::Track> track(trackCollection, index);

        // Classify the tracks
        classifier_.evaluate(track);

        // Fill the histogram with the categories
        for (Int_t i = 0; i != numberTrackCategories_; ++i)
            if (
                classifier_.is( (TrackCategories::Category) i )
            )
                trackCategories_->Fill(i);
    }
}


DEFINE_FWK_MODULE(TrackCategoriesAnalyzer);

