/*
 *  TrackingParticleCategoriesAnalyzer.C
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

class TrackingParticleCategoriesAnalyzer : public edm::EDAnalyzer
{
public:

    explicit TrackingParticleCategoriesAnalyzer(const edm::ParameterSet&);
    ~TrackingParticleCategoriesAnalyzer();

private:

    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

    // Member data

    edm::InputTag trackingTruth_;

    std::size_t totalTrakingParticles_;

    TrackClassifier classifier_;

    TH1F * trackingParticleCategories_;

    Int_t numberTrackingParticleCategories_;
};


TrackingParticleCategoriesAnalyzer::TrackingParticleCategoriesAnalyzer(const edm::ParameterSet& config) : classifier_(config,consumesCollector())
{
    // Get the track collection
    trackingTruth_ = config.getUntrackedParameter<edm::InputTag> ( "trackingTruth" );
    consumes<TrackingParticleCollection>(trackingTruth_);

    // Get the file service
    edm::Service<TFileService> fs;

    // Create a sub directory associated to the analyzer
    TFileDirectory directory = fs->mkdir( "TrackingParticleCategoriesAnalyzer" );

    // Number of track categories
    numberTrackingParticleCategories_ = TrackCategories::Unknown+1;

    // Define a new histograms
    trackingParticleCategories_ = fs->make<TH1F>(
                           "Frequency",
                           "Frequency for the different track categories",
                           numberTrackingParticleCategories_,
                           -0.5,
                           numberTrackingParticleCategories_ - 0.5
                       );

    // Set the proper categories names
    for (Int_t i = 0; i < numberTrackingParticleCategories_; ++i)
        trackingParticleCategories_->GetXaxis()->SetBinLabel(i+1, TrackCategories::Names[i]);
}


TrackingParticleCategoriesAnalyzer::~TrackingParticleCategoriesAnalyzer() { }


void TrackingParticleCategoriesAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
    // Track collection
    edm::Handle<TrackingParticleCollection> TPCollection;
    event.getByLabel(trackingTruth_, TPCollection);

    // Set the classifier for a new event
    classifier_.newEvent(event, setup);

    // Loop over the track collection.
    for (std::size_t index = 0; index < TPCollection->size(); index++)
    {
        TrackingParticleRef trackingParticle(TPCollection, index);

        // Classify the tracks
        classifier_.evaluate(trackingParticle);

        // Fill the histogram with the categories
        for (Int_t i = 0; i != numberTrackingParticleCategories_; ++i)
            if (
                classifier_.is( (TrackCategories::Category) i )
            )
                trackingParticleCategories_->Fill(i);
    }
}


DEFINE_FWK_MODULE(TrackingParticleCategoriesAnalyzer);
