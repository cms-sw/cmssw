
#include <map>
#include <string>

#include "TH1F.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "SimTracker/TrackHistory/interface/VertexClassifier.h"

//
// class decleration
//

class VertexValidationAnalyzer : public edm::EDAnalyzer
{
public:

    explicit VertexValidationAnalyzer(const edm::ParameterSet&);

private:

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    // Member data

    VertexClassifier classifier_;

    Int_t numberVertexCategories_;

    edm::InputTag vertexProducer_;

    std::map<const char*, TH1D *> TH1Index_;
};


VertexValidationAnalyzer::VertexValidationAnalyzer(const edm::ParameterSet& config) : classifier_(config)
{
    // Get the track collection
    vertexProducer_ = config.getUntrackedParameter<edm::InputTag>("vertexProducer");

    // Get the file service
    edm::Service<TFileService> fs;

    // Number of track categories
    numberVertexCategories_ = VertexCategories::Unknown+1;

    // Define a histogram
    TH1Index_["VertexCategories"] = fs->make<TH1D>(
                                        "VertexCategories",
                                        "Frequency for the different track categories",
                                        numberVertexCategories_,
                                        -0.5,
                                        numberVertexCategories_ - 0.5
                                    );

    // Set the proper categories names
    for (Int_t i = 0; i < numberVertexCategories_; ++i)
        TH1Index_["VertexCategories"]->GetXaxis()->SetBinLabel(i+1, VertexCategories::Names[i]);

    // Define histograms
    TH1Index_["VertexPullx"] = fs->make<TH1D>(
                                   "VertexPullx", "VertexPullx", 50, -10., 10.
                               );
    TH1Index_["VertexPully"] = fs->make<TH1D>(
                                   "VertexPully", "VertexPully", 50, -10., 10.
                               );
    TH1Index_["VertexPullz"] = fs->make<TH1D>(
                                   "VertexPullz", "VertexPullz", 50, -10., 10.
                               );

    TH1Index_["VertexPullxBWeakDecay"] = fs->make<TH1D>(
                                             "VertexPullxBWeakDecay", "VertexPullxBWeakDecay", 50, -10., 10.
                                         );
    TH1Index_["VertexPullyBWeakDecay"] = fs->make<TH1D>(
                                             "VertexPullyBWeakDecay", "VertexPullyBWeakDecay", 50, -10., 10.
                                         );
    TH1Index_["VertexPullzBWeakDecay"] = fs->make<TH1D>(
                                             "VertexPullzBWeakDecay", "VertexPullzBWeakDecay", 50, -10., 10.
                                         );

    TH1Index_["VertexPullxCWeakDecay"] = fs->make<TH1D>(
                                             "VertexPullxCWeakDecay", "VertexPullxCWeakDecay", 50, -10., 10.
                                         );
    TH1Index_["VertexPullyCWeakDecay"] = fs->make<TH1D>(
                                             "VertexPullyCWeakDecay", "VertexPullyCWeakDecay", 50, -10., 10.
                                         );
    TH1Index_["VertexPullzCWeakDecay"] = fs->make<TH1D>(
                                             "VertexPullzCWeakDecay", "VertexPullzCWeakDecay", 50, -10., 10.
                                         );

    TH1Index_["VertexPullxLight"] = fs->make<TH1D>(
                                        "VertexPullxLight", "VertexPullxLight", 50, -10., 10.
                                    );
    TH1Index_["VertexPullyLight"] = fs->make<TH1D>(
                                        "VertexPullyLight", "VertexPullyLight", 50, -10., 10.
                                    );
    TH1Index_["VertexPullzLight"] = fs->make<TH1D>(
                                        "VertexPullzLight", "VertexPullzLight", 50, -10., 10.
                                    );

}


void VertexValidationAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
    // Set the classifier for a new event
    classifier_.newEvent(event, setup);

    // Vertex collection
    edm::Handle<reco::VertexCollection> vertexCollection;
    event.getByLabel(vertexProducer_, vertexCollection);

    // Get a constant reference to the track history associated to the classifier
    VertexHistory const & tracer = classifier_.history();

    // Loop over the track collection.
    for (std::size_t index = 0; index < vertexCollection->size(); index++)
    {
        reco::VertexRef vertex(vertexCollection, index);

        // Classify the tracks
        classifier_.evaluate(vertex);

        // Fill the histogram with the categories
        for (Int_t i = 0; i != numberVertexCategories_; ++i)
            if (
                classifier_.is( (VertexCategories::Category) i )
            )
                TH1Index_["VertexCategories"]->Fill(i);

        if ( !classifier_.is(VertexCategories::Fake) )
        {
            // Getting a constant reference to the simulated vertex
            const TrackingVertexRef & simVertex = tracer.simVertex();

            // Calculating PullX, PullY, PullZ
            double pullx = (vertex->x() - simVertex->position().x())/vertex->xError();
            double pully = (vertex->y() - simVertex->position().y())/vertex->yError();
            double pullz = (vertex->z() - simVertex->position().z())/vertex->zError();

            // Fill histograms
            TH1Index_["VertexPullx"]->Fill(pullx);
            TH1Index_["VertexPully"]->Fill(pully);
            TH1Index_["VertexPullz"]->Fill(pullz);

            // Fill histograms by categories
            if ( classifier_.is(VertexCategories::BWeakDecay) )
            {
                TH1Index_["VertexPullxBWeakDecay"]->Fill(pullx);
                TH1Index_["VertexPullyBWeakDecay"]->Fill(pully);
                TH1Index_["VertexPullzBWeakDecay"]->Fill(pullz);
            }
            else if ( classifier_.is(VertexCategories::CWeakDecay) )
            {
                TH1Index_["VertexPullxCWeakDecay"]->Fill(pullx);
                TH1Index_["VertexPullyCWeakDecay"]->Fill(pully);
                TH1Index_["VertexPullzCWeakDecay"]->Fill(pullz);
            }
            else
            {
                TH1Index_["VertexPullxLight"]->Fill(pullx);
                TH1Index_["VertexPullyLight"]->Fill(pully);
                TH1Index_["VertexPullzLight"]->Fill(pullz);
            }

        }
    }
}

DEFINE_ANOTHER_FWK_MODULE(VertexValidationAnalyzer);
