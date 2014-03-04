/* L1ExtraMaker Creates L1 Extra Objects from Clusters and jets

   M.Bachtis,S.Dasu University of Wisconsin-Madison */


// system include files
#include <memory>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeed.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeedFwd.h"


#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "TGraph.h"


class L1CaloGeometry;

class L1NewEgammaExtraCalibrator:public edm::EDProducer
{
    public:
        explicit L1NewEgammaExtraCalibrator( const edm::ParameterSet & );
        ~L1NewEgammaExtraCalibrator(  );

    private:

        virtual void produce( edm::Event &, const edm::EventSetup & );
        virtual void endJob(  );

        edm::InputTag mEgamma;
        edm::InputTag mIsoegamma;

        std::vector < double >mEgammanewcorr;
        std::vector < double >mEgammaneweta;

        TGraph* mNewCalibration;

        void calibrateP4( reco::LeafCandidate & );


};





L1NewEgammaExtraCalibrator::L1NewEgammaExtraCalibrator( const edm::ParameterSet & iConfig ):
    mEgamma( iConfig.getParameter < edm::InputTag > ( "eGamma" ) ),
    mIsoegamma( iConfig.getParameter < edm::InputTag > ( "isoEGamma" ) ),
    mEgammanewcorr( iConfig.getParameter < std::vector < double > > ( "eGammaNewCorr" ) ),
    mEgammaneweta( iConfig.getParameter < std::vector < double > > ( "eGammaEtaPoints" ) ),
    mNewCalibration(NULL)
{
    // Register Product
    produces < l1extra::L1EmParticleCollection > ( "EGamma" );
    produces < l1extra::L1EmParticleCollection > ( "IsoEGamma" );

    //double x[10] = {0.125,0.375,0.625,0.875,1.125,1.3645,1.6145,1.875,2.125,2.375};
    //double y[10] = {0.0952467,0.101389,0.10598,0.12605,0.162749,0.193123,0.249227,0.2800289,0.271548,0.27855};
    int nPoints = std::min(mEgammaneweta.size(),mEgammanewcorr.size());
    mNewCalibration = new TGraph(nPoints, &mEgammaneweta[0], &mEgammanewcorr[0]);

}


L1NewEgammaExtraCalibrator::~L1NewEgammaExtraCalibrator(  )
{
    delete mNewCalibration;
}


void L1NewEgammaExtraCalibrator::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
    // egamma
    edm::Handle < l1extra::L1EmParticleCollection > eg;
    if ( iEvent.getByLabel( mEgamma, eg ) )
    {
        std::auto_ptr < l1extra::L1EmParticleCollection > l1EGamma( new l1extra::L1EmParticleCollection );
        for ( l1extra::L1EmParticleCollection::const_iterator lIt = eg->begin(  ) ; lIt != eg->end() ; ++lIt )
        {
            l1extra::L1EmParticle p( *lIt );
            //double lAbsEta = fabs( p.eta(  ) );
            //if ( lAbsEta < 2.6 )
            //{
                calibrateP4( p );
            //}
            l1EGamma->push_back( p );
        }
        iEvent.put( l1EGamma, "EGamma" );
    }

    // isolated egamma
    edm::Handle < l1extra::L1EmParticleCollection > ieg;
    if ( iEvent.getByLabel( mIsoegamma, ieg ) )
    {
        std::auto_ptr < l1extra::L1EmParticleCollection > l1IsoEGamma( new l1extra::L1EmParticleCollection );
        for ( l1extra::L1EmParticleCollection::const_iterator lIt = ieg->begin(  ) ; lIt != ieg->end() ; ++lIt )
        {
            l1extra::L1EmParticle p( *lIt );
            //double lAbsEta = fabs( p.eta(  ) );
            //if( lAbsEta < 2.6 )
            //{
                calibrateP4( p );
            //}

            l1IsoEGamma->push_back( p );
        }
        iEvent.put( l1IsoEGamma, "IsoEGamma" );
    }


}

// ------------ method called once each job just after ending the event loop ------------
void L1NewEgammaExtraCalibrator::endJob(  )
{
}



void L1NewEgammaExtraCalibrator::calibrateP4( reco::LeafCandidate & p )
{
    double lAbsEta = fabs( p.eta(  ) ) ;
    double etaMax = mNewCalibration->GetX()[mNewCalibration->GetN()-1];
    if(lAbsEta>etaMax) lAbsEta = etaMax; // don't extrapolate the calibration
    double factor = ( (mNewCalibration && mNewCalibration->Eval(lAbsEta)!=1) ? 1./(1.-mNewCalibration->Eval(lAbsEta)) : 1. );
    p.setP4( math::PtEtaPhiMLorentzVector( factor * p.pt(  ), p.eta(  ), p.phi(  ), 0.0 ) );
}


// #define DEFINE_ANOTHER_FWK_MODULE(type) DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type); DEFINE_FWK_PSET_DESC_FILLER(type)
DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1NewEgammaExtraCalibrator >, "L1NewEgammaExtraCalibrator" );
DEFINE_FWK_PSET_DESC_FILLER( L1NewEgammaExtraCalibrator );
// DEFINE_ANOTHER_FWK_MODULE(L1NewEgammaExtraCalibrator);
