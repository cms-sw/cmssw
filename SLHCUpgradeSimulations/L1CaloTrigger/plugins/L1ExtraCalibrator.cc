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
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"

#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "TGraph.h"


class L1CaloGeometry;

class L1ExtraCalibrator:public edm::EDProducer
{
    public:
        explicit L1ExtraCalibrator( const edm::ParameterSet & );
        ~L1ExtraCalibrator(  );

    private:

        virtual void produce( edm::Event &, const edm::EventSetup & );
        virtual void endJob(  );

        edm::InputTag mEgamma;
        edm::InputTag mIsoegamma;
        edm::InputTag mTaus;
        edm::InputTag mIsotaus;
        edm::InputTag mJets;

        std::vector < double >mEgammacoeffb;
        std::vector < double >mTaucoeffb;
        std::vector < double >mEgammacoeffe;
        std::vector < double >mTaucoeffe;
        std::vector < double >mEgammabincorr;
        std::vector < double >mTaubincorr;

        bool mApplyNewCalib;
        std::vector < double >mEgammanewcorr;
        std::vector < double >mEgammaneweta;

        TGraph* mNewCalibration;

        void calibrateP4( reco::LeafCandidate &, const std::vector < double >&, const std::vector < double >& );
        void calibrateP4_New( reco::LeafCandidate & );


};





L1ExtraCalibrator::L1ExtraCalibrator( const edm::ParameterSet & iConfig ):mEgamma( iConfig.getParameter < edm::InputTag > ( "eGamma" ) ),
mIsoegamma( iConfig.getParameter < edm::InputTag > ( "isoEGamma" ) ),
mTaus( iConfig.getParameter < edm::InputTag > ( "taus" ) ),
mIsotaus( iConfig.getParameter < edm::InputTag > ( "isoTaus" ) ),
mJets( iConfig.getParameter < edm::InputTag > ( "jets" ) ),
mEgammacoeffb( iConfig.getParameter < std::vector < double > > ( "eGammaCoefficientsB" ) ),
mTaucoeffb( iConfig.getParameter < std::vector < double > > ( "tauCoefficientsB" ) ),
mEgammacoeffe( iConfig.getParameter < std::vector < double > > ( "eGammaCoefficientsE" ) ),
mTaucoeffe( iConfig.getParameter < std::vector < double > > ( "tauCoefficientsE" ) ),
mEgammabincorr( iConfig.getParameter < std::vector < double > > ( "eGammaBinCorr" ) ), 
mTaubincorr( iConfig.getParameter < std::vector < double > > ( "tauBinCorr" ) ),
mApplyNewCalib( iConfig.getParameter < bool > ("applyNewCalib") ),
mEgammanewcorr( iConfig.getParameter < std::vector < double > > ( "eGammaNewCorr" ) ),
mEgammaneweta( iConfig.getParameter < std::vector < double > > ( "eGammaEtaPoints" ) ),
mNewCalibration(NULL)
{
    // Register Product
    produces < l1extra::L1EmParticleCollection > ( "EGamma" );
    produces < l1extra::L1EmParticleCollection > ( "IsoEGamma" );
    produces < l1extra::L1JetParticleCollection > ( "Taus" );
    produces < l1extra::L1JetParticleCollection > ( "IsoTaus" );
    produces < l1extra::L1JetParticleCollection > ( "Jets" );

    if(mApplyNewCalib)
    {
        int nPoints = std::min(mEgammaneweta.size(),mEgammanewcorr.size());
        mNewCalibration = new TGraph(nPoints, &mEgammaneweta[0], &mEgammanewcorr[0]);
    }

}


L1ExtraCalibrator::~L1ExtraCalibrator(  )
{
    delete mNewCalibration;
}


void L1ExtraCalibrator::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
    edm::Handle < l1extra::L1EmParticleCollection > eg;
    if ( iEvent.getByLabel( mEgamma, eg ) )
    {
        std::auto_ptr < l1extra::L1EmParticleCollection > l1EGamma( new l1extra::L1EmParticleCollection );
        for ( l1extra::L1EmParticleCollection::const_iterator lIt = eg->begin(  ) ; lIt != eg->end() ; ++lIt )
        {
            l1extra::L1EmParticle p( *lIt );
            double lAbsEta = fabs( p.eta(  ) );
            if(mApplyNewCalib)
            {
                if ( lAbsEta < 2.6 )
                {
                    calibrateP4_New ( p );
                }
            }
            else
            {
                // Pass E or B coefficients depending on p.eta().
                if ( lAbsEta < 1.6 )
                {
                    calibrateP4 ( p, mEgammacoeffb, mEgammabincorr );
                }
                else if ( lAbsEta < 2.6 )
                {
                    calibrateP4 ( p, mEgammacoeffe, mEgammabincorr );
                }
            }    
            l1EGamma->push_back( p );
        }
        iEvent.put( l1EGamma, "EGamma" );
    }

    edm::Handle < l1extra::L1EmParticleCollection > ieg;
    if ( iEvent.getByLabel( mIsoegamma, ieg ) )
    {
        std::auto_ptr < l1extra::L1EmParticleCollection > l1IsoEGamma( new l1extra::L1EmParticleCollection );
        for ( l1extra::L1EmParticleCollection::const_iterator lIt = ieg->begin(  ) ; lIt != ieg->end() ; ++lIt )
        {
            l1extra::L1EmParticle p( *lIt );
            double lAbsEta = fabs( p.eta(  ) );
            if(mApplyNewCalib)
            {
                if( lAbsEta < 2.6 )
                {
                    calibrateP4_New ( p );
                }
            }
            else
            {
                if ( lAbsEta < 1.6 )
                {
                    calibrateP4 ( p, mEgammacoeffb, mEgammabincorr );
                }
                else if ( lAbsEta < 2.6 )
                {
                    calibrateP4 ( p, mEgammacoeffe, mEgammabincorr );
                }
            }

            l1IsoEGamma->push_back( p );
        }
        iEvent.put( l1IsoEGamma, "IsoEGamma" );
    }

    edm::Handle < l1extra::L1JetParticleCollection > tau;
    if ( iEvent.getByLabel( mTaus, tau ) )
    {
        std::auto_ptr < l1extra::L1JetParticleCollection > l1Tau( new l1extra::L1JetParticleCollection );
        for ( l1extra::L1JetParticleCollection::const_iterator lIt = tau->begin(  ) ; lIt != tau->end() ; ++lIt )
        {
            l1extra::L1JetParticle p( *lIt );
            double lAbsEta = fabs( p.eta(  ) );

            if ( lAbsEta < 1.6 )
            {
                calibrateP4 ( p, mTaucoeffb, mTaubincorr );
            }
            else if ( lAbsEta < 2.6 )
            {
                calibrateP4 ( p, mTaucoeffe, mTaubincorr );
            }
            l1Tau->push_back( p );
        }
        iEvent.put( l1Tau, "Taus" );
    }

    edm::Handle < l1extra::L1JetParticleCollection > itau;
    if ( iEvent.getByLabel( mIsotaus, itau ) )
    {
        std::auto_ptr < l1extra::L1JetParticleCollection > l1IsoTau( new l1extra::L1JetParticleCollection );
        for ( l1extra::L1JetParticleCollection::const_iterator lIt = itau->begin(  ) ; lIt != itau->end() ; ++lIt )
        {
            l1extra::L1JetParticle p( *lIt );
            double lAbsEta = fabs( p.eta(  ) );

            if ( lAbsEta < 1.6 )
            {
                calibrateP4 ( p, mTaucoeffb, mTaubincorr );
            }
            else if ( lAbsEta < 2.6 )
            {
                calibrateP4 ( p, mTaucoeffe, mTaubincorr );
            }
            l1IsoTau->push_back( p );
        }
        iEvent.put( l1IsoTau, "IsoTaus" );
    }

    edm::Handle < l1extra::L1JetParticleCollection > jets;
    if ( iEvent.getByLabel( mJets, jets ) )
    {
        std::auto_ptr < l1extra::L1JetParticleCollection > l1Jet( new l1extra::L1JetParticleCollection );
        for ( l1extra::L1JetParticleCollection::const_iterator lIt = jets->begin(  ) ; lIt != jets->end() ; ++lIt )
        {
            l1Jet->push_back( *lIt );
        }
        iEvent.put( l1Jet, "Jets" );
    }



}

// ------------ method called once each job just after ending the event loop ------------
void L1ExtraCalibrator::endJob(  )
{
}


void L1ExtraCalibrator::calibrateP4( reco::LeafCandidate & p , const std::vector < double >&coeffs, const std::vector < double >&binCorrs )
{
// Function is never called if lAbsEta >=2.6

    double lAbsEta ( fabs( p.eta(  ) ) );
    double bfactor = binCorrs.at( int( lAbsEta * 5.0 ) );

    if( lAbsEta >= 1.6)    lAbsEta -= 1.6;
    
    double factor( coeffs.at( 0 ) + ( coeffs.at( 1 ) * lAbsEta ) + ( coeffs.at( 2 ) * lAbsEta * lAbsEta ) );

    p.setP4( math::PtEtaPhiMLorentzVector( factor * bfactor * p.pt(  ), p.eta(  ), p.phi(  ), 0.0 ) );
}

void L1ExtraCalibrator::calibrateP4_New( reco::LeafCandidate & p )
{
    double lAbsEta ( fabs( p.eta(  ) ) );
    double factor = ( (mNewCalibration && mNewCalibration->Eval(lAbsEta)!=1) ? 1./(1.-mNewCalibration->Eval(lAbsEta)) : 1. );
    p.setP4( math::PtEtaPhiMLorentzVector( factor * p.pt(  ), p.eta(  ), p.phi(  ), 0.0 ) );
}


// #define DEFINE_ANOTHER_FWK_MODULE(type) DEFINE_EDM_PLUGIN (edm::MakerPluginFactory,edm::WorkerMaker<type>,#type); DEFINE_FWK_PSET_DESC_FILLER(type)
DEFINE_EDM_PLUGIN( edm::MakerPluginFactory, edm::WorkerMaker < L1ExtraCalibrator >, "L1ExtraCalibrator" );
DEFINE_FWK_PSET_DESC_FILLER( L1ExtraCalibrator );
// DEFINE_ANOTHER_FWK_MODULE(L1ExtraCalibrator);
