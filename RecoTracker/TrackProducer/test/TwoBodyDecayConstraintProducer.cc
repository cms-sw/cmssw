/**
 *  \class TwoBodyDecayConstraintProducer TwoBodyDecayConstraintProducer.cc RecoTracker/ConstraintProducerTest/src/TwoBodyDecayConstraintProducer.cc
 *  
 *  Description: Produces track parameter constraints for refitting tracks, according to information TwoBodyDecay kinematic fit.
 *
 *  Original Author:  Edmund Widl
 * 
 */

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TrackingTools/PatternTools/interface/TrackConstraintAssociation.h"

#include "Alignment/TwoBodyDecay/interface/TwoBodyDecay.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayFitter.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayVirtualMeasurement.h"
#include "Alignment/ReferenceTrajectories/interface/TwoBodyDecayTrajectoryState.h"

// // debug
// #include <map>
// #include "TH1F.h"
// #include "TFile.h"
// #include "TLorentzVector.h"
// #include "Alignment/TwoBodyDecay/interface/TwoBodyDecayModel.h"


class TwoBodyDecayConstraintProducer: public edm::EDProducer
{

public:

  explicit TwoBodyDecayConstraintProducer(const edm::ParameterSet&);
  ~TwoBodyDecayConstraintProducer();

private:

  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;

  std::pair<bool, TrajectoryStateOnSurface> innermostState( const reco::TransientTrack& ttrack ) const;
  bool match( const TrajectoryStateOnSurface& newTsos, const TrajectoryStateOnSurface& oldTsos ) const;

  const edm::InputTag srcTag_; 
  const edm::InputTag bsSrcTag_;

  TwoBodyDecayFitter tbdFitter_;

  double primaryMass_;
  double primaryWidth_;
  double secondaryMass_;

  double sigmaPositionCutValue_;
  double chi2CutValue_;
  double errorRescaleValue_;

//   // debug
//   std::map<std::string, TH1F*> histos_;
};


TwoBodyDecayConstraintProducer::TwoBodyDecayConstraintProducer( const edm::ParameterSet& iConfig ) :
  srcTag_( iConfig.getParameter<edm::InputTag>( "src" ) ),
  bsSrcTag_( iConfig.getParameter<edm::InputTag>( "beamSpot" ) ),
  tbdFitter_( iConfig ),
  primaryMass_( iConfig.getParameter<double>( "primaryMass" ) ),
  primaryWidth_( iConfig.getParameter<double>( "primaryWidth" ) ),
  secondaryMass_( iConfig.getParameter<double>( "secondaryMass" ) ),
  sigmaPositionCutValue_( iConfig.getParameter<double>( "sigmaPositionCut" ) ),
  chi2CutValue_( iConfig.getParameter<double>( "chi2Cut" ) ),
  errorRescaleValue_( iConfig.getParameter<double>( "rescaleError" ) )

{
  produces<std::vector<TrackParamConstraint> >();
  produces<TrackParamConstraintAssociationCollection>();

//   //debug
//   histos_["deltaEta1"] = new TH1F( "deltaEta1", "deltaEta1", 200, -1., 1. );
//   histos_["deltaP1"] = new TH1F( "deltaP1", "deltaP1", 200, -50., 50. );

//   histos_["deltaEta2"] = new TH1F( "deltaEta2", "deltaEta2", 200, -1., 1. );
//   histos_["deltaP2"] = new TH1F( "deltaP2", "deltaP2", 200, -50., 50. );
}


TwoBodyDecayConstraintProducer::~TwoBodyDecayConstraintProducer()
{
//   // debug
//   TFile* f = new TFile( "producer.root", "RECREATE" );
//   f->cd();
//   for ( std::map<std::string, TH1F*>::iterator it = histos_.begin(); it != histos_.end(); ++it ) { it->second->Write(); delete it->second; }
//   f->Close();
//   delete f;
}


void TwoBodyDecayConstraintProducer::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace edm;

  Handle<reco::TrackCollection> trackColl;
  iEvent.getByLabel( srcTag_, trackColl );

  Handle<reco::BeamSpot> beamSpot;
  iEvent.getByLabel( bsSrcTag_, beamSpot );

  ESHandle<MagneticField> magField;
  iSetup.get<IdealMagneticFieldRecord>().get( magField );

  std::auto_ptr<std::vector<TrackParamConstraint> > pairs(new std::vector<TrackParamConstraint>);
  std::auto_ptr<TrackParamConstraintAssociationCollection> output(new TrackParamConstraintAssociationCollection);
  
  edm::RefProd<std::vector<TrackParamConstraint> > rPairs = iEvent.getRefBeforePut<std::vector<TrackParamConstraint> >();
  
  if ( trackColl->size() == 2 )
  {
    /// Construct virtual measurement (for TBD)
    TwoBodyDecayVirtualMeasurement vm( primaryMass_, primaryWidth_, secondaryMass_, *beamSpot.product() );

    /// Get transient tracks from track collection
    std::vector<reco::TransientTrack> ttracks(2);
    ttracks[0] = reco::TransientTrack( reco::TrackRef( trackColl, 0 ), magField.product() );
    ttracks[0].setES( iSetup );
    ttracks[1] = reco::TransientTrack( reco::TrackRef( trackColl, 1 ), magField.product() );
    ttracks[1].setES( iSetup );

    /// Fit the TBD
    TwoBodyDecay tbd = tbdFitter_.estimate( ttracks, vm );

    if ( !tbd.isValid() or ( tbd.chi2() > chi2CutValue_ ) ) return;

    /// Get the innermost trajectory states
    std::pair<bool, TrajectoryStateOnSurface> oldInnermostState1 = innermostState( ttracks[0] );
    std::pair<bool, TrajectoryStateOnSurface> oldInnermostState2 = innermostState( ttracks[1] );
    if ( !oldInnermostState1.second.isValid() || !oldInnermostState2.second.isValid() ) return;

    /// Construct the TBD trajectory states
    TwoBodyDecayTrajectoryState::TsosContainer trackTsos( oldInnermostState1.second, oldInnermostState2.second );
    TwoBodyDecayTrajectoryState tbdTrajState( trackTsos, tbd, secondaryMass_, magField.product(), true );
    if ( !tbdTrajState.isValid() ) return;

    /// Match the old and the new estimates for the trajectory state
    bool match1 = match( tbdTrajState.trajectoryStates(true).first, oldInnermostState1.second );
    bool match2 = match( tbdTrajState.trajectoryStates(true).second, oldInnermostState2.second );
    if ( !match1 || !match2 ) return;

    // re-scale error of constraintTsos
    tbdTrajState.rescaleError( errorRescaleValue_ );

    // produce constraint for first track
    pairs->push_back(tbdTrajState.trajectoryStates(true).first);
    output->insert( reco::TrackRef(trackColl,0), edm::Ref<std::vector<TrackParamConstraint> >(rPairs,0) );

    // produce constraint for second track
    pairs->push_back(tbdTrajState.trajectoryStates(true).second);
    output->insert( reco::TrackRef(trackColl,1), edm::Ref<std::vector<TrackParamConstraint> >(rPairs,1) );

//     // debug
//     if ( tbd.isValid() ) {
//       TwoBodyDecayModel model;
//       const std::pair< AlgebraicVector, AlgebraicVector > fitMomenta = model.cartesianSecondaryMomenta( tbd );

//       TLorentzVector recoMomentum1( ttracks[0].track().px(), ttracks[0].track().py(), ttracks[0].track().pz(),
// 				    sqrt((ttracks[0].track().p()*ttracks[0].track().p())+0.105658*0.105658) );
//       TLorentzVector fitMomentum1( fitMomenta.first[0], fitMomenta.first[1], fitMomenta.first[2],
// 				   sqrt( fitMomenta.first.normsq()+0.105658*0.105658) );
//       histos_["deltaP1"]->Fill( recoMomentum1.P() - fitMomentum1.P() );
//       histos_["deltaEta1"]->Fill( recoMomentum1.Eta() - fitMomentum1.Eta() );

//       TLorentzVector recoMomentum2( ttracks[1].track().px(), ttracks[1].track().py(), ttracks[1].track().pz(),
// 				    sqrt((ttracks[1].track().p()*ttracks[1].track().p())+0.105658*0.105658) );
//       TLorentzVector fitMomentum2( fitMomenta.second[0], fitMomenta.second[1], fitMomenta.second[2],
// 				   sqrt( fitMomenta.second.normsq()+0.105658*0.105658) );
//       histos_["deltaP2"]->Fill( recoMomentum2.P() - fitMomentum2.P() );
//       histos_["deltaEta2"]->Fill( recoMomentum2.Eta() - fitMomentum2.Eta() );
//     }
  }
  
  iEvent.put(pairs);
  iEvent.put(output);
}


void TwoBodyDecayConstraintProducer::endJob() {}


std::pair<bool, TrajectoryStateOnSurface>
TwoBodyDecayConstraintProducer::innermostState( const reco::TransientTrack& ttrack ) const
{
  double outerR = ttrack.outermostMeasurementState().globalPosition().perp();
  double innerR = ttrack.innermostMeasurementState().globalPosition().perp();
  bool insideOut = ( outerR > innerR );
  return std::make_pair( insideOut, insideOut ? ttrack.innermostMeasurementState() : ttrack.outermostMeasurementState() );
}


bool
TwoBodyDecayConstraintProducer::match( const TrajectoryStateOnSurface& newTsos,
				       const TrajectoryStateOnSurface& oldTsos ) const
{
  LocalPoint lp1 = newTsos.localPosition();
  LocalPoint lp2 = oldTsos.localPosition();

  double deltaX = lp1.x() - lp2.x();
  double deltaY = lp1.y() - lp2.y();

  return ( ( fabs(deltaX) < sigmaPositionCutValue_ ) && ( fabs(deltaY) < sigmaPositionCutValue_ ) );
}


//define this as a plug-in
DEFINE_FWK_MODULE(TwoBodyDecayConstraintProducer);
