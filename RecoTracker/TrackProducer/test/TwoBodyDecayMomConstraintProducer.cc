/**
 *  \class TwoBodyDecayMomConstraintProducer TwoBodyDecayMomConstraintProducer.cc RecoTracker/ConstraintProducerTest/src/TwoBodyDecayMomConstraintProducer.cc
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
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayModel.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayFitter.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayVirtualMeasurement.h"
#include "Alignment/ReferenceTrajectories/interface/TwoBodyDecayTrajectoryState.h"

// // debug
// #include <map>
// #include "TH1F.h"
// #include "TFile.h"
// #include "TLorentzVector.h"
// #include "Alignment/TwoBodyDecay/interface/TwoBodyDecayModel.h"


class TwoBodyDecayMomConstraintProducer: public edm::EDProducer
{

public:

  explicit TwoBodyDecayMomConstraintProducer(const edm::ParameterSet&);
  ~TwoBodyDecayMomConstraintProducer();

private:

  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;

  std::pair<double, double> momentaAtVertex( const TwoBodyDecay& tbd ) const;

  std::pair<double, double> momentaAtInnermostSurface( const TwoBodyDecay& tbd,
						       const std::vector<reco::TransientTrack>& ttracks,
						       const MagneticField* magField ) const;

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
  double fixedMomentumError_;

  enum MomentumForRefitting { atVertex, atInnermostSurface };
  MomentumForRefitting momentumForRefitting_;

//   // debug
//   std::map<std::string, TH1F*> histos_;
};


TwoBodyDecayMomConstraintProducer::TwoBodyDecayMomConstraintProducer( const edm::ParameterSet& iConfig ) :
  srcTag_( iConfig.getParameter<edm::InputTag>( "src" ) ),
  bsSrcTag_( iConfig.getParameter<edm::InputTag>( "beamSpot" ) ),
  tbdFitter_( iConfig ),
  primaryMass_( iConfig.getParameter<double>( "primaryMass" ) ),
  primaryWidth_( iConfig.getParameter<double>( "primaryWidth" ) ),
  secondaryMass_( iConfig.getParameter<double>( "secondaryMass" ) ),
  sigmaPositionCutValue_( iConfig.getParameter<double>( "sigmaPositionCut" ) ),
  chi2CutValue_( iConfig.getParameter<double>( "chi2Cut" ) ),
  fixedMomentumError_( iConfig.getParameter<double>( "fixedMomentumError" ) )
{
  std::string strMomentumForRefitting = ( iConfig.getParameter<std::string>( "momentumForRefitting" ) );
  if ( strMomentumForRefitting == "atVertex" ) {
    momentumForRefitting_ = atVertex;
  } else if ( strMomentumForRefitting == "atInnermostSurface" ) {
    momentumForRefitting_ = atInnermostSurface;
  } else {
    throw cms::Exception("TwoBodyDecayMomConstraintProducer") << "value of config  variable 'momentumForRefitting': "
							      << "has to be 'atVertex' or 'atInnermostSurface'";
  }

  produces< std::vector<MomentumConstraint> >();
  produces<TrackMomConstraintAssociationCollection>();

//   //debug
//   histos_["deltaEta1"] = new TH1F( "deltaEta1", "deltaEta1", 200, -1., 1. );
//   histos_["deltaP1"] = new TH1F( "deltaP1", "deltaP1", 200, -50., 50. );

//   histos_["deltaEta2"] = new TH1F( "deltaEta2", "deltaEta2", 200, -1., 1. );
//   histos_["deltaP2"] = new TH1F( "deltaP2", "deltaP2", 200, -50., 50. );
}


TwoBodyDecayMomConstraintProducer::~TwoBodyDecayMomConstraintProducer()
{
//   // debug
//   TFile* f = new TFile( "producer_mom.root", "RECREATE" );
//   f->cd();
//   for ( std::map<std::string, TH1F*>::iterator it = histos_.begin(); it != histos_.end(); ++it ) { it->second->Write(); delete it->second; }
//   f->Close();
//   delete f;
}


void TwoBodyDecayMomConstraintProducer::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace edm;

  Handle<reco::TrackCollection> trackColl;
  iEvent.getByLabel( srcTag_, trackColl );

  Handle<reco::BeamSpot> beamSpot;
  iEvent.getByLabel( bsSrcTag_, beamSpot );

  ESHandle<MagneticField> magField;
  iSetup.get<IdealMagneticFieldRecord>().get( magField );

  std::auto_ptr<std::vector<MomentumConstraint> > pairs(new std::vector<MomentumConstraint>);
  std::auto_ptr<TrackMomConstraintAssociationCollection> output(new TrackMomConstraintAssociationCollection);
  
  edm::RefProd<std::vector<MomentumConstraint> > rPairs = iEvent.getRefBeforePut<std::vector<MomentumConstraint> >();
  
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

    std::pair<double, double> fitMomenta;
    if ( momentumForRefitting_ == atVertex ) {
      fitMomenta = momentaAtVertex( tbd );
    } else if ( momentumForRefitting_ == atInnermostSurface ) {
      fitMomenta = momentaAtInnermostSurface( tbd, ttracks, magField.product() );
    } // no other possibility!!!

    if ( ( fitMomenta.first > 0. ) and ( fitMomenta.second > 0. ) )
    {
      // produce constraint for first track
      MomentumConstraint constraint1( fitMomenta.first, fixedMomentumError_ );
      pairs->push_back( constraint1 );
      output->insert( reco::TrackRef(trackColl,0), edm::Ref<std::vector<MomentumConstraint> >(rPairs,0) );

      // produce constraint for second track
      MomentumConstraint constraint2( fitMomenta.second, fixedMomentumError_ );
      pairs->push_back( constraint2 );
      output->insert( reco::TrackRef(trackColl,1), edm::Ref<std::vector<MomentumConstraint> >(rPairs,1) );
    }

//     // debug
//     if ( tbd.isValid() and ( fitMomenta.first > 0. ) and ( fitMomenta.second > 0. ) ) {
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


void TwoBodyDecayMomConstraintProducer::endJob() {}


std::pair<double, double>
TwoBodyDecayMomConstraintProducer::momentaAtVertex( const TwoBodyDecay& tbd ) const
{
  // construct global trajectory parameters at the starting point
  TwoBodyDecayModel tbdDecayModel( tbd[TwoBodyDecayParameters::mass], secondaryMass_ );
  std::pair< AlgebraicVector, AlgebraicVector > secondaryMomenta = tbdDecayModel.cartesianSecondaryMomenta( tbd );

  return std::make_pair( secondaryMomenta.first.norm(),
			 secondaryMomenta.second.norm() );
}


std::pair<double, double>
TwoBodyDecayMomConstraintProducer::momentaAtInnermostSurface( const TwoBodyDecay& tbd,
							      const std::vector<reco::TransientTrack>& ttracks,
							      const MagneticField* magField) const
{
  std::pair<double, double> result = std::make_pair( -1., -1. );

  /// Get the innermost trajectory states
  std::pair<bool, TrajectoryStateOnSurface> oldInnermostState1 = innermostState( ttracks[0] );
  std::pair<bool, TrajectoryStateOnSurface> oldInnermostState2 = innermostState( ttracks[1] );
  if ( !oldInnermostState1.second.isValid() || !oldInnermostState2.second.isValid() ) return result;

  /// Construct the TBD trajectory states
  TwoBodyDecayTrajectoryState::TsosContainer trackTsos( oldInnermostState1.second, oldInnermostState2.second );
  TwoBodyDecayTrajectoryState tbdTrajState( trackTsos, tbd, secondaryMass_, magField, true );
  if ( !tbdTrajState.isValid() ) return result;

  /// Match the old and the new estimates for the trajectory state
  bool match1 = match( tbdTrajState.trajectoryStates(true).first, oldInnermostState1.second );
  bool match2 = match( tbdTrajState.trajectoryStates(true).second, oldInnermostState2.second );
  if ( !match1 || !match2 ) return result;

  result = std::make_pair( fabs( 1./tbdTrajState.trajectoryStates(true).first.localParameters().qbp() ),
			   fabs( 1./tbdTrajState.trajectoryStates(true).second.localParameters().qbp() ) );
  return result;
}


std::pair<bool, TrajectoryStateOnSurface>
TwoBodyDecayMomConstraintProducer::innermostState( const reco::TransientTrack& ttrack ) const
{
  double outerR = ttrack.outermostMeasurementState().globalPosition().perp();
  double innerR = ttrack.innermostMeasurementState().globalPosition().perp();
  bool insideOut = ( outerR > innerR );
  return std::make_pair( insideOut, insideOut ? ttrack.innermostMeasurementState() : ttrack.outermostMeasurementState() );
}


bool
TwoBodyDecayMomConstraintProducer::match( const TrajectoryStateOnSurface& newTsos,
				       const TrajectoryStateOnSurface& oldTsos ) const
{
  LocalPoint lp1 = newTsos.localPosition();
  LocalPoint lp2 = oldTsos.localPosition();

  double deltaX = lp1.x() - lp2.x();
  double deltaY = lp1.y() - lp2.y();

  return ( ( fabs(deltaX) < sigmaPositionCutValue_ ) && ( fabs(deltaY) < sigmaPositionCutValue_ ) );
}


//define this as a plug-in
DEFINE_FWK_MODULE(TwoBodyDecayMomConstraintProducer);
