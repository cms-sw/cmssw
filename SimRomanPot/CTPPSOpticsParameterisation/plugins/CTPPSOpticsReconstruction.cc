// -*- C++ -*-
//
// Package:    SimRomanPot/CTPPSOpticsParameterisation
// Class:      CTPPSOpticsReconstruction
// 
/**\class CTPPSOpticsReconstruction CTPPSOpticsReconstruction.cc SimRomanPot/CTPPSOpticsParameterisation/plugins/CTPPSOpticsReconstruction.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Laurent Forthomme
//         Created:  Wed, 24 May 2017 07:40:20 GMT
//
//


#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "SimDataFormats/CTPPS/interface/CTPPSSimProtonTrack.h"
#include "SimDataFormats/CTPPS/interface/LHCOpticsApproximator.h"

#include "SimRomanPot/CTPPSOpticsParameterisation/interface/ProtonReconstructionAlgorithm.h"

class CTPPSOpticsReconstruction : public edm::stream::EDProducer<> {
  public:
    explicit CTPPSOpticsReconstruction( const edm::ParameterSet& );
    ~CTPPSOpticsReconstruction();

    static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  private:
    virtual void produce( edm::Event&, const edm::EventSetup& ) override;
    void transportProtonTrack( const CTPPSSimProtonTrack&, std::vector<CTPPSLocalTrackLite>& );

    edm::EDGetTokenT< edm::View<CTPPSLocalTrackLite> > hitsToken_;

    edm::ParameterSet beamConditions_;

    bool checkApertures_;
    bool invertBeamCoordinatesSystem_;

    edm::FileInPath opticsFileBeam1_, opticsFileBeam2_;
    std::vector<edm::ParameterSet> detectorPackages_;

    std::unique_ptr<ProtonReconstructionAlgorithm> prAlgo45_;
    std::unique_ptr<ProtonReconstructionAlgorithm> prAlgo56_;
};

CTPPSOpticsReconstruction::CTPPSOpticsReconstruction( const edm::ParameterSet& iConfig ) :
  hitsToken_( consumes< edm::View<CTPPSLocalTrackLite> >( iConfig.getParameter<edm::InputTag>( "potsHitsTag" ) ) ),
  beamConditions_( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ) ),
  checkApertures_( iConfig.getParameter<bool>( "checkApertures" ) ),
  invertBeamCoordinatesSystem_( iConfig.getParameter<bool>( "invertBeamCoordinatesSystem" ) ),
  opticsFileBeam1_( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam1" ) ),
  opticsFileBeam2_( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam2" ) ),
  detectorPackages_( iConfig.getParameter< std::vector<edm::ParameterSet> >( "detectorPackages" ) )
{
  produces< std::vector<CTPPSSimProtonTrack> >( "sector45" );
  produces< std::vector<CTPPSSimProtonTrack> >( "sector56" );

  auto f_in_optics_beam1 = std::make_unique<TFile>( opticsFileBeam1_.fullPath().c_str() ),
       f_in_optics_beam2 = std::make_unique<TFile>( opticsFileBeam2_.fullPath().c_str() );

  // load optics and interpolators
  std::unordered_map<unsigned int,std::string> pots_45, pots_56;
  for ( const auto& rp : detectorPackages_ ) {
    const std::string interp_name = rp.getParameter<std::string>( "interpolatorName" );
    const unsigned int raw_detid = rp.getParameter<unsigned int>( "potId" );
    TotemRPDetId detid( raw_detid );

    if ( detid.arm()==0 ) pots_45.insert( std::make_pair( raw_detid, interp_name ) );
    if ( detid.arm()==1 ) pots_56.insert( std::make_pair( raw_detid, interp_name ) );
  }

  // reconstruction algorithms
  prAlgo45_ = std::make_unique<ProtonReconstructionAlgorithm>( beamConditions_, pots_45, opticsFileBeam2_.fullPath(), checkApertures_, invertBeamCoordinatesSystem_ );
  prAlgo56_ = std::make_unique<ProtonReconstructionAlgorithm>( beamConditions_, pots_56, opticsFileBeam1_.fullPath(), checkApertures_, invertBeamCoordinatesSystem_ );
}

CTPPSOpticsReconstruction::~CTPPSOpticsReconstruction()
{}

void
CTPPSOpticsReconstruction::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::unique_ptr< std::vector<CTPPSSimProtonTrack> > pOut45( new std::vector<CTPPSSimProtonTrack> );
  std::unique_ptr< std::vector<CTPPSSimProtonTrack> > pOut56( new std::vector<CTPPSSimProtonTrack> );

  edm::Handle< edm::View<CTPPSLocalTrackLite> > hits;
  iEvent.getByToken( hitsToken_, hits );

  // run reconstruction
  prAlgo45_->reconstruct( hits->ptrs(), *pOut45 );
  prAlgo56_->reconstruct( hits->ptrs(), *pOut56 );

  iEvent.put( std::move( pOut45 ), "sector45" );
  iEvent.put( std::move( pOut56 ), "sector56" );
}

void
CTPPSOpticsReconstruction::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc );
}

// define this as a plug-in
DEFINE_FWK_MODULE( CTPPSOpticsReconstruction );
