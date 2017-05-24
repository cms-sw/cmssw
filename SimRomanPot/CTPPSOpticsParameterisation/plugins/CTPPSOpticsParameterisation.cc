// -*- C++ -*-
//
// Package:    SimRomanPot/CTPPSOpticsParameterisation
// Class:      CTPPSOpticsParameterisation
// 
/**\class CTPPSOpticsParameterisation CTPPSOpticsParameterisation.cc SimRomanPot/CTPPSOpticsParameterisation/plugins/CTPPSOpticsParameterisation.cc

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

#include "SimDataFormats/CTPPS/interface/CTPPSSimHit.h"

class CTPPSOpticsParameterisation : public edm::stream::EDProducer<> {
  public:
    explicit CTPPSOpticsParameterisation( const edm::ParameterSet& );
    ~CTPPSOpticsParameterisation();

    static void fillDescriptions( edm::ConfigurationDescriptions& descriptions );

  private:
    virtual void beginStream( edm::StreamID ) override;
    virtual void produce( edm::Event&, const edm::EventSetup& ) override;
    virtual void endStream() override;

    //virtual void beginRun( const edm::Run&, const edm::EventSetup& ) override;
    //virtual void endRun( const edm::Run&, const edm::EventSetup& ) override;
    //virtual void beginLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& ) override;
    //virtual void endLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& ) override;

    edm::ParameterSet beamConditions_;
};

CTPPSOpticsParameterisation::CTPPSOpticsParameterisation( const edm::ParameterSet& iConfig ) :
  beamConditions_( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ) )
{
  produces< std::vector<CTPPSSimHit> >();
}


CTPPSOpticsParameterisation::~CTPPSOpticsParameterisation()
{}


// ------------ method called to produce the data  ------------
void
CTPPSOpticsParameterisation::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
/* This is an event example
  //Read 'ExampleData' from the Event
  Handle<ExampleData> pIn;
  iEvent.getByLabel("example",pIn);

*/
  std::unique_ptr< std::vector<CTPPSSimHit> > pOut( new std::vector<CTPPSSimHit> );

/* this is an EventSetup example
  //Read SetupData from the SetupRecord in the EventSetup
  ESHandle<SetupData> pSetup;
  iSetup.get<SetupRecord>().get(pSetup);
*/

  iEvent.put( std::move( pOut ) );
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
CTPPSOpticsParameterisation::beginStream( edm::StreamID )
{}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
CTPPSOpticsParameterisation::endStream()
{}

// ------------ method called when starting to processes a run  ------------
/*
void
CTPPSOpticsParameterisation::beginRun( const edm::Run&, const edm::EventSetup& )
{}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
CTPPSOpticsParameterisation::endRun( const edm::Run&, const edm::EventSetup& )
{}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
CTPPSOpticsParameterisation::beginLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& )
{}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
CTPPSOpticsParameterisation::endLuminosityBlock( const edm::LuminosityBlock&, const edm::EventSetup& )
{}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CTPPSOpticsParameterisation::fillDescriptions( edm::ConfigurationDescriptions& descriptions )
{
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault( desc );
}

// define this as a plug-in
DEFINE_FWK_MODULE( CTPPSOpticsParameterisation );
