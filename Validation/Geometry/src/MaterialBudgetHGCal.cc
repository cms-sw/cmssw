#include "Validation/Geometry/interface/MaterialBudgetHGCal.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "G4Step.hh"
#include "G4Track.hh"

#include <iostream>

MaterialBudgetHGCal::MaterialBudgetHGCal( const edm::ParameterSet& p )
  : m_HistoHGCal(0)
{
  edm::ParameterSet pSet = p.getParameter<edm::ParameterSet>( "MaterialBudgetHGCal" );
  m_rMax = pSet.getUntrackedParameter<double>( "RMax", 4.5 )*m;
  m_zMax = pSet.getUntrackedParameter<double>( "ZMax", 13.0 )*m;
  edm::LogInfo( "MaterialBudget" ) << "MaterialBudgetHGCal initialized with rMax "
				   << m_rMax << " mm and zMax " << m_zMax << " mm";
  m_HistoHGCal = new MaterialBudgetHGCalHistos( pSet );
}

MaterialBudgetHGCal::~MaterialBudgetHGCal( void )
{
  if( m_HistoHGCal) delete m_HistoHGCal;
}

void
MaterialBudgetHGCal::update( const BeginOfJob* job )
{
  edm::ESTransientHandle<DDCompactView> pDD;
  (*job)()->get<IdealGeometryRecord>().get(pDD);
  if( m_HistoHGCal ) m_HistoHGCal->fillBeginJob((*pDD));
}

void
MaterialBudgetHGCal::update( const BeginOfTrack* trk )
{
  const G4Track * aTrack = ( *trk )(); // recover G4 pointer if wanted
  if( m_HistoHGCal ) m_HistoHGCal->fillStartTrack( aTrack );
}
 
void
MaterialBudgetHGCal::update( const G4Step* aStep )
{
  if( m_HistoHGCal ) m_HistoHGCal->fillPerStep( aStep );

  if( stopAfter( aStep ))
  {
    G4Track* track = aStep->GetTrack();
    track->SetTrackStatus( fStopAndKill );
  }
}

void
MaterialBudgetHGCal::update( const EndOfTrack* )
{
  if( m_HistoHGCal ) m_HistoHGCal->fillEndTrack();
}

bool
MaterialBudgetHGCal::stopAfter( const G4Step* aStep )
{
  G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
  double        rr = hitPoint.perp();
  double        zz = std::abs( hitPoint.z());

  if( rr > m_rMax || zz > m_zMax )
  {
    LogDebug("MaterialBudget") << " MaterialBudgetHGCal::StopAfter R = " << rr
			       << " and Z = " << zz;
    return true;
  }
  else
  {
    return false;
  }
}
