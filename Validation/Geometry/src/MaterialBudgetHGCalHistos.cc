#include "Validation/Geometry/interface/MaterialBudgetHGCalHistos.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

MaterialBudgetHGCalHistos::MaterialBudgetHGCalHistos( const edm::ParameterSet &p )
{
  binEta      = p.getUntrackedParameter<int>( "NBinEta", 260 );
  binPhi      = p.getUntrackedParameter<int>( "NBinPhi", 180 );
  maxEta      = p.getUntrackedParameter<double>( "MaxEta", 5.2 );
  etaLow      = p.getUntrackedParameter<double>( "EtaLow", -5.2 );
  etaHigh     = p.getUntrackedParameter<double>( "EtaHigh", 5.2 );
  m_fillHistos  = p.getUntrackedParameter<bool>( "FillHisto", true );
  m_printSum    = p.getUntrackedParameter<bool>( "PrintSummary", false );
  edm::LogInfo("MaterialBudget") << "MaterialBudgetHGCalHistos: FillHisto : "
				 << m_fillHistos << " PrintSummary " << m_printSum
				 << " == Eta plot: NX " << binEta << " Range "
				 << -maxEta << ":" << maxEta <<" Phi plot: NX "
				 << binPhi << " Range " << -pi << ":" << pi 
				 << " (Eta limit " << etaLow << ":" << etaHigh 
				 <<")";
  if( m_fillHistos) book();
}

void
MaterialBudgetHGCalHistos::fillBeginJob( const DDCompactView & cpv )
{
  if( m_fillHistos )
  {
    std::string attribute = "ReadOutName";
    std::string value     = "HGCHits";
    DDSpecificsFilter filter1;
    DDValue           ddv1( attribute, value, 0 );
    filter1.setCriteria( ddv1, DDSpecificsFilter::equals );
    DDFilteredView fv1( cpv );
    fv1.addFilter( filter1 );
    sensitives = getNames( fv1 );
    edm::LogInfo("MaterialBudget") << "MaterialBudgetHGCalHistos: Names to be "
				   << "tested for " << attribute << " = " 
				   << value << " has " << sensitives.size()
				   << " elements";
    for( unsigned int i = 0; i < sensitives.size(); i++ ) 
      edm::LogInfo("MaterialBudget") << "MaterialBudgetHGCalHistos: sensitives["
				     << i << "] = " << sensitives[i];

    attribute = "Volume";
    value     = "HGC";
    DDSpecificsFilter filter2;
    DDValue           ddv2( attribute, value, 0 );
    filter2.setCriteria( ddv2, DDSpecificsFilter::equals );
    DDFilteredView fv2( cpv );
    fv2.addFilter( filter2 );
    hgcNames = getNames( fv2 );
    fv2.firstChild();
    DDsvalues_type sv( fv2.mergedSpecifics());

    std::string hgcalRO[3] = {"HGCHitsEE", "HGCHitsHEfront", "HGCHitsHEback"};
    attribute = "ReadOutName";
    for( int k = 0; k < 3; k++ )
    {
      value = hgcalRO[k];
      DDSpecificsFilter filter3;
      DDValue           ddv3( attribute, value, 0 );
      filter3.setCriteria( ddv3, DDSpecificsFilter::equals );
      DDFilteredView fv3( cpv );
      fv3.addFilter( filter3 );
      std::vector<std::string> senstmp = getNames( fv3 );
      edm::LogInfo("MaterialBudget") << "MaterialBudgetHGCalHistos: Names to be"
				     << " tested for " << attribute << " = " 
				     << value << " has " << senstmp.size()
				     << " elements";
      for( unsigned int i = 0; i < senstmp.size(); i++ )
	sensitiveHGC.push_back( senstmp[i] );
    }
    for( unsigned int i = 0; i < sensitiveHGC.size(); i++ ) 
      edm::LogInfo("MaterialBudget") << "MaterialBudgetHGCalHistos:sensitiveHGC["
				     << i << "] = " << sensitiveHGC[i];
  }
}

void
MaterialBudgetHGCalHistos::fillStartTrack( const G4Track* aTrack )
{
  m_id     = layer  = steps   = 0;
  radLen = intLen = stepLen = 0;

  const G4ThreeVector& dir = aTrack->GetMomentum() ;
  if (dir.theta() != 0 ) {
    eta = dir.eta();
  } else {
    eta = -99;
  }
  phi = dir.phi();
  double theEnergy = aTrack->GetTotalEnergy();
  int    theID     = (int)(aTrack->GetDefinition()->GetPDGEncoding());

  if( m_printSum ) {
    matList.clear();
    stepLength.clear();
    radLength.clear();
    intLength.clear();
  }

  edm::LogInfo("MaterialBudget") << "MaterialBudgetHGCalHistos: Track " 
				 << aTrack->GetTrackID() << " Code " << theID
				 << " Energy " << theEnergy/GeV << " GeV; Eta "
				 << eta << " Phi " << phi/deg << " PT "
				 << dir.perp()/GeV << " GeV *****";
}

void
MaterialBudgetHGCalHistos::fillPerStep( const G4Step* aStep )
{
  G4Material * material = aStep->GetPreStepPoint()->GetMaterial();
  double step    = aStep->GetStepLength();
  double radl    = material->GetRadlen();
  double intl    = material->GetNuclearInterLength();
  double density = material->GetDensity() / (g/cm3);

  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  std::string         name  = touch->GetVolume(0)->GetName();
  std::string         matName = material->GetName();
  if( m_printSum )
  {
    bool found = false;
    for( unsigned int ii = 0; ii < matList.size(); ii++ )
    {
      if( matList[ii] == matName )
      {
	stepLength[ii] += step;
	radLength[ii]  += (step/radl);
	intLength[ii]  += (step/intl);
	found           = true;
	break;
      }
    }
    if( !found )
    {
      matList.push_back( matName );
      stepLength.push_back( step );
      radLength.push_back( step / radl );
      intLength.push_back( step / intl );
    }
    edm::LogInfo("MaterialBudget") << name << " " << step << " " << matName 
				   << " " << stepLen << " " << step/radl << " " 
				   << radLen << " " <<step/intl << " " <<intLen;
  }
  else
  {
    edm::LogInfo("MaterialBudget") << "MaterialBudgetHGCalHistos: Step at " 
				   << name << " Length " << step << " in " 
				   << matName << " of density " << density 
				   << " g/cc; Radiation Length " <<radl <<" mm;"
				   << " Interaction Length " << intl << " mm\n"
				   << "                          Position " 
				   << aStep->GetPreStepPoint()->GetPosition()
				   << " Cylindrical R "
				   <<aStep->GetPreStepPoint()->GetPosition().perp()
				   << " Length (so far) " << stepLen << " L/X0 "
				   << step/radl << "/" << radLen << " L/Lambda "
				   << step/intl << "/" << intLen;
  }

  if( m_fillHistos && isItHGC( name ))
  {
    m_id = 1;
    fillHisto(m_id-1);
  }

  stepLen += step;
  radLen  += step/radl;
  intLen  += step/intl;
}

void
MaterialBudgetHGCalHistos::fillEndTrack( void )
{
  if( m_fillHistos )
  {
    fillHisto( maxSet - 1 );
  }
  if( m_printSum )
  {
    for( unsigned int ii = 0; ii < matList.size(); ii++ )
    {
      edm::LogInfo("MaterialBudget") << matList[ii] << "\t" << stepLength[ii]
				     << "\t" << radLength[ii] << "\t"
				     << intLength[ii];
    }
  }
}

void
MaterialBudgetHGCalHistos::book( void )
{
  edm::Service<TFileService> tfile;
  
  if( !tfile.isAvailable() )
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";

  double maxPhi = pi;
  edm::LogInfo("MaterialBudget") << "MaterialBudgetHGCalHistos: Booking user "
				 << "histos === with " << binEta << " bins "
				 << "in eta from " << -maxEta << " to "
				 << maxEta << " and " << binPhi << " bins "
				 << "in phi from " << -maxPhi << " to " 
				 << maxPhi;

  char  name[10], title[40];
  // total X0
  for( int i = 0; i < maxSet; i++ )
  {
    sprintf(name, "%d", i+100);
    sprintf(title, "MB(X0) prof Eta in region %d", i);
    me100[i] =  tfile->make<TProfile>(name, title, binEta, -maxEta, maxEta);
    sprintf(name, "%d", i+200);
    sprintf(title, "MB(L0) prof Eta in region %d", i);
    me200[i] = tfile->make<TProfile>(name, title, binEta, -maxEta, maxEta);
    sprintf(name, "%d", i+300);
    sprintf(title, "MB(Step) prof Eta in region %d", i);
    me300[i] = tfile->make<TProfile>(name, title, binEta, -maxEta, maxEta);
    sprintf(name, "%d", i+400);
    sprintf(title, "Eta in region %d", i);
    me400[i] = tfile->make<TH1F>(name, title, binEta, -maxEta, maxEta);
    sprintf(name, "%d", i+500);
    sprintf(title, "MB(X0) prof Ph in region %d", i);
    me500[i] = tfile->make<TProfile>(name, title, binPhi, -maxPhi, maxPhi);
    sprintf(name, "%d", i+600);
    sprintf(title, "MB(L0) prof Ph in region %d", i);
    me600[i] = tfile->make<TProfile>(name, title, binPhi, -maxPhi, maxPhi);
    sprintf(name, "%d", i+700);
    sprintf(title, "MB(Step) prof Ph in region %d", i);
    me700[i] = tfile->make<TProfile>(name, title, binPhi, -maxPhi, maxPhi);
    sprintf(name, "%d", i+800);
    sprintf(title, "Phi in region %d", i);
    me800[i] = tfile->make<TH1F>(name, title, binPhi, -maxPhi, maxPhi);
    sprintf(name, "%d", i+900);
    sprintf(title, "MB(X0) prof Eta Phi in region %d", i);
    me900[i] = tfile->make<TProfile2D>(name, title, binEta/2, -maxEta, maxEta,
				       binPhi/2, -maxPhi, maxPhi);
    sprintf(name, "%d", i+1000);
    sprintf(title, "MB(L0) prof Eta Phi in region %d", i);
    me1000[i]= tfile->make<TProfile2D>(name, title, binEta/2, -maxEta, maxEta,
				       binPhi/2, -maxPhi, maxPhi);
    sprintf(name, "%d", i+1100);
    sprintf(title, "MB(Step) prof Eta Phi in region %d", i);
    me1100[i]= tfile->make<TProfile2D>(name, title, binEta/2, -maxEta, maxEta,
				       binPhi/2, -maxPhi, maxPhi);
    sprintf(name, "%d", i+1200);
    sprintf(title, "Eta vs Phi in region %d", i);
    me1200[i]= tfile->make<TH2F>(name, title, binEta/2, -maxEta, maxEta, 
				 binPhi/2, -maxPhi, maxPhi);
  }
  edm::LogInfo("MaterialBudget") << "MaterialBudgetHGCalHistos: Booking user "
				 << "histos done ===";
}

void
MaterialBudgetHGCalHistos::fillHisto( int ii )
{
  LogDebug("MaterialBudget") << "MaterialBudgetHGCalHistos:FillHisto called "
			     << "with index " << ii << " integrated  step "
			     << stepLen << " X0 " << radLen << " Lamda " 
			     << intLen;
  
  if( ii >=0 && ii < maxSet )
  {
    me100[ii]->Fill( eta, radLen );
    me200[ii]->Fill( eta, intLen );
    me300[ii]->Fill( eta, stepLen );
    me400[ii]->Fill( eta );

    if( eta >= etaLow && eta <= etaHigh )
    {
      me500[ii]->Fill( phi, radLen );
      me600[ii]->Fill( phi, intLen );
      me700[ii]->Fill( phi, stepLen );
      me800[ii]->Fill( phi );
    }

    me900[ii]->Fill( eta, phi, radLen );
    me1000[ii]->Fill( eta, phi, intLen );
    me1100[ii]->Fill( eta, phi, stepLen );
    me1200[ii]->Fill( eta, phi );
  }
}

std::vector<std::string>
MaterialBudgetHGCalHistos::getNames( DDFilteredView& fv )
{
  std::vector<std::string> tmp;
  bool dodet = fv.firstChild();
  while (dodet) {
    const DDLogicalPart & log = fv.logicalPart();
    std::string namx = log.name().name();
    bool ok = true;
    for (unsigned int i=0; i<tmp.size(); i++)
      if (namx == tmp[i]) ok = false;
    if (ok) tmp.push_back(namx);
    dodet = fv.next();
  }
  return tmp;
}

bool
MaterialBudgetHGCalHistos::isItHGC( const std::string & name )
{
  std::vector<std::string>::const_iterator it = sensitiveHGC.begin();
  std::vector<std::string>::const_iterator itEnd = sensitiveHGC.end();
  for( ; it != itEnd; ++it )
    if( name == *it ) return true;
  return false;
}
