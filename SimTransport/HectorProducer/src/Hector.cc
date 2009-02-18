#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
//Hector
#include "SimTransport/HectorProducer/interface/Hector.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "HepMC/SimpleVector.h"


#include <math.h>


//#include <iostream>
Hector::Hector(const edm::ParameterSet & param, bool verbosity, bool FP420Transport,bool ZDCTransport) : 
  m_verbosity(verbosity), 
  m_FP420Transport(FP420Transport),
  m_ZDCTransport(ZDCTransport)
{
  
  edm::LogInfo ("Hector") << "===================================================================\n"  
                          << "=== Hector Constructor start                       =====\n";
  
  // Create LHC beam line
  edm::ParameterSet hector_par = param.getParameter<edm::ParameterSet>("Hector");
  
  // User definitons
  lengthfp420    = hector_par.getParameter<double>("BeamLineLengthFP420" );
  m_rpp420_f     = (float) hector_par.getParameter<double>("RP420f");
  m_rpp420_b     = (float) hector_par.getParameter<double>("RP420b");
  lengthzdc      = hector_par.getParameter<double>("BeamLineLengthZDC" );
  lengthd1       = hector_par.getParameter<double>("BeamLineLengthD1" );
  beam1filename  = hector_par.getParameter<string>("Beam1");
  beam2filename  = hector_par.getParameter<string>("Beam2");  
  m_rppzdc       = (float) lengthzdc ;
  m_rppd1        = (float) lengthd1 ;
  m_smearAng     = hector_par.getParameter<bool>("smearAng");
  m_sigmaSTX     = hector_par.getParameter<double>("sigmaSTX" );
  m_sigmaSTY     = hector_par.getParameter<double>("sigmaSTY" );
  m_smearE       = hector_par.getParameter<bool>("smearEnergy");
  m_sig_e        = hector_par.getParameter<double>("sigmaEnergy");
  etacut         = hector_par.getParameter<double>("EtaCutForHector" );
  
  edm::LogInfo ("Hector") << "Hector parameters: \n" 
			  << "   lengthfp420:    " << lengthfp420 << "\n"
			  << "   m_rpp420_f:    " << m_rpp420_f << "\n"
			  << "   m_rpp420_b:    " << m_rpp420_b << "\n"
			  << "   lengthzdc:    " << lengthzdc << "\n"
			  << "   lengthd1:    " << lengthd1 << "\n";
  
  if(m_verbosity) {
    cout << "===================================================================" << endl;  
    cout << " * * * * * * * * * * * * * * * * * * * * * * * * * * * *           " << endl;
    cout << " *                                                         *       " << endl;
    cout << " *                   --<--<--  A fast simulator --<--<--     *     " << endl;
    cout << " *                 | --<--<--     of particle   --<--<--     *     " << endl;
    cout << " *  ----HECTOR----<                                          *     " << endl;
    cout << " *                 | -->-->-- transport through-->-->--      *     " << endl; 
    cout << " *                   -->-->-- generic beamlines -->-->--     *     " << endl;
    cout << " *                                                           *     " << endl; 
    cout << " * JINST 2:P09005 (2007)                                     *     " << endl;
    cout << " *      X Rouby, J de Favereau, K Piotrzkowski (CP3)         *     " << endl;
    cout << " *       http://www.fynu.ucl.ac.be/hector.html               *     " << endl;
    cout << " *                                                           *     " << endl;
    cout << " * Center for Cosmology, Particle Physics and Phenomenology  *     " << endl;
    cout << " *              Universite catholique de Louvain             *     " << endl;
    cout << " *                 Louvain-la-Neuve, Belgium                 *     " << endl;
    cout << " *                                                         *       " << endl;
    cout << " * * * * * * * * * * * * * * * * * * * * * * * * * * * *           " << endl; 
    cout << "=== Hector:    m_FP420Transport=" << m_FP420Transport <<"m_ZDCTransport " << m_ZDCTransport << endl;
    cout << "=== lengthfp420: " << lengthfp420 << endl;
    cout << "=== m_rpp420_f: " << m_rpp420_f << endl;
    cout << "=== m_rpp420_b: " << m_rpp420_b << endl;
    cout << "=== lengthzdc: " << lengthzdc << endl;
    cout << "=== lengthd1: " << lengthd1 << endl;
    cout << "===================================================================" << endl;
  }  
  edm::FileInPath b1(beam1filename.c_str());
  edm::FileInPath b2(beam2filename.c_str());
  
  // construct beam line for FP420:                                                                                           .
  if(m_FP420Transport && lengthfp420>0. ) {
    m_beamlineFP4201 = new H_BeamLine(  1, lengthfp420 + 0.1 ); // (direction, length)
    m_beamlineFP4202 = new H_BeamLine( -1, lengthfp420 + 0.1 ); //
    try {
      m_beamlineFP4201->fill( b1.fullPath(),  1, "IP5" );
      m_beamlineFP4202->fill( b2.fullPath(), -1, "IP5" );
    } catch ( const edm::Exception& e ) {
      std::string msg = e.what();
      msg += " caught in Hector... \nERROR: Could not locate SimTransport/HectorData data files.";
      edm::LogError ("DataNotFound") << msg;
    }
    m_beamlineFP4201->offsetElements( 120, -0.097 );
    m_beamlineFP4202->offsetElements( 120, +0.097 );
    m_beamlineFP4201->calcMatrix();
    m_beamlineFP4202->calcMatrix();
  }  
  else{
    cout << "=== Hector: WARNING: lengthfp420=  " << lengthfp420 << endl;
  }
  
  
  if (m_ZDCTransport && lengthzdc>0. && lengthd1>0.) {
    // construct beam line for ZDC:                                                                                           .
    m_beamlineZDC1 = new H_BeamLine(  1, lengthzdc + 0.1 ); // (direction, length)
    m_beamlineZDC2 = new H_BeamLine( -1, lengthzdc + 0.1 ); //
    try {
      m_beamlineZDC1->fill( b1.fullPath(),  1, "IP5" );
      m_beamlineZDC2->fill( b2.fullPath(), -1, "IP5" );
    } catch ( const edm::Exception& e ) {
      std::string msg = e.what();
      msg += " caught in Hector... \nERROR: Could not locate SimTransport/HectorData data files.";
      edm::LogError ("DataNotFound") << msg;
    }
    m_beamlineZDC1->offsetElements( 120, -0.097 );
    m_beamlineZDC2->offsetElements( 120, +0.097 );
    m_beamlineZDC1->calcMatrix();
    m_beamlineZDC2->calcMatrix();
    
    
    // construct beam line for D1:                                                                                           .
    m_beamlineD11 = new H_BeamLine(  1, lengthd1 + 0.1 ); // (direction, length)
    m_beamlineD12 = new H_BeamLine( -1, lengthd1 + 0.1 ); //
    try {
      m_beamlineD11->fill( b1.fullPath(),  1, "IP5" );
      m_beamlineD12->fill( b2.fullPath(), -1, "IP5" );
    } catch ( const edm::Exception& e ) {
      std::string msg = e.what();
      msg += " caught in Hector... \nERROR: Could not locate SimTransport/HectorData data files.";
      edm::LogError ("DataNotFound") << msg;
    }
    m_beamlineD11->offsetElements( 120, -0.097 );
    m_beamlineD12->offsetElements( 120, +0.097 );
    m_beamlineD11->calcMatrix();
    m_beamlineD12->calcMatrix();
  }  
  else{
    cout << "=== Hector: WARNING: lengthzdc=  " << lengthzdc << "lengthd1=  " << lengthd1 << endl;
  }
  
  edm::LogInfo ("Hector") << "===================================================================\n";
  
}

Hector::~Hector(){
  
  edm::LogInfo ("Hector") << "===================================================================\n"  
                          << "=== Start delete Hector                                       =====\n";
  for (std::map<unsigned int,H_BeamParticle*>::iterator it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
    delete (*it).second;
  }
  //
  delete m_beamlineFP4201;
  delete m_beamlineFP4202;
  delete m_beamlineZDC1;
  delete m_beamlineZDC2;
  delete m_beamlineD11;
  delete m_beamlineD12;
  
  edm::LogInfo ("Hector") << "===================================================================\n";  
}

void Hector::clearApertureFlags(){
  m_isStoppedfp420.clear();
  m_isStoppedzdc.clear();
  m_isStoppedd1.clear();
}

void Hector::clear(){
  for ( std::map<unsigned int,H_BeamParticle*>::iterator it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
    delete (*it).second;
  };
  m_beamPart.clear();
  m_direct.clear();
  m_eta.clear();
  m_pdg.clear();
  m_pz.clear();
  m_isCharged.clear();  
}

/*
  bool Hector::isCharged(const HepMC::GenParticle * p){
  const ParticleData * part = pdt->particle( p->pdg_id() );
  if (part){
  return part->charge()!=0;
  }else{
  // the new/improved particle table doesn't know anti-particles
  return  pdt->particle( -p->pdg_id() )!=0;
  }
  }
*/
void Hector::add( const HepMC::GenEvent * evt ,const edm::EventSetup & iSetup) {
  
  H_BeamParticle * h_p;
  unsigned int line;
  
  //  unsigned int npart = ev->nStable();
  
  /*  for (HepMC::GenEvent::particle_const_iterator eventParticle =evt->particles_begin();
      eventParticle != evt->particles_end();
      eventParticle = find_if(++eventParticle, evt->particles_end(), ( (*eventParticle)->status() == 1 ) ) ) {*/
  //	for ( HepMC::GenVertex::particle_iterator  eventParticle = (*vitr)->particles_begin(HepMC::descendants);
  for (HepMC::GenEvent::particle_const_iterator eventParticle =evt->particles_begin();
       eventParticle != evt->particles_end();
       ++eventParticle ) {
    //    if ( (*eventParticle)->status() == 1 && !(*eventParticle)->end_vertex()) {
    if ( (*eventParticle)->status() == 1 ) {
      //      if ( abs( (*eventParticle)->pdg_id() ) == 2212 ) { // if it's proton
      if ( abs( (*eventParticle)->momentum().eta())>etacut){
	line = (*eventParticle)->barcode();
	if ( m_beamPart.find(line) == m_beamPart.end() ) {
	  double charge=1.;
	  m_isCharged[line] = false;// neutrals
	  HepMC::GenParticle * g = (*eventParticle);	
	  iSetup.getData( pdt );
	  const ParticleData * part = pdt->particle( g->pdg_id() );
	  if (part){
	    charge = part->charge();
	    if(m_verbosity) cout <<"charge=  " << charge << endl;
	  }
	  if(charge !=0) m_isCharged[line] = true;//charged
	  double mass = (*eventParticle)->generatedMass();
	  if(m_verbosity) cout << "=== Hector:add:    status=  "<< g->status() <<"mass =  " << mass <<"charge=  " << charge << endl;

	  //	  h_p = new H_BeamParticle();

	  //	  H_BeamParticle myparticle(mass,charge);
	  h_p = new H_BeamParticle(mass,charge);

	  double px,py,pz;
	  //	  double pptot  ;
	  //	  double pt;
	  px = (*eventParticle)->momentum().px();	  
	  py = (*eventParticle)->momentum().py();	  
	  pz = (*eventParticle)->momentum().pz();	  
	  //	  pptot = (*eventParticle)->momentum().mag();	  
	  
	  h_p->set4Momentum( px, py, pz, (*eventParticle)->momentum().e() );
	  //  h_p->set4Momentum( px, py, pz, pptot );// but energy must be used
	  
	  //  pt = sqrt( (px*px) + (py*py) );
	  //  pt = (*eventParticle)->momentum().pt();	  
	  //  pt = (*eventParticle)->momentum().perp();	  
	  

	  //	    double XforPosition = 0.0, YforPosition = 0.0, ZforPosition = 0.0, TXforPosition=0.0, TYforPosition=0.0;
	  //old//	  double TXforPosition=1000000.*std::atan2( px,pptot );//urad
	  //old//	  double TYforPosition=1000000.*std::atan2( py,pptot );//urad
	  // from mm to um	  
	  double XforPosition = (*eventParticle)->production_vertex()->position().x()*1000.;//um
	  double YforPosition = (*eventParticle)->production_vertex()->position().y()*1000.;//um
	  double ZforPosition = (*eventParticle)->production_vertex()->position().z()*0.001;//m
	  // crossing angle (beam tilt) is not known a priory; keep now 0.0 but, in principle, can be entered as parameters
	  double TXforPosition=0.0, TYforPosition=0.0;//urad
	  
	  // Clears H_BeamParticle::positions and sets the initial one
	  h_p->setPosition(XforPosition, YforPosition, TXforPosition, TYforPosition, ZforPosition );
	  
 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	  m_beamPart[line] = h_p;
	  m_direct[line] = 0;
	  m_direct[line] = ( pz > 0 ) ? 1 : -1;
	  
	  m_eta[line] = (*eventParticle)->momentum().eta();
	  m_pdg[line] = (*eventParticle)->pdg_id();
	  m_pz[line]  = (*eventParticle)->momentum().pz();
	  
	  
	  if(m_verbosity) {
	    cout << "=== Hector:add:            pz=  "<< pz << endl;
	    //	    cout << "=== Hector:add:            pt=  "<<pt  << endl;
	    cout << "=== Hector:add:   m_isCharged[line]=  "<< m_isCharged[line]  << endl;
	  } 
	}// if find line
      }// if 2212 or eta 8.2
    }// if status
  }// for loop
  
}




void Hector::filterFP420(){
  unsigned int line;
  H_BeamParticle * part;
  std::map< unsigned int, H_BeamParticle* >::iterator it;
  
  bool is_stop;
  int direction;
  
  float x1_420;
  float y1_420;
  
  if ( m_beamPart.size() && lengthfp420>0. ) {
    
    for (it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
      line = (*it).first;
      part = (*it).second;
      
      if(m_verbosity) {
	cout << "filterFP420:pdg_id=" << (*m_pdg.find( line )).second << " momentum().eta=" << (*m_eta.find( line )).second<< " pz=" << (*m_pz.find( line )).second<< endl;
      }
      if ( (*m_isCharged.find( line )).second ) {
	direction = (*m_direct.find( line )).second;
	if (m_smearAng) {
	  // the beam transverse direction is centered on (TXforPosition, TYforPosition) at IP
	  if ( m_sigmaSTX>0. && m_sigmaSTY>0.) {
	    part->smearAng(m_sigmaSTX,m_sigmaSTY);
	  }
	  else {
	    // for smearAng() in urad, default are (STX=30.23, STY=30.23)
	    part->smearAng(); 
	  }
	}
	if (m_smearE) {
	  if ( m_sig_e ) {
	    part->smearE(m_sig_e);
	  }
	  else {
	    part->smearE();  // in GeV, default is SBE=0.79
	  }
	}
	if ( direction == 1 ) {
	  part->computePath( m_beamlineFP4201, 1 );
	  is_stop = part->stopped( m_beamlineFP4201 );
	  if(m_verbosity) cout << "=== Hector:filterFP420:   pos          is_stop=  "<< is_stop << endl;
	}
	else if ( direction == -1 ){
	  part->computePath( m_beamlineFP4202, -1 );
	  is_stop = part->stopped( m_beamlineFP4202 );
	  if(m_verbosity) cout << "=== Hector:filterFP420:   neg          is_stop=  "<< is_stop << endl;
	}
	else {
	  is_stop = true;
	  if(m_verbosity) cout << "=== Hector:filterFP420:   0          is_stop=  "<< is_stop << endl;
	}
	
	//propagating
	m_isStoppedfp420[line] = is_stop;
	if(m_verbosity) cout << "=== Hector:filterFP420:  isStopped=" << (*m_isStoppedfp420.find(line)).second <<  endl;
	
	if (!is_stop) {
	  if ( direction == 1 ) part->propagate( m_rpp420_f );
	  if ( direction == -1 ) part->propagate( m_rpp420_b );
	  x1_420 = part->getX();
	  y1_420 = part->getY();
	  if(m_verbosity) cout << "=== Hector:filterFP420: x=  "<< x1_420 <<" y= " << y1_420 << endl;
	  
	  m_xAtTrPoint[line]  = x1_420;
	  m_yAtTrPoint[line]  = y1_420;
	  m_TxAtTrPoint[line] = part->getTX();
	  m_TyAtTrPoint[line] = part->getTY();
	  m_eAtTrPoint[line]  = part->getE();
	  
	}
      }// if isCharged
      else {
	m_isStoppedfp420[line] = true;// imply that neutral particles stopped to reach 420m
	if(m_verbosity) cout << "=== Hector:filterFP420:  isStopped=" << (*m_isStoppedfp420.find(line)).second <<  endl;
      }
      
    } // for (it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) 
  } // if ( m_beamPart.size() )
  
}

void Hector::filterZDC(){
  unsigned int line;
  H_BeamParticle * part;
  std::map< unsigned int, H_BeamParticle* >::iterator it;
  
  bool is_stop_zdc;
  int direction;
  
  if ( m_beamPart.size() && lengthzdc>0. ) {
    
    for (it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
      line = (*it).first;
      part = (*it).second;
      if(m_verbosity) cout << "=== Hector:filterZDC:  isStopped=" << (*m_isStoppedfp420.find(line)).second <<  endl;
      if ( ((*m_isStoppedfp420.find(line)).second) && ((*m_isCharged.find(line)).second) ) {
	
        if(m_verbosity) cout << "filterZDC:pdg_id=" << (*m_pdg.find( line )).second << " momentum().eta=" << (*m_eta.find( line )).second<< " pz=" << (*m_pz.find( line )).second<< endl;
	
	direction = (*m_direct.find( line )).second;
	if(m_verbosity) cout << "filterZDC:direction=" << direction << endl;
	if (m_smearAng) {
	  if ( m_sigmaSTX>0. && m_sigmaSTY>0.) {
	    // the beam transverse direction is centered on (TXforPosition, TYforPosition) at IP
	    part->smearAng(m_sigmaSTX,m_sigmaSTY);
	  }
	  else {
	    // for smearAng() in urad, default are (STX=30.23, STY=30.23)
	    part->smearAng(); 
	  }
	}
	if (m_smearE) {
	  if ( m_sig_e ) {
	    part->smearE(m_sig_e);
	  }
	  else {
	    part->smearE();  // in GeV, default is SBE=0.79
	  }
	}
	if ( direction == 1 ) {
	  part->computePath( m_beamlineZDC1, 1 );
	  is_stop_zdc = part->stopped( m_beamlineZDC1 );
  	  m_isStoppedzdc[line] = is_stop_zdc;
	  if(m_verbosity) cout << "=== Hector:filterZDC:     pos        is_stop_zdc=  "<< is_stop_zdc << endl;
	}
	else if ( direction == -1 ){
	  part->computePath( m_beamlineZDC2, -1 );
	  is_stop_zdc = part->stopped( m_beamlineZDC2 );
	  m_isStoppedzdc[line] = is_stop_zdc;
	  if(m_verbosity) cout << "=== Hector:filterZDC:    neg         is_stop_zdc=  "<< is_stop_zdc << endl;
	}
	else {
	  m_isStoppedzdc[line] = true;
	  if(m_verbosity) cout << "=== Hector:filterZDC:    0         is_stop_zdc=  "<< endl;
	}
      }// if stopfp420 charged particles
      else if ( ((*m_isCharged.find(line)).second) ){
	m_isStoppedzdc[line] = false;// not stopped in propagating to FP420 and therefore in  propagation to ZDC too.
	if(m_verbosity) cout << "=== Hector:filterZDC:  isStopped=" << (*m_isStoppedzdc.find(line)).second <<  endl;
      }
      else {
	m_isStoppedzdc[line] = true;// neutrals particles considered as stopped in propagating to ZDC
	if(m_verbosity) cout << "=== Hector:filterZDC:  isStopped=" << (*m_isStoppedzdc.find(line)).second <<  endl;
      }
      
    } // for (it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) 
  } // if ( m_beamPart.size() )
  
}
void Hector::filterD1(){
  unsigned int line;
  H_BeamParticle * part;
  std::map< unsigned int, H_BeamParticle* >::iterator it;
  
  bool is_stop_d1;
  int direction;
  
  float x1_d1;
  float y1_d1;
  
  if ( m_beamPart.size() && lengthd1>0.) {
    
    for (it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
      line = (*it).first;
      part = (*it).second;
      if(m_verbosity) cout << "=== Hector:filterD1:  isStopped=" << (*m_isStoppedzdc.find(line)).second <<  endl;
      if ( ((*m_isStoppedzdc.find(line)).second) || !((*m_isCharged.find( line )).second) ) {
	
        if(m_verbosity) cout << "filterD1:pdg_id=" << (*m_pdg.find( line )).second << " momentum().eta=" << (*m_eta.find( line )).second<< " pz=" << (*m_pz.find( line )).second<< endl;
	
	direction = (*m_direct.find( line )).second;
	if(m_verbosity) cout << "filterD1:direction=" << direction << endl;
	if (m_smearAng) {
	  if ( m_sigmaSTX>0. && m_sigmaSTY>0.) {
	    // the beam transverse direction is centered on (TXforPosition, TYforPosition) at IP
	    part->smearAng(m_sigmaSTX,m_sigmaSTY);
	  }
	  else {
	    // for smearAng() in urad, default are (STX=30.23, STY=30.23)
	    part->smearAng(); 
	  }
	}
	if (m_smearE) {
	  if ( m_sig_e ) {
	    part->smearE(m_sig_e);
	  }
	  else {
	    part->smearE();  // in GeV, default is SBE=0.79
	  }
	}
	if ( direction == 1 ) {
	  part->computePath( m_beamlineD11, 1 );
	  is_stop_d1 = part->stopped( m_beamlineD11 );
	  m_isStoppedd1[line] = is_stop_d1;
	  if(m_verbosity) cout << "=== Hector:filterD1:    pos         is_stop_d1=  "<< is_stop_d1 << endl;
	}
	else  if ( direction == -1 ){
	  part->computePath( m_beamlineD12, -1 );
	  is_stop_d1 = part->stopped( m_beamlineD12 );
	  m_isStoppedd1[line] = is_stop_d1;
	  if(m_verbosity) cout << "=== Hector:filterD1:    neg         is_stop_d1=  "<< is_stop_d1 << endl;
	}
	else {
	  is_stop_d1 = true;
	  m_isStoppedd1[line] = is_stop_d1;
	  if(m_verbosity) cout << "=== Hector:filterD1:    0         is_stop_d1=  "<< endl;
	}
	//propagating
	if (!is_stop_d1 ) {
	  part->propagate( lengthd1 );
	  x1_d1 = part->getX();
	  y1_d1 = part->getY();
	  m_xAtTrPoint[line]  = x1_d1;
	  m_yAtTrPoint[line]  = y1_d1;
	  m_TxAtTrPoint[line] = part->getTX();
	  m_TyAtTrPoint[line] = part->getTY();
	  m_eAtTrPoint[line]  = part->getE();
	}
      }// if stopzdc
      else {
	m_isStoppedd1[line] = false;// not stopped in propagating to ZDC and therefore in  propagation to D1 too.
	if(m_verbosity) cout << "=== Hector:filterD1:  isStopped=" << (*m_isStoppedd1.find(line)).second <<  endl;
      }
      
    } // for (it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) 
  } // if ( m_beamPart.size() )
  
}

int  Hector::getDirect( unsigned int part_n ) const {
  std::map<unsigned int, int>::const_iterator it = m_direct.find( part_n );
  if ( it != m_direct.end() ){
    return (*it).second;
  }
  return 0;
}

void Hector::print() const {
  for (std::map<unsigned int,H_BeamParticle*>::const_iterator it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
    (*it).second->printProperties();
  };
}


HepMC::GenEvent * Hector::addPartToHepMC( HepMC::GenEvent * evt ){
  
  unsigned int line;
  //  H_BeamParticle * part;
  HepMC::GenParticle * gpart;
  long double tx,ty,theta,fi,energy,time = 0;
  std::map< unsigned int, H_BeamParticle* >::iterator it;
  
  
  for (it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
    line = (*it).first;
    if(!m_FP420Transport) m_isStoppedfp420[line] = true;
    if(!m_ZDCTransport) {m_isStoppedzdc[line] = false;m_isStoppedd1[line] = true;}
    if(m_verbosity) {
      cout << "=== Hector:addPartToHepMC: isStoppedfp420=" << (*m_isStoppedfp420.find(line)).second <<  endl;
      cout << "=== Hector:addPartToHepMC: isStoppedzdc=" << (*m_isStoppedzdc.find(line)).second <<  endl;
      cout << "=== Hector:addPartToHepMC: isStoppedd1=" << (*m_isStoppedd1.find(line)).second <<  endl;
    }
    if (!((*m_isStoppedfp420.find(line)).second) || (!((*m_isStoppedd1.find(line)).second) && ((*m_isStoppedzdc.find(line)).second))){
      //  if ( !((*m_isStoppedfp420.find(line)).second) || !((*m_isStoppedd1.find(line)).second)  ) {
      gpart = evt->barcode_to_particle( line );
      if ( gpart ) {
	tx     = (*m_TxAtTrPoint.find(line)).second / 1000000.;
	ty     = (*m_TyAtTrPoint.find(line)).second / 1000000.;
	theta  = sqrt((tx*tx) + (ty*ty));
	double ddd = 0.;
	if ( !((*m_isStoppedfp420.find(line)).second)) {
	  if( (*m_direct.find( line )).second >0 ) {
	    ddd = m_rpp420_f;
	  }
	  else if((*m_direct.find( line )).second <0 ) {
	    ddd = m_rpp420_b;
	    theta= pi-theta;
	  }
	}
	else {
	  ddd = lengthd1;
	  if((*m_direct.find( line )).second <0 ) theta= pi-theta;
	}
	
	fi     = std::atan2(tx,ty); // tx, ty never == 0?
	energy = (*m_eAtTrPoint.find(line)).second;
	
	//	HepMC::GenEvent::vertex_iterator v_it;
	//	time   = ( *evt->vertices_begin() )->position().t(); // does time important?
	//long double time_buf = 0;
	//for (v_it = evt->vertices_begin(); v_it != evt->vertices_end(); ++v_it)  { // since no operator--
	//	  time_buf = ( *v_it )->position().t();
	//  time = (  time > time_buf ) ? time : time_buf;
	//	}

	//	time = ( ddd*1000. - gpart->production_vertex()->position().z()*mm )/c_light;

	time = ( ddd*1000. - gpart->production_vertex()->position().z()*mm ); // mm
	//                               decaylength/(p.Beta()*p.Gamma()*c_light);
	//	time = ( ddd*1000. - gpart->production_vertex()->position().z()*mm )/300.; // nsec

	if(ddd != 0.) {
	  if(m_verbosity) {
	    std::cout<<"=========Hector:: x= "<< (*(m_xAtTrPoint.find(line))).second*0.001<<std::endl;
	    std::cout<<"=========Hector:: y= "<< (*(m_yAtTrPoint.find(line))).second*0.001<<std::endl;
	    std::cout<<"=========Hector:: z= "<< ddd * (*(m_direct.find( line ))).second*1000.<<std::endl;
	    std::cout<<"=========Hector:: t= "<< time  <<std::endl;
	  }
	  
	  
	  
	  HepMC::GenVertex * vert = new HepMC::GenVertex( HepMC::FourVector( (*(m_xAtTrPoint.find(line))).second*0.001,
									     (*(m_yAtTrPoint.find(line))).second*0.001,
									     ddd * (*(m_direct.find( line ))).second*1000.,
									     time + .001*time ) );
	  
	  
	  
          gpart->set_status( 2 );
          vert->add_particle_in( gpart );
          vert->add_particle_out( new HepMC::GenParticle( HepMC::FourVector(energy*std::sin(theta)*std::sin(fi),
                                                                            energy*std::sin(theta)*std::cos(fi),
                                                                            energy*std::cos(theta),
                                                                            energy ),
                                                          gpart->pdg_id(), 1, gpart->flow() ) );
          evt->add_vertex( vert );
	  
	if(m_verbosity) std::cout << "Hector::TRANSPORTED pz= "<< gpart->momentum().pz()  << " eta= "<< gpart->momentum().eta()  << " status= "<< gpart->status()  <<std::endl;
	  
	  
	}// ddd
      }// if gpart
    }// if !isStopped
    
    else {
      gpart = evt->barcode_to_particle( line );
      if ( gpart ) {
	HepMC::GenVertex * vert= new HepMC::GenVertex();
	gpart->set_status( 2 );
	vert->add_particle_in( gpart );
	vert->add_particle_out( gpart );
	evt->add_vertex( vert );
	if(m_verbosity) std::cout << "Hector::NON-transp. pz= "<< gpart->momentum().pz()  << " eta= "<< gpart->momentum().eta()  << " status= "<< gpart->status()  <<std::endl;
      }
    }

  }//for 
  //  cout << "=== Hector:addPartToHepMC: end " << endl;
  
  //  if(m_verbosity){
  //   std::cout<<"============================================================================"<<std::endl;
  //  std::cout<<"=========Hector::addPartToHepMC print:"<<std::endl;
  //  evt->print();
  //}
  
  
  
  return evt;
} 
