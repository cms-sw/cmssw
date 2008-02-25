#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
//Hector
#include "SimTransport/HectorProducer/interface/Hector.h"
#include "SimTransport/HectorProducer/interface/HectorGenParticle.h"

#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "HepMC/SimpleVector.h"



#include <math.h>

//#include <iostream>

Hector::Hector(const edm::ParameterSet & param) : 
  m_IPx(0),m_IPy(0),m_IPz(0),m_IPt(0)
{
  
  edm::LogInfo ("Hector") << "===================================================================\n"  
                          << "=== Hector Constructor start                       =====\n";
  
  // Create LHC beam line
  edm::ParameterSet hector_par = param.getParameter<edm::ParameterSet>("Hector");
  
  // User definitons
  m_verbosity    = hector_par.getParameter<bool>("Verbosity");
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
  Rpipe          = hector_par.getParameter<double>("InnerPipeRadius" );
  x0_zdc         = hector_par.getParameter<double>("XpipeCenterAtZ140" );
  y0_zdc         = hector_par.getParameter<double>("YpipeCenterAtZ140" );
  etacut         = hector_par.getParameter<double>("AddEtaCut" );
  
  //  etacut = 8.2;
  //  m_smearAng = true;
  //  Rpipe=3.5;// inner radius of beam pipe mm
  //  x0_zdc=0.0;// X0-coord. center of beam pipe at z=140m 
  // y0_zdc=0.0;// Y0-coord. center of beam pipe at z=140m 
  
  edm::LogInfo ("Hector") << "Hector parameters: \n" 
			  << "   Verbosity: " << m_verbosity << "\n"
			  << "   lengthfp420:    " << lengthfp420 << "\n"
			  << "   m_rpp420_f:    " << m_rpp420_f << "\n"
			  << "   m_rpp420_b:    " << m_rpp420_b << "\n"
			  << "   lengthzdc:    " << lengthzdc << "\n"
			  << "   lengthd1:    " << lengthd1 << "\n";
  
  if(m_verbosity){
    std::cout<<"============================================================================"<<std::endl;
    std::cout<<"=========Hector parameters: :"
	     << "   lengthfp420:    " << lengthfp420 << "\n"
	     << "   m_rpp420_f:    " << m_rpp420_f << "\n"
	     << "   m_rpp420_b:    " << m_rpp420_b << "\n"
	     << "   lengthzdc:    " << lengthzdc << "\n"
	     << "   lengthd1:    " << lengthd1 << "\n" <<std::endl;
  }
  edm::FileInPath b1(beam1filename.c_str());
  edm::FileInPath b2(beam2filename.c_str());
  
  // construct beam line for FP420:                                                                                           .
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
  
  
  
  edm::LogInfo ("Hector") << "===================================================================\n";
  
}

Hector::~Hector(){
  
  edm::LogInfo ("Hector") << "===================================================================\n"  
                          << "=== Start delete Hector                                       =====\n";
  for (std::map<unsigned int,H_BeamParticle*>::iterator it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
    delete (*it).second;
  };
  //
  delete m_beamlineFP4201;
  delete m_beamlineFP4202;
  
  edm::LogInfo ("Hector") << "===================================================================\n";  
}

void Hector::clear(){
  for ( std::map<unsigned int,H_BeamParticle*>::iterator it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
    delete (*it).second;
  };
  
  m_beamPart.clear();
  m_direct.clear();
  m_isStoppedfp420.clear();
  m_isStoppedzdc.clear();
  m_isStoppedd1.clear();
  
  //  m_eta.clear();
  //  m_pdg.clear();
  //  m_pz.clear();
  
}

void Hector::add( const HepMC::GenEvent * evt ) {
  
  H_BeamParticle * h_p;
  double px,py,pz,pt;
  unsigned int line;
  
  //  unsigned int npart = ev->nStable();
  const HectorGenParticle * part;
  
  /*  for (HepMC::GenEvent::particle_const_iterator eventParticle =evt->particles_begin();
      eventParticle != evt->particles_end();
      eventParticle = find_if(++eventParticle, evt->particles_end(), ( (*eventParticle)->status() == 1 ) ) ) {*/
  for (HepMC::GenEvent::particle_const_iterator eventParticle =evt->particles_begin();
       eventParticle != evt->particles_end();
       ++eventParticle ) {
    if ( (*eventParticle)->status() == 1 ) {
      part = new HectorGenParticle( *(*eventParticle) );
      //      if ( abs( part->pdg_id() ) == 2212 ) { // if it's proton
      if ( abs( part->momentum().eta())>etacut){
	line = part->barcode();
	if ( m_beamPart.find(line) == m_beamPart.end() ) {
	  h_p = new H_BeamParticle();
	  
	  px = part->px();
	  py = part->py();
	  pz = part->pz();
	  
	  h_p->set4Momentum( px, py, pz, part->e() );
	  
	  //	pt = sqrt( (px*px) + (py*py) );
	  pt = part->pt();
	  /// Clears H_BeamParticle::positions and sets the initial one
	    // h_p->setPosition( m_IPx + part->x(), m_IPy + part->y(), std::atan2( px, pt ), std::atan2( py, pt ), m_IPz + part->z() );
	  h_p->setPosition( m_IPx + (*eventParticle)->production_vertex()->position().x(), m_IPy + (*eventParticle)->production_vertex()->position().y(), std::atan2( px, pt ), std::atan2( py, pt ), m_IPz + (*eventParticle)->production_vertex()->position().z() );
	    
	    m_beamPart[line] = h_p;
	    m_direct[line] = ( pz > 0 ) ? 1 : -1;
	    
	    if (m_smearAng) {   
	      h_p->smearAng();
	    }
	    //    m_eta[line] = part->momentum().eta();
	    // m_pdg[line] = part->pdg_id();
	    //  m_pz[line]  = part->momentum().pz();
	    
	}// if find line
      }// if 2212 or eta 8.2
    }// if status
  }// for loop
  
}


void Hector::filterFP420(){
  unsigned int line;
  H_BeamParticle * part;
  std::map< unsigned int, H_BeamParticle* >::iterator it;
  
  bool is_stop=true;
  int direction;
  
  float x1_420;
  float y1_420;
  
  if ( m_beamPart.size() && lengthfp420>0. ) {
    
    for (it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
      line = (*it).first;
      part = (*it).second;
      
      
      //propagating
      direction = (*(m_direct.find( line ))).second;
      if ( direction == 1 ) {
	part->computePath( m_beamlineFP4201, 1 );
	is_stop = part->stopped( m_beamlineFP4201 );
      }
      else {
	part->computePath( m_beamlineFP4202, -1 );
	is_stop = part->stopped( m_beamlineFP4202 );
      }
      
      m_isStoppedfp420[line] = is_stop;
      if (!is_stop) {
	part->propagate( m_rpp420_f );
	x1_420 = part->getX();
	y1_420 = part->getY();
	
	m_xAtTrPoint[line]  = x1_420;
	m_yAtTrPoint[line]  = y1_420;
	m_TxAtTrPoint[line] = part->getTX();
	m_TyAtTrPoint[line] = part->getTY();
	m_eAtTrPoint[line]  = part->getE();
	
      }
      
    } // for (it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) 
  } // if ( m_beamPart.size() )
  
}

void Hector::filterZDC(){
  unsigned int line;
  H_BeamParticle * part;
  std::map< unsigned int, H_BeamParticle* >::iterator it;
  
  bool is_stop_zdc=true;
  bool is_stop_d1=true;
  int direction;
  
  float x1_d1;
  float y1_d1;
  
  if ( m_beamPart.size() && lengthzdc>0. && lengthd1>0.) {
    
    for (it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
      line = (*it).first;
      part = (*it).second;
      if ( ((*m_isStoppedfp420.find(line)).second) ) {
	
       	direction = (*(m_direct.find( line ))).second;
	if ( direction == 1 ) {
	  part->computePath( m_beamlineFP4201, 1 );
	}
	else {
	  part->computePath( m_beamlineFP4202, -1 );
	}
	part->propagate(  lengthzdc );
	x1_d1 = part->getX()*0.001;
	y1_d1 = part->getY()*0.001;
	if(sqrt((x1_d1-x0_zdc)*(x1_d1-x0_zdc)+(y1_d1-y0_zdc)*(y1_d1-y0_zdc))< Rpipe) { is_stop_zdc=false;}
	m_isStoppedzdc[line] = is_stop_zdc;
	
	part->propagate(  lengthd1 );
	x1_d1 = part->getX()*0.001;
	y1_d1 = part->getY()*0.001;
	if(sqrt((x1_d1)*(x1_d1)+(y1_d1)*(y1_d1))< Rpipe) { is_stop_d1=false;}
	m_isStoppedd1[line] = is_stop_d1;
	
	
	if (!is_stop_d1 && is_stop_zdc) {
	  part->propagate(  lengthd1);
	  x1_d1 = part->getX();
	  y1_d1 = part->getY();
	  m_xAtTrPoint[line]  = x1_d1;
	  m_yAtTrPoint[line]  = y1_d1;
	  m_TxAtTrPoint[line] = part->getTX();
	  m_TyAtTrPoint[line] = part->getTY();
	  m_eAtTrPoint[line]  = part->getE();
	}
      }// if stopfp420
      else {
	m_isStoppedzdc[line] = is_stop_zdc;
	m_isStoppedd1[line] = is_stop_d1;
      }
    } // for (it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) 
  } // if ( m_beamPart.size() )
  // cout << "=================== Hector:filterZDC end: " << endl;
  
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
/*
  std::vector<unsigned int> Hector::part_list() const {
  std::vector<unsigned int> list( m_beamPart.size() );
  std::map<unsigned int, H_BeamParticle*>::const_iterator it;
  int ii = 0;
  for (it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
  list[ii] = (*it).first;
  ++ii;
  };
  return list;
  }
*/
HepMC::GenEvent * Hector::addPartToHepMC( HepMC::GenEvent * evt ){
  
  unsigned int line;
  //  H_BeamParticle * part;
  HepMC::GenParticle * gpart;
  long double tx,ty,theta,fi,energy,time = 0;
  std::map< unsigned int, H_BeamParticle* >::iterator it;
  
  
  for (it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) {
    line = (*it).first;
    if ( !((*m_isStoppedfp420.find(line)).second) || (!((*m_isStoppedd1.find(line)).second) && ((*m_isStoppedzdc.find(line)).second)) ) {
      gpart = evt->barcode_to_particle( line );
      if ( gpart ) {
	tx     = (*m_TxAtTrPoint.find(line)).second / 1000000.;
	ty     = (*m_TyAtTrPoint.find(line)).second / 1000000.;
	theta  = sqrt((tx*tx) + (ty*ty));
	double ddd = 0.;
	if ( !((*m_isStoppedfp420.find(line)).second)) {
	  if( (*(m_direct.find( line ))).second >0 ) {
	    ddd = m_rpp420_f;
	  }
	  else if((*(m_direct.find( line ))).second <0 ) {
	    ddd = m_rpp420_b;
	    theta= pi-theta;
	  }
	}
	else {
	  ddd = lengthd1;
	  if((*(m_direct.find( line ))).second <0 ) theta= pi-theta;
	}
	
	fi     = std::atan2(tx,ty); // tx, ty never == 0?
	energy = (*(m_eAtTrPoint.find(line))).second;
	
	HepMC::GenEvent::vertex_iterator v_it;
	
	time   = ( *evt->vertices_begin() )->position().t(); // does time important?
	
	long double time_buf = 0;
	for (v_it = evt->vertices_begin(); v_it != evt->vertices_end(); ++v_it)  { // since no operator--
	  time_buf = ( *v_it )->position().t();
	  time = (  time > time_buf ) ? time : time_buf;
	}
	if(ddd != 0.) {
	  HepMC::GenVertex * vert = new HepMC::GenVertex( HepMC::FourVector( (*(m_xAtTrPoint.find(line))).second*0.001,
									     (*(m_yAtTrPoint.find(line))).second*0.001,
									     ddd * (*(m_direct.find( line ))).second*1000.,
									     time + .1*time ) );
	  
	  
	  
          gpart->set_status( 2 );
          vert->add_particle_in( gpart );
          vert->add_particle_out( new HepMC::GenParticle( HepMC::FourVector(energy*std::sin(theta)*std::sin(fi),
                                                                            energy*std::sin(theta)*std::cos(fi),
                                                                            energy*std::cos(theta),
                                                                            energy ),
                                                          gpart->pdg_id(), 1, gpart->flow() ) );
          evt->add_vertex( vert );
	  
	  
	  
	}// ddd
      }// if gpart
    }// if !isStopped
  }//for 
  //  cout << "=== Hector:addPartToHepMC: end " << endl;
  
  if(m_verbosity){
    std::cout<<"============================================================================"<<std::endl;
    std::cout<<"=========Hector::addPartToHepMC print:"<<std::endl;
    evt->print();
  }
  
  
  
  return evt;
} 
