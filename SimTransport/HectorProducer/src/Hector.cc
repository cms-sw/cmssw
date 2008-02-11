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
                          << "=== Start create new Hector                                   =====\n";

  // Create LHC beam line
  //  double length  = param.getParameter<double>("BeamLineLength");
  //  std::cout << " BeamLineLength = " << length << std::endl;

  edm::ParameterSet hector_par = param.getParameter<edm::ParameterSet>("Hector");
  // Defaults
  
  double length  = 430.;
  /*
  m_verbosity    = false;
  m_smearPos     = true;
  m_smearAng     = false;
  m_smearE       = false;
  m_smearS       = true;
  m_rpp420_f     = 420.;
  m_rpp420_b     = 420.;
  m_rpp220_f     = 220.;
  m_rpp220_b     = 220.;
  m_sig_e        = 0.;
  */
  // User definitons
  //  double length  = hector_par.getUntrackedParameter<double>("BeamLineLength", 430. );
  length         = hector_par.getParameter<double>("BeamLineLength" );
  m_verbosity    = hector_par.getParameter<bool>("Verbosity");
  m_smearPos     = hector_par.getParameter<bool>("smearPos");
  m_smearAng     = hector_par.getParameter<bool>("smearAng");
  m_smearE       = hector_par.getParameter<bool>("smearE");
  m_smearS       = hector_par.getParameter<bool>("smearS");
  m_rpp420_f     = (float) hector_par.getParameter<double>("RP420f");
  m_rpp420_b     = (float) hector_par.getParameter<double>("RP420b");
  m_rpp220_f     = (float) hector_par.getParameter<double>("RP220f");
  m_rpp220_b     = (float) hector_par.getParameter<double>("RP220b");
  m_sig_e        = hector_par.getParameter<double>("SigmaE");
  beam1filename  = hector_par.getParameter<string>("Beam1");
  beam2filename  = hector_par.getParameter<string>("Beam2");  
  

  edm::LogInfo ("Hector") << "Hector parameters: \n" 
			  << "   Verbosity: " << m_verbosity << "\n"
			  << "   smearPos:  " << m_smearPos << "\n"
			  << "   smearAng:  " << m_smearAng << "\n"
			  << "   smearE:    " << m_smearE << "\n"
			  << "   smearS:    " << m_smearS << "\n"
			  << "   RP420f:    " << m_rpp420_f << "\n"
			  << "   RP420b:    " << m_rpp420_b << "\n"
			  << "   RP220f:    " << m_rpp220_f << "\n"
			  << "   RP220b:    " << m_rpp220_b << "\n"
			  << "   SigmaE:    " << m_sig_e << "\n";


  //edm::FileInPath b1("SimTransport/HectorData/twiss_ip5_b1_v6.5.txt");
  //edm::FileInPath b2("SimTransport/HectorData/twiss_ip5_b2_v6.5.txt");

edm::FileInPath b1(beam1filename.c_str());
edm::FileInPath b2(beam2filename.c_str());

  m_beamline1 = new H_BeamLine(  1, length + 0.1 ); // (direction, length)
  m_beamline2 = new H_BeamLine( -1, length + 0.1 ); //

  try {
    m_beamline1->fill( b1.fullPath(),  1, "IP5" );
    m_beamline2->fill( b2.fullPath(), -1, "IP5" );
  } catch ( const edm::Exception& e ) {
    std::string msg = e.what();
    msg += " caught in Hector... \nERROR: Could not locate SimTransport/HectorData data files.";
    //    std::cout << msg << std::endl;
    edm::LogError ("DataNotFound") << msg;
  }

  
  //  m_rp420_f = new H_RecRPObject( m_rpp420_f, m_rpp420_f + 8., *m_beamline1 );
  //  m_rp420_b = new H_RecRPObject( m_rpp420_b, m_rpp420_b + 8., *m_beamline2 );
  
  /*
  m_rp220_f = new H_RecRPObject( m_rpp220_f, m_rpp220_f + 8., *m_beamline1 );
  m_rp220_b = new H_RecRPObject( m_rpp220_b, m_rpp220_b + 8., *m_beamline2 );
  */

  m_beamline1->offsetElements( 120, -0.097 );
  m_beamline2->offsetElements( 120, +0.097 );

  m_beamline1->calcMatrix();
  m_beamline2->calcMatrix();

  edm::LogInfo ("Hector") << "===================================================================\n";

  /*
  if (m_verbosity) {
    std::cout << "Forward beam: " << std::endl;
    m_beamline1->printProperties();
    std::cout << "Backward beam: " << std::endl;
    m_beamline2->printProperties();

    std::cout << "============BEAM 1==============" << std::endl;
    m_beamline1->showElements();
    std::cout << "============BEAM 2==============" << std::endl;
    m_beamline2->showElements();
  }
  */
}

Hector::~Hector(){

  edm::LogInfo ("Hector") << "===================================================================\n"  
                          << "=== Start delete Hector                                       =====\n";
  for (std::map<unsigned int,H_BeamParticle*>::iterator it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) {
    delete (*it).second;
  };
  //
  //  delete m_rp220_f;
  //  delete m_rp220_b;
  /*
  delete m_rp420_f;
  delete m_rp420_b;  

  delete m_beamline1;
  delete m_beamline2;
  */

  edm::LogInfo ("Hector") << "===================================================================\n";  
}

void Hector::clear(){
  for ( std::map<unsigned int,H_BeamParticle*>::iterator it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) {
    delete (*it).second;
  };

  m_beamPart.clear();
  m_direct.clear();
  m_isStopped.clear();
  /*
  m_gen_x.clear();
  m_gen_y.clear();
  m_gen_tx.clear();
  m_gen_ty.clear();
  m_gen_e.clear();
*/
  /*
  m_tx0.clear();
  m_ty0.clear();
  m_x0.clear();
  m_y0.clear();
  m_sim_e.clear();
*/
  m_shiftX = 0;
  m_shiftY = 0;
  m_shiftZ = 0;
}

void Hector::setIP( double ip_x, double ip_y, double ip_z, double ip_time){
  //  if (m_verbosity) std::cout << "IP point from CMSSW: " << ip_x << " " << ip_y << " " << ip_z << std::endl;
  LogDebug ("Hector") << "IP point from CMSSW: " << ip_x << " " << ip_y << " " << ip_z << std::endl;
  m_IPx = ip_x;
  m_IPy = ip_y;
  m_IPz = ip_z;
  m_IPt = ip_time;
}

unsigned int Hector::add( const HepMC::GenParticle * eventParticle ) {
  H_BeamParticle * h_p;
  const HectorGenParticle * part = new HectorGenParticle( *eventParticle );
  double px,py,pz,pt;
  unsigned int line;

  //  if ( abs( part->pdg_id() ) == 2212 ) { // if it's proton
    line = part->barcode();
    if ( m_beamPart.find(line) == m_beamPart.end() ) {
      h_p = new H_BeamParticle();

      px = part->px();
      py = part->py();
      pz = part->pz();

      h_p->set4Momentum( px, py, pz, part->e() );

      //      pt = sqrt( (px*px) + (py*py) );
      pt = part->pt();
      /// Clears H_BeamParticle::positions and sets the initial one
      h_p->setPosition( m_IPx + part->x(), m_IPy + part->y(), std::atan2( px, pt ), std::atan2( py, pt ), m_IPz + part->z() );

      m_beamPart[line] = h_p;
      m_direct[line] = ( pz > 0 ) ? 1 : -1;
      return line;
    }
    else {
      return line;
    }
    //  }
    //  else {
    //    return 0;
    //  }
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
       eventParticle++ ) {
       if ( (*eventParticle)->status() == 1 ) {
      part = new HectorGenParticle( *(*eventParticle) );
      //      if ( abs( part->pdg_id() ) == 2212 ) { // if it's proton
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
	  h_p->setPosition( m_IPx + part->x(), m_IPy + part->y(), std::atan2( px, pt ), std::atan2( py, pt ), m_IPz + part->z() );

	  m_beamPart[line] = h_p;
	  m_direct[line] = ( pz > 0 ) ? 1 : -1;
	}
	//      }// if ( m_beamPart.find(line) == m_beamPart.end() )
    }
  }

}


void Hector::reconstruct(){
  unsigned int line;
  H_BeamParticle * part;
  std::map< unsigned int, H_BeamParticle* >::iterator it;

  float part_x;
  float part_y;
  float part_z;
  float tx;
  float ty;

  bool is_stop;
  int direction;

  float x1_420;
  float y1_420;
  //  float x2_420;
  //  float y2_420;

  if ( m_beamPart.size() ) {
    it = m_beamPart.begin();
    line = (*it).first;
    part = (*it).second;

    if ( m_smearPos || m_smearS ) {
      part_x  = part->getX();
      part_y  = part->getY();
      part_z  = part->getS();
      tx = part->getTX();
      ty = part->getTY();

      if (m_smearPos) {
	part->smearPos();
	m_shiftX = part_x - part->getX();
	m_shiftY = part_y - part->getY();
      }

      if (m_smearS) {
	part->smearS();
	m_shiftZ = part_z - part->getS();
      }

      /// Clears H_BeamParticle::positions and sets the initial one

      part->setPosition( part_x, part_y, tx, ty, part_z );
    }

    for (it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) {
      line = (*it).first;
      part = (*it).second;
      //smearing
      // Position
      if ( m_smearPos || m_smearS) {
	part_x  = part->getX();
	part_y  = part->getY();
	part_z  = part->getS();
	tx = part->getTX();
	ty = part->getTY();

	LogDebug ("Hector") << "Particle angles before smearing: tx = " << tx << " ty = " << ty << std::endl;

	if (m_smearPos) {
	  part_x = part_x + m_shiftX;
	  part_y = part_y + m_shiftY;
	}

	if (m_smearS) {
	  part_z = part_z + m_shiftZ;
	}
	/// Clears H_BeamParticle::positions and sets the initial one
	part->setPosition( part_x, part_y, tx, ty, part_z );
      }
      //m_gen_x[line] = part->getX();
      //m_gen_y[line] = part->getY();

      if (m_smearAng) {   
	part->smearAng();
      }

      //_gen_tx[line] = part->getTX();
      //_gen_ty[line] = part->getTY();

      if (m_smearE) {
	if ( m_sig_e ) {
	  part->smearE(m_sig_e);
	}
	else {
	  part->smearE(); 
	}
      }
      //   m_gen_e[line] = part->getE();

      //propagating
      direction = (*(m_direct.find( line ))).second;
      if ( direction == 1 ) {
	part->computePath( m_beamline1, 1 );
	is_stop = part->stopped( m_beamline1 );
	m_isStopped[line] = is_stop;
	if (!is_stop) {
	  part->propagate( m_rpp420_f );
	  x1_420 = part->getX();
	  y1_420 = part->getY();

	  m_xAtRP420[line]  = x1_420;
	  m_yAtRP420[line]  = y1_420;
	  m_TxAtRP420[line] = part->getTX();
	  m_TyAtRP420[line] = part->getTY();
	  m_eAtRP420[line]  = part->getE();

/*	  part->propagate( m_rpp420_f + 8.);
	  x2_420 = part->getX();
	  y2_420 = part->getY();

	    	  if ( m_rp420_f ) {
		    //	std::cout << "  Hector: input coord. in um   " << std::endl;
		    //	std::cout << "   x1_420:    " << x1_420 << "   y1_420:    " << y1_420 << std::endl;
		    //	std::cout << "   x2_420:    " << x2_420 << "   y2_420:    " << y2_420 << std::endl;
		    
	    m_rp420_f->setPositions( x1_420, y1_420, x2_420 ,y2_420 );
	    m_sim_e[line] = m_rp420_f->getE( AM );
	    m_tx0[line] = m_rp420_f->getTXIP();
	    m_ty0[line] = m_rp420_f->getTYIP();
	    m_x0[line] = m_rp420_f->getX0();
	    m_y0[line] = m_rp420_f->getY0();
	    //  std::cout << "  Hector:     " << std::endl;
	    // std::cout << "   m_e:    " << m_sim_e[line] << std::endl;
	    // std::cout << "   m_x0:    " << m_x0[line] << std::endl;
	    // std::cout << "   m_y0:    " << m_y0[line] << std::endl;
	    // std::cout << "   m_tx0:    " << m_tx0[line]  << std::endl;
	    // std::cout << "   m_ty0:    " << m_ty0[line]  << std::endl;
	  }*/
	}
      }
      else {
	part->computePath( m_beamline2, -1 );
	is_stop = part->stopped( m_beamline2 );
	m_isStopped[line] = is_stop;
	if (!is_stop) {
	  part->propagate( m_rpp420_b );
	  x1_420 = part->getX();
	  y1_420 = part->getY();

	  m_xAtRP420[line]  = x1_420;
	  m_yAtRP420[line]  = y1_420;
	  m_TxAtRP420[line] = part->getTX();
	  m_TyAtRP420[line] = part->getTY();
	  m_eAtRP420[line]  = part->getE();
/*
	  part->propagate( m_rpp420_b + 8.);
	  x2_420 = part->getX();
	  y2_420 = part->getY();

	  	  if ( m_rp420_b ) {
	    m_rp420_b->setPositions( x1_420, y1_420, x2_420 ,y2_420 );
	    m_sim_e[line] = m_rp420_b->getE( AM );
	    m_tx0[line] = m_rp420_b->getTXIP();
	    m_ty0[line] = m_rp420_b->getTYIP();
	    m_x0[line] = m_rp420_b->getX0();
	    m_y0[line] = m_rp420_b->getY0();
	  } */

	}
      }
    } // for (it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) 
  } // if ( m_beamPart.size() )

}

/*
double Hector::getX ( double z_position, unsigned int part_n ) const {
  std::map<unsigned int, H_BeamParticle*>::const_iterator it = m_beamPart.find( part_n );
  if ( it != m_beamPart.end() ) {
    (*it).second->propagate( z_position );
    return (*it).second->getX();
  }
  else {
    return -100000000.;
  }
}

double Hector::getY ( double z_position, unsigned int part_n ) const {
  std::map<unsigned int,H_BeamParticle*>::const_iterator it = m_beamPart.find( part_n );
  if ( it != m_beamPart.end() ) {
    (*it).second->propagate( z_position );
    return (*it).second->getY();
  }
    return -100000000.;
}

bool Hector::isStopped( unsigned int part_n ) const {
  std::map<unsigned int, bool>::const_iterator it = m_isStopped.find( part_n );
  if ( it != m_isStopped.end() ){
    return (*it).second;
  }
  return true;
}
float Hector::getSimIPX( unsigned int part_n ) const {
  std::map<unsigned int, float>::const_iterator it = m_x0.find( part_n );
  if ( it != m_x0.end() ){
    return (*it).second;
  }
  return -100000000.;
}

float Hector::getSimIPY( unsigned int part_n ) const {
  std::map<unsigned int, float>::const_iterator it = m_y0.find( part_n );
  if ( it != m_y0.end() ){
    return (*it).second;
  }
  return -100000000.;
}

float Hector::getSimIPTX( unsigned int part_n ) const {
  std::map<unsigned int, float>::const_iterator it = m_tx0.find( part_n );
  if ( it != m_tx0.end() ){
    return (*it).second;
  }
  return -100000000.;
}

float Hector::getSimIPTY( unsigned int part_n ) const {
  std::map<unsigned int, float>::const_iterator it = m_ty0.find( part_n );
  if ( it != m_ty0.end() ){
    return (*it).second;
  }
  return -100000000.;
}

float Hector::getSimIPdE( unsigned int part_n ) const {
  std::map<unsigned int, float>::const_iterator it = m_sim_e.find( part_n );
  if ( it != m_sim_e.end() ){
    return (*it).second;
  }
  return -100000000.;
}

float Hector::getGenIPX( unsigned int part_n ) const {
  std::map<unsigned int, float>::const_iterator it = m_gen_x.find( part_n );
  if ( it != m_gen_x.end() ){
    return (*it).second;
  }
  return -100000000.;
}

float Hector::getGenIPY( unsigned int part_n ) const {
  std::map<unsigned int, float>::const_iterator it = m_gen_y.find( part_n );
  if ( it != m_gen_y.end() ){
    return (*it).second;
  }
  return -100000000.;
}

float Hector::getGenIPTX( unsigned int part_n ) const {
  std::map<unsigned int, float>::const_iterator it = m_gen_tx.find( part_n );
  if ( it != m_gen_tx.end() ){
    return (*it).second;
  }
  return -100000000.;
}

float Hector::getGenIPTY( unsigned int part_n ) const {
  std::map<unsigned int, float>::const_iterator it = m_gen_ty.find( part_n );
  if ( it != m_gen_ty.end() ){
    return (*it).second;
  }
  return -100000000.;
}

float Hector::getGenIPE( unsigned int part_n ) const {
  std::map<unsigned int, float>::const_iterator it = m_gen_e.find( part_n );
  if ( it != m_gen_e.end() ){
    return (*it).second;
  }
  return -100000000.;
}
*/
int  Hector::getDirect( unsigned int part_n ) const {
  std::map<unsigned int, int>::const_iterator it = m_direct.find( part_n );
  if ( it != m_direct.end() ){
    return (*it).second;
  }
  return 0;
}

void Hector::print() const {
  for (std::map<unsigned int,H_BeamParticle*>::const_iterator it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) {
    (*it).second->printProperties();
  };
}

std::vector<unsigned int> Hector::part_list() const {
  std::vector<unsigned int> list( m_beamPart.size() );
  std::map<unsigned int, H_BeamParticle*>::const_iterator it;
  int ii = 0;
  for (it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) {
    list[ii] = (*it).first;
    ii++;
  };
  return list;
}

HepMC::GenEvent * Hector::addPartToHepMC( HepMC::GenEvent * evt ){

  unsigned int line;
  //  H_BeamParticle * part;
  HepMC::GenParticle * gpart;
  long double tx,ty,theta,fi,energy,time = 0;
  std::map< unsigned int, H_BeamParticle* >::iterator it;

  for (it = m_beamPart.begin(); it != m_beamPart.end(); it++ ) {
    line = (*it).first;
    if ( !((*m_isStopped.find(line)).second) ) {
      gpart = evt->barcode_to_particle( line );
      if ( gpart ) {
	tx     = (*m_TxAtRP420.find(line)).second / 1000000.;
	ty     = (*m_TyAtRP420.find(line)).second / 1000000.;
	theta  = sqrt((tx*tx) + (ty*ty));
	double ddd = 0.;
	if( (*(m_direct.find( line ))).second >0 ) {
	  ddd = m_rpp420_f;
	}
	else if((*(m_direct.find( line ))).second <0 ) {
	  ddd = m_rpp420_b;
	  theta= pi-theta;
	}

	fi     = std::atan2(tx,ty); // tx, ty never == 0?
	energy = (*(m_eAtRP420.find(line))).second;

	HepMC::GenEvent::vertex_iterator v_it;

	time   = ( *evt->vertices_begin() )->position().t(); // does time important?

	long double time_buf = 0;
	for (v_it = evt->vertices_begin(); v_it != evt->vertices_end(); v_it++)  { // since no operator--
	  time_buf = ( *v_it )->position().t();
	  time = (  time > time_buf ) ? time : time_buf;
	}
	// Since no documentation suppouse vertex position in mm, like in HEPEVT

	//	HepMC::GenVertex * vert = new HepMC::GenVertex( HepMC::FourVector( (*(m_xAtRP420.find(line))).second,
	//									   (*(m_yAtRP420.find(line))).second,
	//									   420* (*(m_direct.find( line ))).second ) );
	/*
	if(abs(energy)>6000){
	  std::cout << " energy= " <<energy  << " theta= " << theta << " fi= " << fi << std::endl;
	  std::cout << " dir= " <<(*(m_direct.find( line ))).second  << " m_rpp420_f= " << m_rpp420_f << " cos(theta)= " << cos(theta) << std::endl;
	} 
*/
	if(ddd != 0.) {
	  HepMC::GenVertex * vert = new HepMC::GenVertex( HepMC::FourVector( (*(m_xAtRP420.find(line))).second*0.001,
										   (*(m_yAtRP420.find(line))).second*0.001,
										   ddd * (*(m_direct.find( line ))).second*1000.,
										   time + .1*time ) );
	  evt->add_vertex( vert );
	  vert->add_particle_in( gpart );
	  gpart->set_status( 2 );
	  vert->add_particle_out( new HepMC::GenParticle( HepMC::FourVector(energy*std::sin(theta)*std::sin(fi), 
										  energy*std::sin(theta)*std::cos(fi), 
										  energy*std::cos(theta),
										  energy ),
							  gpart->pdg_id(), 1, gpart->flow() ) );
	}// ddd
      }
    }
  }
  
  // Just chesk
  /*
  for (HepMC::GenEvent::vertex_iterator  vitr = evt->vertices_begin(); vitr != evt->vertices_end(); vitr++) {
    for (HepMC::GenVertex::particle_iterator  pitr= (*vitr)->particles_begin(HepMC::children); pitr != (*vitr)->particles_end(HepMC::children); ++pitr)
      {
	if (!(*pitr)->end_vertex() && (*pitr)->status()==1) {
	  std::cout << "Barcode of good particle: " << (*pitr)->barcode() << std::endl;
	}
      }
  }
  evt->print();
  */

  return evt;
} 
