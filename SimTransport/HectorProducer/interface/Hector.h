#ifndef SimTransport_Hector_h
#define SimTransport_Hector_h

// HepMC headers
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

// SimpleConfigurable replacement
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Hector headers
#include "H_BeamLine.h"
#include "H_RecRPObject.h"
#include "H_BeamParticle.h"
#include <string>

class Hector {

public:
  Hector(const edm::ParameterSet & ps);
  //  Hector();
  virtual ~Hector();

  /*!Clears all data, prepares Hector for a next event*/
  void clear();
  /*!Adds the stable protons from the event \a ev to a beamline*/
  void add( const HepMC::GenEvent * ev );
  /*!propagate the particles through a beamline to FP420*/
  void filterFP420();
  /*!propagate the particles through a beamline to ZDC*/
  void filterZDC();

  int getDirect( unsigned int part_n ) const;

  /*!Prints properties of all particles in a beamline*/
    void print() const;
  /*!Return vector of the particle lines (HepMC::GenParticle::barcode()) in a beamline*/
  // std::vector<unsigned int> part_list() const;

  HepMC::GenEvent * addPartToHepMC( HepMC::GenEvent * event );

  /*  
private:
  //  edm::ParameterSet m_pBeamLine;
  */
private:
  bool m_verbosity;


  // Defaults
  double lengthfp420 ;
  double lengthzdc ;
  double lengthd1 ;

  float etacut;
  bool m_smearAng;

  float m_rpp420_f;
  float m_rpp420_b;
  float m_rppzdc;
  float m_rppd1;
  
  double m_IPx;
  double m_IPy;
  double m_IPz;
  double m_IPt;

  double Rpipe;// inner radius of beam pipe mm
  double x0_zdc;// X0-coord. center of beam pipe at z=140m 
  double y0_zdc;// Y0-coord. center of beam pipe at z=140m 

  // Hector
  H_BeamLine * m_beamlineFP4201;
  H_BeamLine * m_beamlineFP4202;
  //

    H_RecRPObject * m_rp420_f;
    H_RecRPObject * m_rp420_b;

  std::map<unsigned int, H_BeamParticle*> m_beamPart;
  std::map<unsigned int, int> m_direct;
  std::map<unsigned int, bool> m_isStoppedfp420;
  std::map<unsigned int, bool> m_isStoppedzdc;
  std::map<unsigned int, bool> m_isStoppedd1;
  std::map<unsigned int, double> m_xAtTrPoint;
  std::map<unsigned int, double> m_yAtTrPoint;
  std::map<unsigned int, double> m_TxAtTrPoint;
  std::map<unsigned int, double> m_TyAtTrPoint;
  std::map<unsigned int, double> m_eAtTrPoint;

  //std::map<unsigned int, double> m_eta;
  //std::map<unsigned int, int> m_pdg;
  //std::map<unsigned int, double> m_pz;


  string beam1filename;
  string beam2filename;
};
#endif
