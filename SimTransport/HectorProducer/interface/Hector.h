#ifndef SimTransport_Hector_h
#define SimTransport_Hector_h

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
/*
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
*/
// HepMC headers
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimDataFormats/Forward/interface/LHCTransportLink.h"
#include <vector>

// SimpleConfigurable replacement
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Hector headers
#include "H_BeamLine.h"
#include "H_RecRPObject.h"
#include "H_BeamParticle.h"
#include <string>
#include <map>

class TRandom3;

class Hector {
  
 public:
  //  Hector(const edm::ParameterSet & ps);
  Hector(const edm::ParameterSet & ps, bool verbosity, bool FP420Transport,bool ZDCTransport);
  //  Hector();
  virtual ~Hector();
  
  /*!Clears ApertureFlags, prepares Hector for a next event*/
  void clearApertureFlags();
  /*!Clears BeamParticle, prepares Hector for a next Aperture check or/and a next event*/
  void clear();
  /*!Adds the stable protons from the event \a ev to a beamline*/
  void add( const HepMC::GenEvent * ev , const edm::EventSetup & es);
  /*!propagate the particles through a beamline to FP420*/
  void filterFP420(TRandom3*);
  /*!propagate the particles through a beamline to ZDC*/
  void filterZDC(TRandom3*);
  /*!propagate the particles through a beamline to ZDC*/
  void filterD1(TRandom3*);
  
  int getDirect( unsigned int part_n ) const;

  /*!Prints properties of all particles in a beamline*/
    void print() const;
  /*!Return vector of the particle lines (HepMC::GenParticle::barcode()) in a beamline*/
  // std::vector<unsigned int> part_list() const;
    
    //    bool isCharged(const HepMC::GenParticle * p);
    
    HepMC::GenEvent * addPartToHepMC( HepMC::GenEvent * event );

    std::vector<LHCTransportLink> & getCorrespondenceMap() { return theCorrespondenceMap; }
    
    /*  
	private:
	//  edm::ParameterSet m_pBeamLine;
	*/
 private:
    
    
    // Defaults
    double lengthfp420 ;
    double lengthzdc ;
    double lengthd1 ;
    
    double etacut;
    bool m_smearAng;
    double m_sig_e;
    bool m_smearE;
    double m_sigmaSTX;
    double m_sigmaSTY;
   
    float m_rpp420_f;
    float m_rpp420_b;
    float m_rppzdc;
    float m_rppd1;
    
    edm::ESHandle < ParticleDataTable > pdt;
    
    // Hector
    H_BeamLine * m_beamlineFP4201;
    H_BeamLine * m_beamlineFP4202;
    H_BeamLine * m_beamlineZDC1;
    H_BeamLine * m_beamlineZDC2;
    H_BeamLine * m_beamlineD11;
    H_BeamLine * m_beamlineD12;
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
    
    std::map<unsigned int, double> m_eta;
    std::map<unsigned int, int> m_pdg;
    std::map<unsigned int, double> m_pz;
    std::map<unsigned int, bool> m_isCharged;
    
    string beam1filename;
    string beam2filename;
    
    bool m_verbosity;
    bool m_FP420Transport;
    bool m_ZDCTransport;

    std::vector<LHCTransportLink> theCorrespondenceMap;
};
#endif
