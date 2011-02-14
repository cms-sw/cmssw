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

#include "TRandom3.h"

class Hector {
  
 public:
  //  Hector(const edm::ParameterSet & ps);
  Hector(const edm::ParameterSet & ps, bool verbosity, bool FP420Transport, bool HPS240Transport, bool ZDCTransport);
  //  Hector();
  virtual ~Hector();
  
  /*!Clears ApertureFlags, prepares Hector for a next event*/
  void clearApertureFlags();
  /*!Clears BeamParticle, prepares Hector for a next Aperture check or/and a next event*/
  void clear();
  /*!Adds the stable protons from the event \a ev to a beamline*/
  void add( const HepMC::GenEvent * ev , const edm::EventSetup & es);
  /*!propagate the particles through a beamline to FP420*/
  void filterFP420();
  /*!propagate the particles through a beamline to HPS240*/
  void filterHPS240();
  /*!propagate the particles through a beamline to ZDC*/
  void filterZDC();
  
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
    double lengthhps240 ;
    double lengthzdc ;
    
    double etacut;
    bool m_smearAng;
    double m_sig_e;
    bool m_smearE;
    double m_sigmaSTX;
    double m_sigmaSTY;
   
    float m_rpp420_f;
    float m_rpp420_b;
    float m_rpp240_f;
    float m_rpp240_b;
    float m_rppzdc;
    
    edm::ESHandle < ParticleDataTable > pdt;
    
    // Hector
    H_BeamLine * m_beamlineFP4201;
    H_BeamLine * m_beamlineFP4202;
    H_BeamLine * m_beamlineHPS2401;
    H_BeamLine * m_beamlineHPS2402;
    H_BeamLine * m_beamlineZDC1;
    H_BeamLine * m_beamlineZDC2;
    //
    
//    H_RecRPObject * m_rp420_f;
//    H_RecRPObject * m_rp420_b;
    //    
    std::map<unsigned int, H_BeamParticle*> m_beamPart;
    std::map<unsigned int, int> m_direct;
    std::map<unsigned int, bool> m_isStoppedfp420;
    std::map<unsigned int, bool> m_isStoppedhps240;
    std::map<unsigned int, bool> m_isStoppedzdc;
    std::map<unsigned int, bool> m_isStoppedd1;
    //ZDC
    std::map<unsigned int, double> m_xAtZDCTrPoint;
    std::map<unsigned int, double> m_yAtZDCTrPoint;
    std::map<unsigned int, double> m_TxAtZDCTrPoint;
    std::map<unsigned int, double> m_TyAtZDCTrPoint;
    std::map<unsigned int, double> m_eAtZDCTrPoint;
    //FP420    
    std::map<unsigned int, double> m_xAtFP420TrPoint;
    std::map<unsigned int, double> m_yAtFP420TrPoint;
    std::map<unsigned int, double> m_TxAtFP420TrPoint;
    std::map<unsigned int, double> m_TyAtFP420TrPoint;
    std::map<unsigned int, double> m_eAtFP420TrPoint;
    //HPS240   
    std::map<unsigned int, double> m_xAtHPS240TrPoint;
    std::map<unsigned int, double> m_yAtHPS240TrPoint;
    std::map<unsigned int, double> m_TxAtHPS240TrPoint;
    std::map<unsigned int, double> m_TyAtHPS240TrPoint;
    std::map<unsigned int, double> m_eAtHPS240TrPoint;
    //
    std::map<unsigned int, double> m_eAtTrPoint;
    std::map<unsigned int, double> m_eta;
    std::map<unsigned int, int> m_pdg;
    std::map<unsigned int, double> m_pz;
    std::map<unsigned int, bool> m_isCharged;
    
    string beam1filename;
    string beam2filename;
    
    bool m_verbosity;
    bool m_FP420Transport;
    bool m_HPS240Transport;
    bool m_ZDCTransport;

    std::vector<LHCTransportLink> theCorrespondenceMap;

    TRandom3* rootEngine_;

};
#endif
