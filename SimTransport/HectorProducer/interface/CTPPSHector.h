#ifndef SimTransport_CTPPSHector_h
#define SimTransport_CTPPSHector_h

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
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

//CTPPSHector headers
#include "H_BeamLine.h"
#include "H_RecRPObject.h"
#include "H_BeamParticle.h"

#include <string>
#include <map>
#include <cmath>
#include <math.h>

#include "SimTransport/HectorProducer/interface/CTPPSHectorParameters.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include <CLHEP/Vector/LorentzVector.h>

class TRandom3;

class CTPPSHector {

    public:
        //  CTPPSHector(const edm::ParameterSet & ps);
        CTPPSHector(const edm::ParameterSet & ps, bool verbosity,bool CTPPSTransport);
        //  CTPPSHector();
        virtual ~CTPPSHector();
        typedef CLHEP::HepLorentzVector LorentzVector;
        /*!Clears ApertureFlags, prepares CTPPSHector for a next event*/
        void clearApertureFlags();
        /*!Clears BeamParticle, prepares CTPPSHector for a next Aperture check or/and a next event*/
        void clear();

        /*!Adds the stable protons from the event \a ev to a beamline*/
        void add( const HepMC::GenEvent * ev , const edm::EventSetup & es, CLHEP::HepRandomEngine * engine);

        /*!propagate the particles through a beamline to CTPPS*/
        void filterCTPPS(TRandom3*);

        // New function to calculate the LorentzBoost 
        void LorentzBoost(LorentzVector& p_out, const string& frame);

        void set_BeamEnergy(double e) {fBeamEnergy=e;fBeamMomentum = sqrt(fBeamEnergy*fBeamEnergy - ProtonMassSQ);};

        double get_BeamEnergy() {return fBeamEnergy;};

        double get_BeamMomentum() {return fBeamMomentum;};

        void ApplyBeamCorrection(LorentzVector&, CLHEP::HepRandomEngine * engine);

        int getDirect( unsigned int part_n ) const;

        /*!Prints properties of all particles in a beamline*/
        void print() const;

        HepMC::GenEvent * addPartToHepMC( HepMC::GenEvent * event );

        std::vector<LHCTransportLink> & getCorrespondenceMap() { return theCorrespondenceMap; }

    private:
        // Defaults
        double lengthctpps ;

        double etacut;
        bool m_smearAng;
        double m_sig_e;
        bool m_smearE;
        double m_sigmaSTX;
        double m_sigmaSTY;
        float m_f_ctpps_f;
        float m_b_ctpps_b;	

        //HECTOR CTPPS Parameters
        bool fCrossAngleCorr;
        double fCrossingAngle;
        double fBeamMomentum;
        double fBeamEnergy;
        double fVtxMeanX;
        double fVtxMeanY;
        double fVtxMeanZ;
        double fMomentumMin;

        edm::ESHandle < ParticleDataTable > pdt;

        // CTPPSHector
        H_BeamLine * m_beamlineCTPPS1;
        H_BeamLine * m_beamlineCTPPS2;	
        H_RecRPObject * m_ctpps_f;
        H_RecRPObject * m_ctpps_b; 
        std::map<unsigned int, H_BeamParticle*> m_beamPart;
        std::map<unsigned int, int> m_direct;
        std::map<unsigned int, bool> m_isStoppedctpps;
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
        bool m_CTPPSTransport;

        std::vector<LHCTransportLink> theCorrespondenceMap;
        int NEvent;
};
#endif
