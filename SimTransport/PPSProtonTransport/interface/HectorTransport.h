#ifndef HECTOR_TRANSPORT
#define HECTOR_TRANSPORT
#include "SimTransport/PPSProtonTransport/interface/ProtonTransport.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Random/RandGauss.h"
#include "HepMC/SimpleVector.h"

#include <TMath.h>
#include <TMatrixT.h>
#include <TH2F.h>
#include <TFile.h>
#include <TEllipse.h>

#include <cmath>
#include <iomanip>
#include <cstdlib>


// HepMC headers
#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"
#include <CLHEP/Vector/LorentzVector.h>
//Hector headers
#include "H_BeamLine.h"
#include "H_BeamParticle.h"

// user include files
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <tuple>


//class TRandom3;

class HectorTransport: public ProtonTransport {
      public:
            typedef CLHEP::HepLorentzVector LorentzVector;

            HectorTransport();
            HectorTransport(const edm::ParameterSet & ps, bool verbosity);
            ~HectorTransport() override;
            void process( const HepMC::GenEvent * , const edm::EventSetup & , CLHEP::HepRandomEngine *) override;


      private:
            //!propagate the particles through a beamline to PPS
            bool  transportProton(const HepMC::GenParticle*);
            /*!Adds the stable protons from the event \a ev to a beamline*/

              //!Clears BeamParticle, prepares PPSHector for a next Aperture check or/and a next event
              void GenProtonsLoop( const HepMC::GenEvent *  , const edm::EventSetup &);

              // New function to calculate the LorentzBoost 
              void set_BeamEnergy(double e) {fBeamEnergy=e;fBeamMomentum = sqrt(fBeamEnergy*fBeamEnergy - ProtonMassSQ);}

              double get_BeamEnergy() {return fBeamEnergy;}

              double get_BeamMomentum() {return fBeamMomentum;}

              bool SetBeamLine();
/*
 *
 *                        ATTENTION:  DATA MEMBERS AND FUNCTIONS COMMON TO BOTH METHODS SHOULD BE MOVED TO THE BASE CLASS
 *
 */  
              // Defaults
            edm::ESHandle < ParticleDataTable > pdt;

            double fEtacut;
            double fMomentumMin;

            double lengthctpps ;
            double m_f_ctpps_f;
            double m_b_ctpps_b;


              // PPSHector
              std::unique_ptr<H_BeamLine> m_beamline45;
              std::unique_ptr<H_BeamLine> m_beamline56;

              string beam1filename;
              string beam2filename;

};
#endif
