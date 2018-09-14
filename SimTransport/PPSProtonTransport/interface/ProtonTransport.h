#ifndef PROTONTRANSPORT
#define PROTONTRANSPORT
#include "FWCore/Framework/interface/EventSetup.h"
#include "HepMC/GenEvent.h"
#include "CLHEP/Random/RandGauss.h"
#include <CLHEP/Vector/LorentzVector.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/PPS/interface/PPSUnitConversion.h"

#include "SimDataFormats/Forward/interface/LHCTransportLink.h"
#include <vector>
#include <map>

class ProtonTransport {
      public:
            //ProtonTransport(const edm::ParameterSet & ps, bool verbosity);
            ProtonTransport();
            virtual ~ProtonTransport();
            std::vector<LHCTransportLink>& getCorrespondenceMap() { return theCorrespondenceMap; }
            virtual void process(const HepMC::GenEvent * ev , const edm::EventSetup & es, CLHEP::HepRandomEngine * engine)=0;
            void ApplyBeamCorrection(HepMC::GenParticle* p);
            void ApplyBeamCorrection(CLHEP::HepLorentzVector& p);
            void addPartToHepMC( HepMC::GenEvent * );
            void clear() { for (std::map<unsigned int,CLHEP::HepLorentzVector* >::iterator it = m_beamPart.begin(); it != m_beamPart.end(); ++it ) delete (*it).second;
                               m_beamPart.clear(); m_xAtTrPoint.clear(); m_yAtTrPoint.clear();
                         };

      protected:
            enum transport_mode {HECTOR,TOTEM};
            transport_mode MODE;
            int NEvent;
            bool m_verbosity;
            CLHEP::HepRandomEngine * engine;

            bool bApplyZShift;
            double fPPSRegionStart_56;
            double fPPSRegionStart_45;
            double fCrossingAngle_45;
            double fCrossingAngle_56;

            std::vector<LHCTransportLink> theCorrespondenceMap;
            std::map<unsigned int, CLHEP::HepLorentzVector*> m_beamPart;
            std::map<unsigned int, double> m_xAtTrPoint;
            std::map<unsigned int, double> m_yAtTrPoint;

            double m_sigmaSX;
            double m_sigmaSY;
            double m_sigmaSTX;
            double m_sigmaSTY;
            double m_sig_E;
            double fVtxMeanX;
            double fVtxMeanY;
            double fVtxMeanZ;
            double fBeamXatIP;
            double fBeamYatIP;
            double fBeamMomentum;
            double fBeamEnergy;

};
#endif
