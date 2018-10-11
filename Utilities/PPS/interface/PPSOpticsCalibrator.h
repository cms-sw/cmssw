#ifndef PPS_OPTICS_CALIBRATION
#define PPS_OPTICS_CALIBRATION
#include "Utilities/PPS/interface/PPSUtilities.h"
#include <boost/program_options.hpp>
#include "TROOT.h"
#include "Rtypes.h"
#include "TH2F.h"
#include "TTree.h"
#include "TDirectory.h"
#include "TSystem.h"
#include "TRandom3.h"
#include "TFile.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <tuple>
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream, std::stringbuf

class H_BeamParticle;
class H_BeamLine;
class H_OpticalElement;

class PPSOpticsCalibrator {
public:
        enum OptElement_type {
             Quadrupole,
             Dipole,
             Kicker,
             Collimator,
             Invalid};

        PPSOpticsCalibrator(const std::string,const std::string,int =250);
        ~PPSOpticsCalibrator(){};

        void AlignBeamLine();
        void BeamProfile(TFile* , int);
        void CalibrateBeamPositionatIP(double& xpos,double& ypos);
        void ReadEmittance(std::ifstream& tabfile);
        void ReadParameterIndex(std::ifstream& tabfile);
        void FindIP(std::ifstream& tabfile);
        void ReadBeamPositionFromOpticsFile(std::ifstream& tabfile);

        OptElement_type OptElementType(const std::string type) {
             if (type.find("Quadrupole")<type.length()) return Quadrupole;
             if (type.find("Dipole")<type.length())     return Dipole;
             if (type.find("Kicker")<type.length())     return Kicker;
             if (type.find("Collimator")<type.length()) return Collimator;
             return Invalid;
        };


private:
        void AlignObject(H_OpticalElement* opt,H_BeamLine* bline,double disp);

        bool IP_Found;
        bool ParIdx_Found;
        bool Emmitance_OK;
        int    beamline_length;
        std::unique_ptr<H_BeamLine> m_beamline56;
        std::unique_ptr<H_BeamLine> m_beamline45;
        double fBeamXatIP;
        double fBeamYatIP;
        double m_sigmaSX;
        double m_sigmaSY;
        double m_sigmaSTX;
        double m_sigmaSTY;
        double m_sigE;
        int x_idx = 0;
        int s_idx = 0;
        int y_idx = 0; 
        int betx_idx = 0;
        int bety_idx = 0;
        double IPposition;
        double emittanceX;
        double emittanceY; 
        std::vector<std::tuple<double,double,double,double,double> >  PosP;
        std::vector<std::tuple<double,double,double,double,double> >  PosN;
        std::vector<std::tuple<double,double,double> >  BdistP;
        std::vector<std::tuple<double,double,double> >  BdistN;

};
#endif
