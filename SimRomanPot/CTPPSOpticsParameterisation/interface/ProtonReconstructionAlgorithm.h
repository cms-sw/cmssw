#ifndef SimRomanPot_CTPPSOpticsParameterisation_ProtonReconstructionAlgorithm_h
#define SimRomanPot_CTPPSOpticsParameterisation_ProtonReconstructionAlgorithm_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/PtrVector.h"

#include "SimRomanPot/CTPPSOpticsParameterisation/interface/LHCOpticsApproximator.h"
#include "SimRomanPot/CTPPSOpticsParameterisation/interface/LHCApertureApproximator.h"

#include "SimDataFormats/CTPPS/interface/CTPPSSimProtonTrack.h"
#include "SimDataFormats/CTPPS/interface/CTPPSSimHit.h"

#include "TFile.h"
#include "TSpline.h"
#include "Fit/Fitter.h"

#include <map>
#include <string>
#include <cmath>

enum LHCSector { unknownSector, sector45, sector56 };

class ProtonReconstructionAlgorithm
{
  public:
    ProtonReconstructionAlgorithm( const edm::ParameterSet& beam_conditions, const std::string& optics_file_beam );
    ~ProtonReconstructionAlgorithm();

    CTPPSSimProtonTrack Reconstruct( const std::vector<CTPPSSimHit>& tracks ) const;

  private:
    /// optics data associated with 1 RP
    struct RPOpticsData {
      LHCOpticsApproximator *optics;
      TSpline3 *s_xi_vs_x, *s_y0_vs_xi, *s_v_y_vs_xi, *s_L_y_vs_xi;
    };

    /// map: RP id --> optics data
    std::map<unsigned int, RPOpticsData> m_rp_optics;

    /// class for calculation of chi^2
    class ChiSquareCalculator {
      public:
        ChiSquareCalculator( const edm::ParameterSet& bc ) : beamConditions_( bc ) {}
        double operator() ( const double* ) const;

        const std::vector<CTPPSSimHit>* tracks;
        const std::map<unsigned int, RPOpticsData>* m_rp_optics;

      private:
        edm::ParameterSet beamConditions_;
    };

    ChiSquareCalculator *chiSquareCalculator = NULL;

    // fitter object
    std::unique_ptr<ROOT::Fit::Fitter> fitter_;

    edm::ParameterSet beamConditions_;
};

#endif
