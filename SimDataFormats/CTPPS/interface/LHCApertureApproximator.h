#ifndef SimRomanPot_CTPPSOpticsParameterisation_LHCApertureApproximator_h
#define SimRomanPot_CTPPSOpticsParameterisation_LHCApertureApproximator_h

/*#include <string>
#include <iostream>
#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TFile.h"

#include "TMultiDimFet.h"*/
#include "LHCOpticsApproximator.h"

/// \brief Aperture approximator
class LHCApertureApproximator : public LHCOpticsApproximator
{
  public:
    enum aperture_type { NO_APERTURE, RECTELLIPSE };

    LHCApertureApproximator();
    LHCApertureApproximator( const LHCOpticsApproximator &in,
                             double rect_x, double rect_y, double r_el_x, double r_el_y,
                             aperture_type type = RECTELLIPSE );
    ~LHCApertureApproximator() {} //FIXME

    bool CheckAperture( double *in, bool invert_beam_coord_sytems=true );

  private:
    double rect_x_, rect_y_, r_el_x_, r_el_y_;
    aperture_type ap_type_;
};

#endif
