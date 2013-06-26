//
//     $Id: EtaDepResElement.cc,v 1.1 2011/05/26 09:46:59 mseidel Exp $
//
//
// File   : src/EtaDepResolution.cc
// Author : Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
// Purpose: Hold on to an element eta-dependent resolution.
//

/**

    @file EtaDepResElement.cc

    @brief Hold on to an element of eta-dependent
    resolution object, namely a resolution and the eta range in which the
    resolution is valid.  See the documentation for the header file
    EtaDepResElement.h for details.

    @author
    Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Creation Date:
    June 2009

 */

#include <sstream>
#include <stdexcept>
#include "TopQuarkAnalysis/TopHitFit/interface/EtaDepResElement.h"

namespace hitfit {

EtaDepResElement::EtaDepResElement(double eta1,double eta2,
                                   const Vector_Resolution& res):
    _Vector_Resolution(res)
{

    SetEta(eta1,eta2);

}


EtaDepResElement::EtaDepResElement(double eta1,double eta2,
                                   std::string res):
    _Vector_Resolution(res)
{

    SetEta(eta1,eta2);

}


EtaDepResElement::EtaDepResElement(double eta1,double eta2,
                                   const Resolution& p_res,
                                   const Resolution& eta_res,
                                   const Resolution& phi_res,
                                   bool use_et):
    _Vector_Resolution(p_res,eta_res,phi_res,use_et)
{

    SetEta(eta1,eta2);

}


EtaDepResElement::~EtaDepResElement()
{
}


void
EtaDepResElement::SetEta(double eta1,double eta2)
{

    if (fabs(eta1 - eta2) < (1.0/double(InverseEtaPrecision))) {
        throw
            std::runtime_error("EtaDepResElement::equal EtaMin and EtaMax");
    }

    if (eta1 < eta2) {
        _EtaMin = eta1 ;
        _EtaMax = eta2 ;
    } else {
        _EtaMin = eta2 ;
        _EtaMax = eta1 ;
    }

}

/**
   @brief Comparison operator, compare two EtaDepResElement instances
   based on their respective valid \f$\eta\f$ ranges.

   @param a The first instance of EtaDepResElement to be compared.

   @param b The second instance of EtaDepResElement to be compared.

   @par Return:
   <b>TRUE</b> if <i>a</i>'s upper limit is less than <i>b</i>'s lower
   limit.<br>
   <b>FALSE</b> all other cases.
*/
bool operator < (const EtaDepResElement& a, const EtaDepResElement& b)
{

    if (a.IsOverlap(b)) { return false;}
    return !(a._EtaMax > b._EtaMin);

}


const double
EtaDepResElement::EtaMin() const
{

    return _EtaMin;

}


const double
EtaDepResElement::EtaMax() const
{

    return _EtaMax;

}


bool
EtaDepResElement::IsOverlap(const EtaDepResElement& e) const
{

    return (IsInInterval(e._EtaMin) || IsInInterval(e._EtaMax));

}


bool
EtaDepResElement::IsNotOverlap(const EtaDepResElement& e) const
{

    return !(IsOverlap(e));

}


bool
EtaDepResElement::IsInInterval(const double& eta) const
{

    return ((_EtaMin < eta) && (eta < _EtaMax));

}


bool
EtaDepResElement::IsOnEdge(const double& eta) const
{

    bool nearEtaMin = fabs(eta - _EtaMin) < (1.0/double(InverseEtaPrecision));
    bool nearEtaMax = fabs(eta - _EtaMax) < (1.0/double(InverseEtaPrecision));
    return nearEtaMin || nearEtaMax ;

}


bool
EtaDepResElement::IsOnEdge(const EtaDepResElement& e) const
{

    return (e.IsOnEdge(_EtaMin) || e.IsOnEdge(_EtaMax));

}


const Vector_Resolution
EtaDepResElement::GetResolution() const
{

    return _Vector_Resolution;

}


std::ostream& operator<< (std::ostream& s, const EtaDepResElement& e)
{

    s << "(" << e._EtaMin << " to " << e._EtaMax << ")" << " / " << e.GetResolution ();
    return s ;

}

} // namespace hitfit
