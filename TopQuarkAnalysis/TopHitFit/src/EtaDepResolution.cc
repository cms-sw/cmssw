//
//     $Id: EtaDepResolution.cc,v 1.2 2011/10/13 12:34:16 snaumann Exp $
//
// File   : src/EtaDepResolution.cc
// Author : Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
// Purpose: Hold on to eta-dependent resolution.
//

/**

    @file EtaDepResolution.cc

    @brief Hold on to \f$\eta\f$-dependent
    resolution.  See the documentation for the header file EtaDepResolution.h
    for details.

    @author
    Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Creation date:
    June 2009

 */


#include <algorithm>
#include <sstream>
#include <stdexcept>
#include "TopQuarkAnalysis/TopHitFit/interface/EtaDepResolution.h"

namespace hitfit {

std::vector<EtaDepResElement>::const_iterator
EtaDepResolution::FindResolution(double& eta) const
{

    for (std::vector<EtaDepResElement>::const_iterator res = _EtaDepResElement.begin() ;
         res != _EtaDepResElement.end() ;
         ++res) {
        if (res->IsInInterval(eta) || res->IsOnEdge(eta)) {
            return res;
        }
    }
    return _EtaDepResElement.end() ;

}


void
EtaDepResolution::sort()
{
    std::stable_sort(_EtaDepResElement.begin(),_EtaDepResElement.end());
}


EtaDepResolution::EtaDepResolution()
{
}


EtaDepResolution::EtaDepResolution(const std::string& default_file)
{
    Read(default_file);
}


EtaDepResolution::~EtaDepResolution()
{
}


std::vector<EtaDepResElement>::size_type
EtaDepResolution::Read(const std::string& default_file)
{
    const Defaults_Text defs(default_file);
    Read(defs);
    return _EtaDepResElement.size();
}


std::vector<EtaDepResElement>::size_type
EtaDepResolution::Read(const Defaults_Text& defs)
{

    _EtaDepResElement.clear();

    for (std::vector<EtaDepResElement>::size_type i = 0 ;
         ;
         ++i) {

        std::ostringstream os_etamin ;
        std::ostringstream os_etamax ;
        std::ostringstream os_res ;

        os_etamin << "etadep_etamin" << i ;
        os_etamax << "etadep_etamax" << i ;
        os_res    << "etadep_vecres" << i ;

        if (defs.exists(os_etamin.str()) &&
            defs.exists(os_etamax.str()) &&
            defs.exists(os_res.str())) {

            double            etamin = defs.get_float(os_etamin.str());
            double            etamax = defs.get_float(os_etamax.str());
            Vector_Resolution res(defs.get_string(os_res.str()));
            _EtaDepResElement.push_back(EtaDepResElement(etamin,etamax,res));

        }
        else {
            break;
        }

    }

    if (CheckNoOverlap(_EtaDepResElement)) {
        sort();
    } else {
        _EtaDepResElement.clear();
    }

    return _EtaDepResElement.size();

}


bool
EtaDepResolution::CheckNoOverlap(const std::vector<EtaDepResElement>& v)
{
    for (std::vector<EtaDepResElement>::size_type i = 0 ;
         i != v.size() ;
         i++) {
        for (std::vector<EtaDepResElement>::size_type j = i + 1 ;
             j != v.size() ;
             j++) {
            if (v[i].IsOverlap(v[j])) {
                return false;
            }
        }
    }
    return true;
}


const double
EtaDepResolution::EtaMin() const
{
    if (!(_EtaDepResElement.empty())) {
        return _EtaDepResElement.front().EtaMin();
    }
    return 999.; // a ridiculously positive large number
}


const double
EtaDepResolution::EtaMax() const
{
    if (!(_EtaDepResElement.empty())) {
        return _EtaDepResElement.back().EtaMax();
    }
    return -999.; // a ridiculously negative large number
}


const bool
EtaDepResolution::CheckEta(double eta) const
{
    return FindResolution(eta) != _EtaDepResElement.end();
}


Vector_Resolution
EtaDepResolution::operator () (double& eta)
{
    return GetResolution(eta);
}


Vector_Resolution
EtaDepResolution::GetResolution(double& eta) const
{

    std::vector<EtaDepResElement>::const_iterator etaDepResEleVecIter = FindResolution(eta);
    if (etaDepResEleVecIter != _EtaDepResElement.end()) {
        return etaDepResEleVecIter->GetResolution();
    }

    std::stringstream message;
    message << "Error, the given eta value : "
            << eta << " is not inside the valid eta range!" ;

    throw std::runtime_error(message.str());
}

EtaDepResElement
EtaDepResolution::GetEtaDepResElement(double& eta) const
{
    return *(FindResolution(eta));
}


const std::vector<EtaDepResElement>
EtaDepResolution::GetEtaDepResElement() const
{
    return _EtaDepResElement;
}

} // namespace hitfit
