//
//     $Id: EtaDepResolution.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// CMSSW File      : interface/EtaDepResolution.h
// Original Author : Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
// Purpose         : Hold on to eta-dependent resolution.
//                   Return Vector_Resolution as a function of eta.
//                   Does not assume symmetry between +/- eta.
//                   The interpretation of eta (physics or detector) is
//                   left to user's implementation.
//

/**

    @file EtaDepResolution.h

    @brief Hold on to \f$\eta\f$-dependent resolution.
    This class acts
    as a function object and returns Vector_Resolution as a function
    of \f$\eta\f$.  It does not assume symmetry between \f$+\eta\f$ and
    \f$-\eta\f$.  The interpretation of \f$\eta\f$ as physics \f$\eta\f$
    or detector \f$\eta\f$ is left to users's implementation.

    @author
    Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Creation date:
    June 2009

 */


#ifndef HITFIT_ETA_DEP_RESOLUTION
#define HITFIT_ETA_DEP_RESOLUTION

#include "TopQuarkAnalysis/TopHitFit/interface/Defaults_Text.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Vector_Resolution.h"
#include "TopQuarkAnalysis/TopHitFit/interface/EtaDepResElement.h"

namespace hitfit {

/**
    @class EtaDepResolution

    @brief Hold on to \f$\eta\f$-dependent resolution.  This class acts
    as a function object and returns Vector_Resolution as a function
    of \f$\eta\f$.  It does not assume symmetry between \f$+\eta\f$ and
    \f$-\eta\f$.  The interpretation of \f$\eta\f$ as physics \f$\eta\f$
    or detector \f$\eta\f$ is left to users's implementation.

    @par Usage:
    Users should write the \f$\eta\f$-dependent resolution in a plain ASCII
    text file with the following format:<br>

    etadep_etaminX = etamin

    etadep_etamaxX = etamax

    etadep_vecresX = vecres

    where:

    <b>X</b> is an integer starting from 0 (0, 1, 2, 3, ...).  Users can
    write as many resolutions as long as each of them has different
    integer, and complete information regarding \f$\eta\f$ range an
    resolution.

    <b>etamin</b> is a floating-point number, the lower limit of the valid
    \f$\eta\f$ range.

    <b>etamax</b> is a floating-point number, the upper limit of the valid
    \f$\eta\f$ range.

    <b>vecres</b> is a string-encoded Vector_Resolution, see the
    documentation for Vector_Resolution class for details.

    The constructor will read the ASCII text file and read all resolutions
    in the file.  Then it will instantiate the EtaDepResolution and sort
    the internal list of EtaDepResElement.

    @author
    Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Creation date:
    June 2009.

 */
class EtaDepResolution {

private:

    /**
       List of \f$\eta\f$-dependent resolution elements.
     */
    std::vector<EtaDepResElement> _EtaDepResElement;

    /**
       @brief Internal method to return the corresponding \f$\eta\f$-dependent
       resolution element for a given \f$\eta\f$ value.
       @param eta The value of \f$\eta\f$ whose corresponding resolution
       is to be found.
       @par Return:
       An iterator (pointer) to the corresponding \f$\eta\f$-dependent
       resolution element.
     */
    std::vector<EtaDepResElement>::const_iterator FindResolution(double& eta) const;

public:

    /**
       Sort the internal list of \f$\eta\f$-dependent resolution elements.
     */
    void sort();

    /**
       @brief Default constructor, instantiate an EtaDepResolution
       object with empty list of \f$\eta\f$-dependent resolution elements.
     */
    EtaDepResolution();


    /**
       @brief Instantiate an EtaDepResolution
       object with a filename which contains information about
       \f$\eta\f$-dependent resolution.
       @param default_file The input ASCII text file.
     */
    EtaDepResolution(const std::string& default_file);

    /**
       Destructor.
     */
    ~EtaDepResolution();

    /**
       @brief Read the \f$\eta\f$-dependent resolution information
       from an ASCII text file.
       @param default_file The ASCII text file to read.
       @par Return:
       The number of \f$\eta\f$-dependent resolution element read from the file.
     */
    std::vector<EtaDepResElement>::size_type Read(const std::string& default_file);

    /**
       @brief Read the \f$\eta\f$-dependent resolution information
       from a Defaults_Text object.
       @param defs The Defaults_Text object to read.
       @par Return:
       The number of \f$\eta\f$-dependent resolution element read from the file.
     */
    std::vector<EtaDepResElement>::size_type Read(const Defaults_Text& defs);

    /**
       @brief Check for non-overlapping \f$\eta\f$-range between
       \f$\eta\f$-dependent resolution elements in a list.
       @param v The list of \f$\eta\f$-dependent resolution elements to check.
       @par Return:
       <b>true</b> if there is no overlap.<br>
       <b>false</b> if there is overlap.
     */
    bool CheckNoOverlap(const std::vector<EtaDepResElement>& v);

    /**
       @brief Return the lower limit of the valid \f$\eta\f$-range.
     */
    const double EtaMin() const;

    /**
       @brief Return the upper limit of the valid \f$\eta\f$-range.
     */
    const double EtaMax() const;

    /**
       @brief Check is an input \f$\eta\f$ value is within the valid \f$\eta\f$-range
       of this instance.
       @param eta The \f$\eta\f$ value to be check.
       @par Return:
       <b>true</b> if <i>eta</i> is within the valid \f$\eta\f$-range.<br>
       <b>false</b> if is not within the valid \f$\eta\f$-range.
     */
    const bool CheckEta(double eta) const;

    /**
       @brief Allow users to call this instance as a function to access
       the corresponding resolution for an input value of \f$\eta\f$.
       @param eta The \f$\eta\f$ value for which the corresponding resolution is desired.
       @par Return:
       The corresponding resolution if the input \f$\eta\f$ value is valid.
       Throw a runtime error if the \f$\eta\f$ value is invalid.
     */
    Vector_Resolution operator () (double& eta);

    /**
       @brief Return the corresponding resolution for a value of \f$\eta\f$.
       @param eta The \f$\eta\f$ value for which the corresponding resolution is desired.
       @par Return:
       The corresponding resolution if the input \f$\eta\f$ value is valid.
       Throw a runtime error if the \f$\eta\f$ value is invalid.
     */
    Vector_Resolution GetResolution(double& eta) const;


    /**
       @brief Return the corresponding \f$\eta\f$-dependent resolution element
       (of type EtaDepResElement)for a value of \f$\eta\f$.
       @param eta The \f$\eta\f$ value for which the corresponding element
       @par Return:
       The corresponding element if the input \f$\eta\f$ value is valid.
       Throw a runtime error if the \f$\eta\f$ value is invalid.
     */
    EtaDepResElement  GetEtaDepResElement(double& eta) const;

    /**
       @brief Access the internal list of \f$\eta\f$-dependent resolution
       elements.
       @par Return:
       The list of \f$\eta\f$-dependent resolution elements.
     */
    const std::vector<EtaDepResElement> GetEtaDepResElement() const;

};

} // namespace hitfit
#endif // not #ifndef HITFIT_ETA_DEP_RESOLUTION
