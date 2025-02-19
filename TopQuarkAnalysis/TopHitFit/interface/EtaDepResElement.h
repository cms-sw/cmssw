//
//     $Id: EtaDepResElement.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// CMSSW File      : interface/EtaDepResElement.h
// Original Author : Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
// Purpose         : Hold on to an element of eta-dependent resolution.
//                   A class which holds to the information of resolution
//                   and the eta range in which the resolution is valid.
//

/**

    @file EtaDepResElement.h
    @brief Hold on to an element of  \f$ \eta \f$ -dependent resolution object,
    namely a resolution and  \f$ \eta \f$  range in which the resolution is valid.

    @author
    Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Creation date:
    June 2009

 */

#ifndef HITFIT_ETA_DEP_RES_ELEMENT
#define HITFIT_ETA_DEP_RES_ELEMENT

#include <iostream>

#include "TopQuarkAnalysis/TopHitFit/interface/Defaults_Text.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Vector_Resolution.h"


namespace hitfit {

/**
    @class EtaDepResElement

    @brief Represent a resolution and an  \f$ \eta \f$  range in which the
    resolution is valid.

    @par Usage:
    Users instantiate multiple instances of this class.  For each instance
    users provide:

    - The lower limit of the valid  \f$ \eta \f$  range.
    - The upper limit of the valid  \f$ \eta \f$  range.
    - The resolution, in one of the three recognized forms:
      - string encoded Vector_Resolution.
      - an instance of Vector_Resolution.
      - three instances of Resolution which forms a proper Vector_Resolution,

    See the documentation for Vector_Resolution and Resolution classes for
    more details.

    @author
    Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Creation date:
    June 2009

 */
class EtaDepResElement {

private:

    /**
       Lower limit of the valid  \f$ \eta \f$  range.
     */
    double            _EtaMin;

    /**
       Upper limit of the valid  \f$ \eta \f$  range.
     */
    double            _EtaMax;

    /**
       Set the lower and upper limit of the valid eta range.
       @param eta1 Value of  \f$ \eta \f$  in one end of the valid  \f$ \eta \f$
       range.
       @param eta2 Value of  \f$ \eta \f$  in the other end of the valid
        \f$ \eta \f$  range.
     */
    void              SetEta(double eta1,double eta2);

    /**
       The resolution.
     */
    Vector_Resolution _Vector_Resolution;

public:

    /**
       @brief Construct an instance of EtaDepResElement
       from the lower limit, upper limit, and the resolution.  The constructor
       will determine automatically which one among the two input  \f$ \eta \f$ s
       is the lower (upper) limit.

       @param eta1 Value of  \f$ \eta \f$  in one end/edge/boundary of the
       valid  \f$ \eta \f$  range.
       @param eta2 Value of  \f$ \eta \f$  in the other end/edge/boundary of the
       valid  \f$ \eta \f$  range.
       @param res The resolution.
     */
    EtaDepResElement(double eta1, double eta2,
                     const Vector_Resolution& res);

    /**
       @brief Construct an instance of EtaDepResElement
       from the lower limit, upper limit, and a string which encoded the
       resolution.  The constructor
       will determine automatically which one among the two input  \f$ \eta \f$ s
       is the lower (upper) limit.

       @param eta1 Value of  \f$ \eta \f$  in one end/edge/boundary of the
       valid  \f$ \eta \f$  range.
       @param eta2 Value of  \f$ \eta \f$  in the other end/edge/boundary of the
       valid  \f$ \eta \f$  range.
       @param res The resolution encoded in string.
     */
    EtaDepResElement(double eta1, double eta2,
                     std::string res);

    /**
       @brief Construct an instance of EtaDepResElement
       from the lower limit, upper limit, and the resolution.  The constructor
       will determine automatically which one among the two input  \f$ \eta \f$ s
       is the lower (upper) limit.

       @param eta1 Value of  \f$ \eta \f$  in one end of the valid  \f$ \eta \f$
       range.
       @param eta2 Walue of  \f$ \eta \f$  in the other end of the valid
        \f$ \eta \f$  range.
       @param p_res The energy/momentum resolution.
       @param eta_res The  \f$ \eta \f$  resolution.
       @param phi_res The  \f$ \phi \f$  resolution.
       @param use_et If true, then the energy/momentum resolution is
       for transverse component  \f$ (p_{T}/E_{T}) \f$  instead for
       radial  \f$ (p/E) \f$  component.
     */
    EtaDepResElement(double eta1, double eta2,
                     const Resolution& p_res,
                     const Resolution& eta_res,
                     const Resolution& phi_res,
                     bool use_et);

    /**
       Destructor.
     */
    ~EtaDepResElement();

    friend bool operator < (const EtaDepResElement& a,
                            const EtaDepResElement& b) ;

    /**
       @brief Return the lower limit of valid  \f$ \eta \f$  range.
     */
    const double EtaMin() const;

    /**
       @brief Return the lower limit of valid  \f$ \eta \f$  range.
     */
    const double EtaMax() const;

    /**
       @brief Check if this instance has overlapping  \f$ \eta \f$  range
       with another instance of EtaDepResElement.
       @param e The other instance of EtaDepResElement to be checked.
       @par Return:
       <b>true</b> if this instance has overlapping  \f$ \eta \f$  range
       with another instance's  \f$ \eta \f$  range.<br>
       <b>false</b> if this instance doesn't have overlapping  \f$ \eta \f$  range
       with another instance's  \f$ \eta \f$  range.
     */
    bool IsOverlap(const EtaDepResElement& e) const ;

    /**
       @brief Check if this instance does not have overlapping  \f$ \eta \f$  range
       with another instance.
       @param e The other instance of EtaDepResElement to be checked.
       @par Return:
       <b>true</b> if this instance does not have overlapping  \f$ \eta \f$  range
       with another instance's  \f$ \eta \f$  range.
       <b>false</b> if this instance has overlapping  \f$ \eta \f$  range
       with another instance's  \f$ \eta \f$  range.
     */
    bool IsNotOverlap(const EtaDepResElement& e) const ;

    /**
       @brief Check if an  \f$ \eta \f$  value is within
       this instance's  \f$ \eta \f$  range.
       @param eta The  \f$ \eta \f$  value to be checked.
       @par Return:
       <b>true</b> if  \f$ \eta \f$  is within the instance's  \f$ \eta \f$  range.
       <b>false</b> if  \f$ \eta \f$  is not within the instnace's  \f$ \eta \f$
       range.
     */
    bool IsInInterval(const double& eta) const ;


    /**
       @brief Check if an  \f$ \eta \f$  value is at the edge/boundary of
       this instance's valid  \f$ \eta \f$  range.
       @param eta The  \f$ \eta \f$  value to be checked.
     */
    bool IsOnEdge(const double& eta) const;

    /**
       @brief Check if another instance of EtaDepResElement lies
       at the edge/boundary of this instance's  \f$ \eta \f$  range.
       this instance's valid  \f$ \eta \f$  range.  A tolerance factor of
       1/1000000 is used.
       @param e The  \f$ \eta \f$  value to be checked.
       @par Return:
       <b>true</b> if  \f$ \eta \f$  is at the edge/boundary the instance's
        \f$ \eta \f$  range.
       <b>false</b> if  \f$ \eta \f$  is not at the edge/boundary
       within the instnace's  \f$ \eta \f$  range.
     */
    bool IsOnEdge(const EtaDepResElement& e) const;

    /**
       @brief Access the resolution.
       @par Return:
       The resolution.

     */
    const Vector_Resolution GetResolution() const;

    /**
       @brief Output stream operator.
       @param s The output stream to write to.
       @param e The instance of EtaDepResElement to be printed.
       @par Return:
       The output stream <i>s</i>
     */
    friend std::ostream& operator<<(std::ostream& s,
                                    const EtaDepResElement& e);

    /**
       @brief Constant, the inverse of precision expected.
     */
    static const int  InverseEtaPrecision = 1000000; // Precision of 1/1000000

};

} // namespace hitfit
#endif // not #ifndef HITFIT_ETA_DEP_RES_ELEMENT
