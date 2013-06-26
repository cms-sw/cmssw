//
//     $Id: LeptonTranslatorBase.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//

/**
    @file LeptonTranslatorBase.h

    @brief Template class of function object to translate lepton physics
    object to HitFit's Lepjets_Event_Lep object.

    @author Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Creation date:
    Nov-2009

    @par Terms of Usage:
    With consent from the original author (Haryo Sumowidagdo).
 */

#ifndef HitFit_LeptonTranslatorBase_h
#define HitFit_LeptonTranslatorBase_h

#include "TopQuarkAnalysis/TopHitFit/interface/EtaDepResolution.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event_Lep.h"
#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"

namespace hitfit{


    /**
       @class LeptonTranslatorBase.

       @brief Template class of function object to translate lepton physics
       object to HitFit's Lepjets_Event_Lep object.    Users need to write an
       implementation of a template specialization of this class for their
       lepton physics object class.  Then users combine this header file and
       their implementation for their analysis code.  With this approach, it is
       possible to use HitFit for different lepton physics object class in
       different experiments.

       @param ALepton The typename of the lepton physics object class to
       be translated into HitFit's Lepjets_Event_Lep.

     */
    template <class ALepton>
    class LeptonTranslatorBase {


    public:

        /**
           @brief Default constructor.
         */
        LeptonTranslatorBase();

        /**
           @brief Constructor, instantiate a LeptonTranslatorBase object
           using the name of and input file in std::string format.

           @param ifile The path of the input file.
         */
        LeptonTranslatorBase(const std::string& ifile);

        /**
           @brief Destructor.
         */
        ~LeptonTranslatorBase();

        /**
           @brief Convert a lepton physics object of type ALepton into
           HitFit lepton physics object of type Lepjets_Event_Lep.

           @param lepton The lepton physics object to be translated.

           @param type The typecode of the lepton to be translated.

           @param useObjEmbRes Boolean parameter to indicate if the
           user would like to use the resolution embedded in the object,
           and not the resolution read when instantiating the class.
         */
        Lepjets_Event_Lep operator()(const ALepton& lepton,
                                     int type = hitfit::lepton_label,
                                     bool useObjEmbRes = false);

        /**
           @brief Return the  \f$ \eta- \f$ dependent resolution of the lepton.
         */
        const EtaDepResolution& resolution() const;

        /**
           @brief Check if a lepton has  \f$ \eta \f$  value which is within the
           valid  \f$ \eta \f$  range of the resolution.

           @param lepton The lepton whose  \f$ \eta \f$  value is to be checked.
        */
        bool CheckEta(const ALepton& lepton) const;


    private:

        /**
           @brief The  \f$ \eta- \f$ dependent resolution.
         */
        EtaDepResolution resolution_;

    };

} // namespace hitfit

#endif // #ifndef HitFit_LeptonTranslatorBase_h
