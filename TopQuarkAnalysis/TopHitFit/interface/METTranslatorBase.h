//
//     $Id: METTranslatorBase.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File   : interface/METTranslatorBase.h
// Author : Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
// Purpose: Template class of function object to translate missing transverse
//          energy physics object to HitFit's Fourvec object.
//

/**
    @file METTranslatorBase.h

    @brief Template class of function object to translate missing transverse
    energy object to HitFit's Fourvec object.

    @author Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Creation date:
    Nov-2009

    @par Terms of Usage:
    With consent from the original author (Haryo Sumowidagdo).
 */

#ifndef HitFit_METTranslatorBase_h
#define HitFit_METTranslatorBase_h

#include "TopQuarkAnalysis/TopHitFit/interface/EtaDepResolution.h"
#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"

namespace hitfit{


    /**
       @class METTranslatorBase.

       @brief Template class of function object to translate missing
       transverse energy physics object to HitFit's Fourvec object.  Users
       need to write an implementation of a template specialization of this
       class for their missing transverse energy physics object class,
       Then users combine this header file and their implementation for their
       analysis code.  With this approach, it is possible to use HitFit for
       different missing transverse energy physics object class
       indifferent experiments.

       @param AMet The typename of the missing transverse energy physics
       object class be translated into HitFit's Fourvec.

     */
    template <class AMet>
    class METTranslatorBase {

    public:

        /**
           @brief Default constructor.
         */
        METTranslatorBase();

        /**
           @brief Constructor, instantiate a METTranslatorBase object
           using the name of and input file in std::string format.

           @param ifile The path of the input file.
         */
        METTranslatorBase(const std::string& ifile);

        /**
           @brief Destructor.
         */
        ~METTranslatorBase();

        /**
           @brief Convert a missing transverse energy object of type AMet
           into HitFit four-momentum object of type Fourvec.

           @param met The missing transverse energy object to be translated.

           @param useObjEmbRes Boolean parameter to indicate if the
           user would like to use the resolution embedded in the object,
           and not the resolution read when instantiating the class.
         */
        Fourvec operator() (const AMet& met,
                            bool useObjEmbRes = false);

        /**
           @brief Return the  \f$ k_{T} \f$  resolution corresponding to
           an instance of missing transverse energy object.

           @param met The missing transverse energy object whose
           resolution is wished to be known.

           @param useObjEmbRes Boolean parameter to indicate if the
           user would like to use the resolution embedded in the object,
           and not the resolution read when instantiating the class.
         */
        Resolution KtResolution(const AMet& met,
                                bool useObjEmbRes = false) const;

        /**
           @brief Alias for KtResolution(AMet& met)

           @param met The missing transverse energy object whose
           resolution is wished to be known.

           @param useObjEmbRes Boolean parameter to indicate if the
           user would like to use the resolution embedded in the object,
           and not the resolution read when instantiating the class.
         */
        Resolution METResolution(const AMet& met,
                                 bool useObjEmbRes = false) const;


    private:

        /**
           @brief The resolution.
         */
        Resolution resolution_;

    };

} // namespace hitfit

#endif // #ifndef HitFit_METTranslatorBase_h
