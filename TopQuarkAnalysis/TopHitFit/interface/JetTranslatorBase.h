//
//     $Id: JetTranslatorBase.h,v 1.3 2011/07/11 07:59:12 mseidel Exp $
//

/**
    @file JetTranslatorBase.h

    @brief Template class of function object to translate jet physics
    object to HitFit's Lepjets_Event_Jet object.

    @author Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Creation date:
    Nov-2009

    @par Terms of Usage:
    With consent from the original author (Haryo Sumowidagdo).
 */

#ifndef HitFit_JetTranslatorBase_h
#define HitFit_JetTranslatorBase_h

#include "TopQuarkAnalysis/TopHitFit/interface/EtaDepResolution.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event_Jet.h"
#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"

namespace hitfit{


    /**
       @class JetTranslatorBase.

       @brief Template class of function object to translate jet physics
       object to HitFit's Lepjets_Event_Jet object.    Users need to write an
       implementation of a template specialization of this class for their jet
       physics object class.  Then users combine this header file and their
       implementation for their analysis code.  With this approach, it is
       possible to use HitFit for different jet physics object class in
       different experiments.

       @param AJet The typename of the jet physics object class to
       be translated into HitFit's Lepjets_Event_Jet.

     */
    template <class AJet>
    class JetTranslatorBase {

    public:

        /**
           @brief Default constructor.
         */
        JetTranslatorBase();

        /**
           @brief Constructor, instantiate a JetTranslatorBase object
           using the names of input files in std::string format.

           @param udscFile The path of the input file containing
           resolution for \f$udsc\f$ jets.

           @param bFile The path of the input file containing
           resolution for \f$b\f$ jets.

         */
        JetTranslatorBase(const std::string& udscFile,
                          const std::string& bFile);
        
        /**
           @brief Constructor, instantiate a JetTranslatorBase object
           using the names of input files in std::string format.

           @param udscFile The path of the input file containing
           resolution for \f$udsc\f$ jets.

           @param bFile The path of the input file containing
           resolution for \f$b\f$ jets.
           
           @param jetCorrectionLevel The jet correction level.
           
           @param jes The jet energy scale.
           
           @param jesB The b-jet energy scale.

         */
        JetTranslatorBase(const std::string& udscFile,
                          const std::string& bFile,
                          const std::string& jetCorrectionLevel,
                          double jes,
                          double jesB);

        /**
           @brief Destructor.
         */
        ~JetTranslatorBase();

        /**
           @brief Convert a jet physics object of type AJet into
           HitFit jet physics object of type Lepjets_Event_Jet.
           This operator must be able to apply the appropriate jet
           energy correction in accord with the type of the jet.

           @param jet The jet physics object to be translated.

           @param type The typecode of the jet to be translated
           (leptonic b, hadronic b, or hadronic W).

           @param useObjEmbRes Boolean parameter to indicate if the
           user would like to use the resolution embedded in the object,
           and not the resolution read when instantiating the class.
         */
        Lepjets_Event_Jet operator()(const AJet& jet,
                                     int type = hitfit::unknown_label,
                                     bool useObjEmbRes = false);

        /**
           @brief Return the  \f$ \eta- \f$ dependent resolution for \f$udsc\f$
           jets.
         */
        const EtaDepResolution& udscResolution() const;

        /**
           @brief Return the  \f$ \eta- \f$ dependent resolution for \f$b\f$
           jets.
         */
        const EtaDepResolution& bResolution() const;

        /**
           @brief Check if a jet has  \f$ \eta \f$  value which is within the
           valid  \f$ \eta \f$  range of the resolution.

           @param jet The jet whose  \f$ \eta \f$  value is to be checked.
         */
        bool CheckEta(const AJet& jet) const;


    private:

        /**
           @brief The  \f$ \eta- \f$ dependent resolution for $udsc$ jets.
         */
        EtaDepResolution udscResolution_;

        /**
           @brief The  \f$ \eta- \f$ dependent resolution for $b$ jets.
         */
        EtaDepResolution bResolution_;
        
        /**
           @brief The jet correction level.
         */
        std::string jetCorrectionLevel_;
        
        /**
           @brief The jet energy scale.
         */
        double jes_;
        
        /**
           @brief The b-jet energy scale.
         */
        double jesB_;

    };

} // namespace hitfit

#endif // #ifndef HitFit_JetTranslatorBase_h
