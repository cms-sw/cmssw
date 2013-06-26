//
//     $Id: PatMETHitFitTranslator.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//

/**
    @file PatMETHitFitTranslator.cc

    @brief Specialization of template class METTranslatorBase in the
    package HitFit for pat::MET

    @author Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Created
    Sat Jun 27 17:49:32 2009 UTC

    @version $Id: PatMETHitFitTranslator.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
 */


#include "TopQuarkAnalysis/TopHitFit/interface/METTranslatorBase.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include <cmath>

namespace hitfit {


template<>
METTranslatorBase<pat::MET>::METTranslatorBase()
{
    resolution_ = Resolution(std::string("0,0,12"));
} // METTranslatorBase<pat::MET>::METTranslatorBase()


template<>
METTranslatorBase<pat::MET>::METTranslatorBase(const std::string& ifile)
{
    const Defaults_Text defs(ifile);
    std::string resolution_string(defs.get_string("met_resolution"));
    resolution_ = Resolution(resolution_string);

} // METTranslatorBase<pat::MET>::METTranslatorBase(const std::string& ifile)


template<>
METTranslatorBase<pat::MET>::~METTranslatorBase()
{
} // METTranslatorBase<pat::MET>::~METTranslatorBase()


template<>
Fourvec
METTranslatorBase<pat::MET>::operator()(const pat::MET& m,
                                        bool useObjEmbRes /* = false */)
{
    double px = m.px();
    double py = m.py();

    return Fourvec (px,py,0.0,sqrt(px*px + py*py));

} // Fourvec METTranslatorBase<pat::MET>::operator()(const pat::MET& m)



template<>
Resolution
METTranslatorBase<pat::MET>::KtResolution(const pat::MET& m,
                                          bool useObjEmbRes /* = false */) const
{
    return resolution_;
} // Resolution METTranslatorBase<pat::MET>::KtResolution(const pat::MET& m)



template<>
Resolution
METTranslatorBase<pat::MET>::METResolution(const pat::MET& m,
                                           bool useObjEmbRes /* = false */) const
{
    return KtResolution(m,useObjEmbRes);
} // Resolution METTranslatorBase<pat::MET>::METResolution(const pat::MET& m)


} // namespace hitfit
