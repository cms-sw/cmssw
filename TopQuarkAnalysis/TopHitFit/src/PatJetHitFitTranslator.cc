/**
    @file PatJetHitFitTranslator.cc

    @brief Specialization of template class JetTranslatorBase in the
    package HitFit

    @author Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

    @par Created
    Sat Jun 27 17:49:21 2009 UTC

    @version $Id: PatJetHitFitTranslator.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
 */


#include <TopQuarkAnalysis/TopHitFit/interface/JetTranslatorBase.h>
#include <DataFormats/PatCandidates/interface/Jet.h>

namespace hitfit {


template<>
JetTranslatorBase<pat::Jet>::JetTranslatorBase()
{

    std::string CMSSW_BASE(getenv("CMSSW_BASE"));
    std::string resolution_filename = CMSSW_BASE +
        std::string("/src/TopQuarkAnalysis/PatHitFit/data/exampleJetResolution.txt");
    udscResolution_ = EtaDepResolution(resolution_filename);
    bResolution_    = EtaDepResolution(resolution_filename);
    jetCorrectionLevel_ = "L7Parton";

} // JetTranslatorBase<pat::Jet>::JetTranslatorBase()


template<>
JetTranslatorBase<pat::Jet>::JetTranslatorBase(const std::string& udscFile,
                                               const std::string& bFile)
{

    std::string CMSSW_BASE(getenv("CMSSW_BASE"));
    std::string udscResolution_filename;
    std::string bResolution_filename;

    if (udscFile.empty()) {
        udscResolution_filename = CMSSW_BASE +
        std::string("/src/TopQuarkAnalysis/PatHitFit/data/exampleJetResolution.txt");
    } else {
        udscResolution_filename = udscFile;
    }

    if (bFile.empty()) {
        bResolution_filename = CMSSW_BASE +
        std::string("/src/TopQuarkAnalysis/PatHitFit/data/exampleJetResolution.txt");
    } else {
        bResolution_filename = bFile;
    }

    udscResolution_ = EtaDepResolution(udscResolution_filename);
    bResolution_    = EtaDepResolution(bResolution_filename);
    jetCorrectionLevel_ = "L7Parton";

} // JetTranslatorBase<pat::Jet>::JetTranslatorBase(const std::string& ifile)


template<>
JetTranslatorBase<pat::Jet>::JetTranslatorBase(const std::string& udscFile,
                                               const std::string& bFile,
                                               const std::string& jetCorrectionLevel)
{

    std::string CMSSW_BASE(getenv("CMSSW_BASE"));
    std::string udscResolution_filename;
    std::string bResolution_filename;

    if (udscFile.empty()) {
        udscResolution_filename = CMSSW_BASE +
        std::string("/src/TopQuarkAnalysis/TopHitFit/data/resolution/tqafUdscJetResolution.txt");
    } else {
        udscResolution_filename = udscFile;
    }

    if (bFile.empty()) {
        bResolution_filename = CMSSW_BASE +
        std::string("/src/TopQuarkAnalysis/TopHitFit/data/resolution/tqafBJetResolution.txt");
    } else {
        bResolution_filename = bFile;
    }

    udscResolution_ = EtaDepResolution(udscResolution_filename);
    bResolution_    = EtaDepResolution(bResolution_filename);
    jetCorrectionLevel_ = jetCorrectionLevel;

} // JetTranslatorBase<pat::Jet>::JetTranslatorBase(const std::string& ifile)


template<>
JetTranslatorBase<pat::Jet>::~JetTranslatorBase()
{
} // JetTranslatorBase<pat::Jet>::~JetTranslatorBase()


template<>
Lepjets_Event_Jet
JetTranslatorBase<pat::Jet>::operator()(const pat::Jet& jet,
                                        int type /*= hitfit::unknown_label */,
                                        bool useObjEmbRes /* = false */)
{

    Fourvec p;

    double            jet_eta        = jet.eta();

    if (jet.isCaloJet()) {
        jet_eta = ((reco::CaloJet*) jet.originalObject())->detectorP4().eta();
    }
    if (jet.isPFJet()) {
        // do nothing at the moment!
    }

    Vector_Resolution jet_resolution;

    if (type == hitfit::hadb_label || type == hitfit::lepb_label || type == hitfit::higgs_label) {
        jet_resolution = bResolution_.GetResolution(jet_eta);
        pat::Jet bPartonCorrJet(jet.correctedJet(jetCorrectionLevel_,"BOTTOM"));
        p = Fourvec(bPartonCorrJet.px(),bPartonCorrJet.py(),bPartonCorrJet.pz(),bPartonCorrJet.energy());

    } else {
        jet_resolution = udscResolution_.GetResolution(jet_eta);
        pat::Jet udsPartonCorrJet(jet.correctedJet(jetCorrectionLevel_,"UDS"));
        p = Fourvec(udsPartonCorrJet.px(),udsPartonCorrJet.py(),udsPartonCorrJet.pz(),udsPartonCorrJet.energy());
    }



    Lepjets_Event_Jet retjet(p,
                             type,
                             jet_resolution);
    return retjet;

} // Lepjets_Event_Jet JetTranslatorBase<pat::Jet>::operator()(const pat::Jet& j,int type)


template<>
const EtaDepResolution&
JetTranslatorBase<pat::Jet>::udscResolution() const
{
    return udscResolution_;
}


template<>
const EtaDepResolution&
JetTranslatorBase<pat::Jet>::bResolution() const
{
    return bResolution_;
}


template<>
bool
JetTranslatorBase<pat::Jet>::CheckEta(const pat::Jet& jet) const
{
    double            jet_eta        = jet.eta();

    if (jet.isCaloJet()) {
        jet_eta = ((reco::CaloJet*) jet.originalObject())->detectorP4().eta();
    }
    if (jet.isPFJet()) {
        // do nothing at the moment!
    }
    return bResolution_.CheckEta(jet_eta) && udscResolution_.CheckEta(jet_eta);
}


    //

} // namespace hitfit
