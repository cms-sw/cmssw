import FWCore.ParameterSet.Config as cms

from Validation.CTPPS.simu_config.year_2017_cff import profile
profile_2017_preTS2=profile.clone()

#LHCInfo
profile_2017_preTS2.ctppsLHCInfo.xangleBetaStarHistogramObject="2017_preTS2/h2_betaStar_vs_xangle"

# alignment
alignmentFile = "Validation/CTPPS/alignment/2017_preTS2.xml"
profile_2017_preTS2.ctppsRPAlignmentCorrectionsDataXML.MisalignedFiles = [alignmentFile]
profile_2017_preTS2.ctppsRPAlignmentCorrectionsDataXML.RealFiles = [alignmentFile]

# aperture cuts
profile_2017_preTS2.ctppsDirectSimuData.useEmpiricalApertures = True

profile_2017_preTS2.ctppsDirectSimuData.empiricalAperture45="-(8.71198E-07*[xangle]-0.000134726)+(([xi]<(0.000264704*[xangle]+0.081951))*-(4.32065E-05*[xangle]-0.0130746)+([xi]>=(0.000264704*[xangle]+0.081951))*-(0.000183472*[xangle]-0.0395241))*([xi]-(0.000264704*[xangle]+0.081951))"
profile_2017_preTS2.ctppsDirectSimuData.empiricalAperture56="3.43116E-05+(([xi]<(0.000626936*[xangle]+0.061324))*0.00654394+([xi]>=(0.000626936*[xangle]+0.061324))*-(0.000145164*[xangle]-0.0272919))*([xi]-(0.000626936*[xangle]+0.061324))"

# timing resolution
profile_2017_preTS2.ctppsDirectSimuData.timeResolutionDiamonds45 = "2*(-0.10784+0.105194*x-0.0182611*x^2+0.00134731*x^3-3.58212E-05*x^4)"
#pol6 with higher accuracy
#ctppsDirectProtonSimulation.timeResolutionDiamonds45 = "-0.516674+0.392549*x-0.0962956*x^2+0.0117729*x^3-0.000759737*x^4+2.47004E-05*x^5-3.19432E-07*x^6"
profile_2017_preTS2.ctppsDirectSimuData.timeResolutionDiamonds56 = "2*(0.00735552+0.0272707*x-0.00247151*x^2+8.62788E-05*x^3-7.99605E-07*x^4)"
