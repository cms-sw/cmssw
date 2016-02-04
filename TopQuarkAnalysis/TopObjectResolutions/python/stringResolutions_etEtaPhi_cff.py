import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.stringResolutionProvider_cfi import *

## electron resolutions
elecResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                        functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.01188^2 + (0.045/sqrt(et))^2 + (0.29/et)^2))'),
    eta = cms.string('sqrt(0.0004763^2 + (0.00059/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0.0014437/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.01256^2 + (0.0564/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003963^2 + (0.000848/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(8.8e-05^2 + (0.001193/sqrt(et))^2 + (0.0041/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.01129^2 + (0.0703/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.000348^2 + (0.00091/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(9.5e-05^2 + (0.001192/sqrt(et))^2 + (0.00437/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.01275^2 + (0.0621/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003152^2 + (0.00096/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(5.5e-05^2 + (0.00143/sqrt(et))^2 + (0.00293/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.01256^2 + (0.0678/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003111^2 + (0.00093/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(7.4e-05^2 + (0.001391/sqrt(et))^2 + (0.00326/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.01139^2 + (0.0729/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003167^2 + (0.00088/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.000114^2 + (0.001294/sqrt(et))^2 + (0.00392/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.01285^2 + (0.0599/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003251^2 + (0.00102/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(7.8e-05^2 + (0.001452/sqrt(et))^2 + (0.00304/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.01147^2 + (0.0784/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003363^2 + (0.001/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.000108^2 + (0.001513/sqrt(et))^2 + (0.00293/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.01374^2 + (0.0761/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.000324^2 + (0.00106/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.000127^2 + (0.001556/sqrt(et))^2 + (0.00294/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.01431^2 + (0.0754/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003081^2 + (0.001/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.000164^2 + (0.00149/sqrt(et))^2 + (0.00411/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.01196^2 + (0.1066/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003212^2 + (0.001/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.000111^2 + (0.001933/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.01613^2 + (0.1164/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003348^2 + (0.0011/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.000164^2 + (0.00195/sqrt(et))^2 + (0.0022/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0.0227^2 + (0.1091/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003474^2 + (0.00109/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.000191^2 + (0.00216/sqrt(et))^2 + (0.0026/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0.0158^2 + (0.1718/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003354^2 + (0.00102/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.000274^2 + (0.00208/sqrt(et))^2 + (0.0028/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.0176^2 + (0.1718/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.000332^2 + (0.00109/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.000253^2 + (0.002472/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0.0077^2 + (0.2288/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.000317^2 + (0.001049/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.000285^2 + (0.00255/sqrt(et))^2 + (0.003/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0.047^2 + (0.158/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003479^2 + (0/sqrt(et))^2 + (0.0036/et)^2)'),
    phi = cms.string('sqrt(0.000333^2 + (0.00277/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0^2 + (0.2/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.00038^2 + (0.00282/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(0.04019^2 + (0/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00033^2 + (0.0009/sqrt(et))^2 + (0.0019/et)^2)'),
    phi = cms.string('sqrt(0.000269^2 + (0.00324/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(0^2 + (0/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.000348^2 + (0.00096/sqrt(et))^2 + (0.0016/et)^2)'),
    phi = cms.string('sqrt(0.000271^2 + (0.00369/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(0.038^2 + (0.096/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0003786^2 + (0/sqrt(et))^2 + (0.00424/et)^2)'),
    phi = cms.string('sqrt(0.00028^2 + (0.0031/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(0.0382^2 + (0.076/sqrt(et))^2 + (0.28/et)^2))'),
    eta = cms.string('sqrt(0.000389^2 + (0.00106/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0.000401^2 + (0.0025/sqrt(et))^2 + (0.0114/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0.035^2 + (0.11/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.000486^2 + (0.0002/sqrt(et))^2 + (0.0052/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0.00432/sqrt(et))^2 + (0.0088/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(0.0354^2 + (0.123/sqrt(et))^2 + (0.1/et)^2))'),
    eta = cms.string('sqrt(0.000568^2 + (0/sqrt(et))^2 + (0.00734/et)^2)'),
    phi = cms.string('sqrt(0.000671^2 + (0/sqrt(et))^2 + (0.0158/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )

## muon resolutions
muonResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                        functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.100'),
    et  = cms.string('et * (0.00475 + 0.0002365 * et)'),
    eta = cms.string('sqrt(0.0004348^2 + (0.001063/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(6.28e-05^2 + (0/sqrt(et))^2 + (0.004545/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.100<=abs(eta) && abs(eta)<0.200'),
    et  = cms.string('et * (0.00509 + 0.0002298 * et)'),
    eta = cms.string('sqrt(0.0004348^2 + (0.001063/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(5.53e-05^2 + (0/sqrt(et))^2 + (0.004763/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.200<=abs(eta) && abs(eta)<0.300'),
    et  = cms.string('et * (0.005942 + 0.0002138 * et)'),
    eta = cms.string('sqrt(0.0003412^2 + (0.000857/sqrt(et))^2 + (0.00147/et)^2)'),
    phi = cms.string('sqrt(5.39e-05^2 + (0/sqrt(et))^2 + (0.004842/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.300<=abs(eta) && abs(eta)<0.400'),
    et  = cms.string('et * (0.006989 + 0.0002003 * et)'),
    eta = cms.string('sqrt(0.0003208^2 + (0.000604/sqrt(et))^2 + (0.00187/et)^2)'),
    phi = cms.string('sqrt(5.63e-05^2 + (0/sqrt(et))^2 + (0.00494/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.400<=abs(eta) && abs(eta)<0.500'),
    et  = cms.string('et * (0.007227 + 0.0001996 * et)'),
    eta = cms.string('sqrt(0.0002908^2 + (0.000733/sqrt(et))^2 + (0.00151/et)^2)'),
    phi = cms.string('sqrt(5.58e-05^2 + (0/sqrt(et))^2 + (0.00501/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.500<=abs(eta) && abs(eta)<0.600'),
    et  = cms.string('et * (0.007528 + 0.0001935 * et)'),
    eta = cms.string('sqrt(0.000289^2 + (0.00076/sqrt(et))^2 + (0.00154/et)^2)'),
    phi = cms.string('sqrt(5.65e-05^2 + (0/sqrt(et))^2 + (0.005082/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.600<=abs(eta) && abs(eta)<0.700'),
    et  = cms.string('et * (0.007909 + 0.0001863 * et)'),
    eta = cms.string('sqrt(0.000309^2 + (0.000667/sqrt(et))^2 + (0.00194/et)^2)'),
    phi = cms.string('sqrt(5.58e-05^2 + (0/sqrt(et))^2 + (0.005241/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.700<=abs(eta) && abs(eta)<0.800'),
    et  = cms.string('et * (0.008298 + 0.000185 * et)'),
    eta = cms.string('sqrt(0.0002887^2 + (0.000876/sqrt(et))^2 + (0.00179/et)^2)'),
    phi = cms.string('sqrt(5.97e-05^2 + (0/sqrt(et))^2 + (0.005085/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.800<=abs(eta) && abs(eta)<0.900'),
    et  = cms.string('et * (0.00918 + 0.0001911 * et)'),
    eta = cms.string('sqrt(0.0002956^2 + (0.000752/sqrt(et))^2 + (0.00208/et)^2)'),
    phi = cms.string('sqrt(5.9e-05^2 + (0/sqrt(et))^2 + (0.005506/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.900<=abs(eta) && abs(eta)<1.000'),
    et  = cms.string('et * (0.01096 + 0.0001899 * et)'),
    eta = cms.string('sqrt(0.0002734^2 + (0.000967/sqrt(et))^2 + (0.00134/et)^2)'),
    phi = cms.string('sqrt(7.48e-05^2 + (0/sqrt(et))^2 + (0.005443/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.000<=abs(eta) && abs(eta)<1.100'),
    et  = cms.string('et * (0.01262 + 0.0001614 * et)'),
    eta = cms.string('sqrt(0.0002831^2 + (0.000968/sqrt(et))^2 + (0.00166/et)^2)'),
    phi = cms.string('sqrt(7.81e-05^2 + (0/sqrt(et))^2 + (0.005585/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.100<=abs(eta) && abs(eta)<1.200'),
    et  = cms.string('et * (0.01379 + 0.0001618 * et)'),
    eta = cms.string('sqrt(0.000293^2 + (0.000942/sqrt(et))^2 + (0.002/et)^2)'),
    phi = cms.string('sqrt(8.19e-05^2 + (0/sqrt(et))^2 + (0.005921/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.200<=abs(eta) && abs(eta)<1.300'),
    et  = cms.string('et * (0.01485 + 0.0001574 * et)'),
    eta = cms.string('sqrt(0.0002907^2 + (0.000832/sqrt(et))^2 + (0.002/et)^2)'),
    phi = cms.string('sqrt(7.89e-05^2 + (0.00039/sqrt(et))^2 + (0.00593/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.300<=abs(eta) && abs(eta)<1.400'),
    et  = cms.string('et * (0.0152 + 0.0001719 * et)'),
    eta = cms.string('sqrt(0.0002937^2 + (0.000839/sqrt(et))^2 + (0.00232/et)^2)'),
    phi = cms.string('sqrt(5.9e-05^2 + (0.000724/sqrt(et))^2 + (0.005664/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.400<=abs(eta) && abs(eta)<1.500'),
    et  = cms.string('et * (0.01471 + 0.0001828 * et)'),
    eta = cms.string('sqrt(0.0002999^2 + (0.000864/sqrt(et))^2 + (0.00229/et)^2)'),
    phi = cms.string('sqrt(4.7e-05^2 + (0.000834/sqrt(et))^2 + (0.00527/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.500<=abs(eta) && abs(eta)<1.600'),
    et  = cms.string('et * (0.01337 + 0.0002375 * et)'),
    eta = cms.string('sqrt(0.0003035^2 + (0.000746/sqrt(et))^2 + (0.00258/et)^2)'),
    phi = cms.string('sqrt(8.16e-05^2 + (0.000757/sqrt(et))^2 + (0.00558/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.600<=abs(eta) && abs(eta)<1.700'),
    et  = cms.string('et * (0.01308 + 0.000285 * et)'),
    eta = cms.string('sqrt(0.0002967^2 + (0.000798/sqrt(et))^2 + (0.00263/et)^2)'),
    phi = cms.string('sqrt(6.2e-05^2 + (0.001025/sqrt(et))^2 + (0.00523/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.700<=abs(eta) && abs(eta)<1.800'),
    et  = cms.string('et * (0.01302 + 0.0003797 * et)'),
    eta = cms.string('sqrt(0.0003063^2 + (0.000776/sqrt(et))^2 + (0.00278/et)^2)'),
    phi = cms.string('sqrt(0.000107^2 + (0.001011/sqrt(et))^2 + (0.00554/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.800<=abs(eta) && abs(eta)<1.900'),
    et  = cms.string('et * (0.0139 + 0.000492 * et)'),
    eta = cms.string('sqrt(0.0003285^2 + (0.00077/sqrt(et))^2 + (0.00292/et)^2)'),
    phi = cms.string('sqrt(0.000119^2 + (0.001163/sqrt(et))^2 + (0.00519/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.900<=abs(eta) && abs(eta)<2.000'),
    et  = cms.string('et * (0.01507 + 0.000581 * et)'),
    eta = cms.string('sqrt(0.0003365^2 + (0.00084/sqrt(et))^2 + (0.00323/et)^2)'),
    phi = cms.string('sqrt(0.000193^2 + (0.00067/sqrt(et))^2 + (0.00613/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.000<=abs(eta) && abs(eta)<2.100'),
    et  = cms.string('et * (0.01711 + 0.000731 * et)'),
    eta = cms.string('sqrt(0.0003504^2 + (0.00078/sqrt(et))^2 + (0.00365/et)^2)'),
    phi = cms.string('sqrt(0.000217^2 + (0.00121/sqrt(et))^2 + (0.00558/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.100<=abs(eta) && abs(eta)<2.200'),
    et  = cms.string('et * (0.01973 + 0.000823 * et)'),
    eta = cms.string('sqrt(0.000381^2 + (0.00088/sqrt(et))^2 + (0.00369/et)^2)'),
    phi = cms.string('sqrt(0.000293^2 + (0.00082/sqrt(et))^2 + (0.00608/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.200<=abs(eta) && abs(eta)<2.300'),
    et  = cms.string('et * (0.02159 + 0.001052 * et)'),
    eta = cms.string('sqrt(0.00042^2 + (0.00097/sqrt(et))^2 + (0.00393/et)^2)'),
    phi = cms.string('sqrt(0.000304^2 + (0.00149/sqrt(et))^2 + (0.00549/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.300<=abs(eta) && abs(eta)<2.400'),
    et  = cms.string('et * (0.02155 + 0.001346 * et)'),
    eta = cms.string('sqrt(0.000403^2 + (0.00153/sqrt(et))^2 + (0.00403/et)^2)'),
    phi = cms.string('sqrt(0.000331^2 + (0.00183/sqrt(et))^2 + (0.00585/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )

## light jet resolutions (AK5 calo)
udscResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                        functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.087'),
    et  = cms.string('et * (sqrt(0.031^2 + (1.236/sqrt(et))^2 + (4.44/et)^2))'),
    eta = cms.string('sqrt(0.00836^2 + (1.4036/et)^2)'),
    phi = cms.string('sqrt(0.00858^2 + (2.475/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.087<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.0446^2 + (1.185/sqrt(et))^2 + (5.03/et)^2))'),
    eta = cms.string('sqrt(0.00792^2 + (1.4432/et)^2)'),
    phi = cms.string('sqrt(0.00734^2 + (2.547/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.0478^2 + (1.172/sqrt(et))^2 + (5.23/et)^2))'),
    eta = cms.string('sqrt(0.00807^2 + (1.4603/et)^2)'),
    phi = cms.string('sqrt(0.00912^2 + (2.502/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.0438^2 + (1.169/sqrt(et))^2 + (5.21/et)^2))'),
    eta = cms.string('sqrt(0.00755^2 + (1.4781/et)^2)'),
    phi = cms.string('sqrt(0.00742^2 + (2.513/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.0443^2 + (1.163/sqrt(et))^2 + (5.14/et)^2))'),
    eta = cms.string('sqrt(0.00772^2 + (1.5064/et)^2)'),
    phi = cms.string('sqrt(0.00828^2 + (2.529/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.0499^2 + (1.142/sqrt(et))^2 + (5.06/et)^2))'),
    eta = cms.string('sqrt(0.00793^2 + (1.4902/et)^2)'),
    phi = cms.string('sqrt(0.00676^2 + (2.534/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.0536^2 + (1.121/sqrt(et))^2 + (5.24/et)^2))'),
    eta = cms.string('sqrt(0.00803^2 + (1.4472/et)^2)'),
    phi = cms.string('sqrt(0.00659^2 + (2.498/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.0487^2 + (1.129/sqrt(et))^2 + (5.26/et)^2))'),
    eta = cms.string('sqrt(0.00831^2 + (1.4409/et)^2)'),
    phi = cms.string('sqrt(0.00812^2 + (2.465/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.0434^2 + (1.194/sqrt(et))^2 + (4.64/et)^2))'),
    eta = cms.string('sqrt(0.00844^2 + (1.4536/et)^2)'),
    phi = cms.string('sqrt(0.00706^2 + (2.504/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.0447^2 + (1.23/sqrt(et))^2 + (4.37/et)^2))'),
    eta = cms.string('sqrt(0.00777^2 + (1.5148/et)^2)'),
    phi = cms.string('sqrt(0.00688^2 + (2.535/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.0383^2 + (1.263/sqrt(et))^2 + (4.45/et)^2))'),
    eta = cms.string('sqrt(0.00753^2 + (1.5043/et)^2)'),
    phi = cms.string('sqrt(0.00698^2 + (2.512/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.0471^2 + (1.198/sqrt(et))^2 + (5.1/et)^2))'),
    eta = cms.string('sqrt(0.00756^2 + (1.5162/et)^2)'),
    phi = cms.string('sqrt(0.00731^2 + (2.519/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.0485^2 + (1.245/sqrt(et))^2 + (4.88/et)^2))'),
    eta = cms.string('sqrt(0.00737^2 + (1.5445/et)^2)'),
    phi = cms.string('sqrt(0.00755^2 + (2.526/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0.043^2 + (1.271/sqrt(et))^2 + (5/et)^2))'),
    eta = cms.string('sqrt(0.00779^2 + (1.56/et)^2)'),
    phi = cms.string('sqrt(0.00668^2 + (2.574/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0.0361^2 + (1.323/sqrt(et))^2 + (4.63/et)^2))'),
    eta = cms.string('sqrt(0.0084^2 + (1.622/et)^2)'),
    phi = cms.string('sqrt(0.0073^2 + (2.61/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.0449^2 + (1.319/sqrt(et))^2 + (5.24/et)^2))'),
    eta = cms.string('sqrt(0.01231^2 + (1.653/et)^2)'),
    phi = cms.string('sqrt(0.00773^2 + (2.646/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0^2 + (1.423/sqrt(et))^2 + (4.42/et)^2))'),
    eta = cms.string('sqrt(0.01187^2 + (1.668/et)^2)'),
    phi = cms.string('sqrt(0.00789^2 + (2.823/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.566'),
    et  = cms.string('et * (sqrt(0^2 + (1.341/sqrt(et))^2 + (5.48/et)^2))'),
    eta = cms.string('sqrt(0.01267^2 + (1.647/et)^2)'),
    phi = cms.string('sqrt(0.0084^2 + (2.813/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.566<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0^2 + (1.242/sqrt(et))^2 + (5.75/et)^2))'),
    eta = cms.string('sqrt(0.00941^2 + (1.584/et)^2)'),
    phi = cms.string('sqrt(0.00523^2 + (2.672/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0^2 + (1.1864/sqrt(et))^2 + (5.461/et)^2))'),
    eta = cms.string('sqrt(0.00891^2 + (1.647/et)^2)'),
    phi = cms.string('sqrt(0.00773^2 + (2.487/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(0.028^2 + (1.115/sqrt(et))^2 + (5.5/et)^2))'),
    eta = cms.string('sqrt(0.01023^2 + (1.649/et)^2)'),
    phi = cms.string('sqrt(0.00953^2 + (2.394/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(0.016^2 + (1.101/sqrt(et))^2 + (4.92/et)^2))'),
    eta = cms.string('sqrt(0.01151^2 + (1.535/et)^2)'),
    phi = cms.string('sqrt(0.01088^2 + (2.223/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(0.0396^2 + (0.915/sqrt(et))^2 + (5.11/et)^2))'),
    eta = cms.string('sqrt(0.00989^2 + (1.511/et)^2)'),
    phi = cms.string('sqrt(0.01146^2 + (2.071/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(0.032^2 + (0.907/sqrt(et))^2 + (4.44/et)^2))'),
    eta = cms.string('sqrt(0.01029^2 + (1.495/et)^2)'),
    phi = cms.string('sqrt(0.01175^2 + (1.939/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0.0347^2 + (0.875/sqrt(et))^2 + (3.96/et)^2))'),
    eta = cms.string('sqrt(0.01098^2 + (1.428/et)^2)'),
    phi = cms.string('sqrt(0.01079^2 + (1.827/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(0.0199^2 + (0.851/sqrt(et))^2 + (3.36/et)^2))'),
    eta = cms.string('sqrt(0.01314^2 + (1.43/et)^2)'),
    phi = cms.string('sqrt(0.01029^2 + (1.745/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.500<=abs(eta) && abs(eta)<3.000'),
    et  = cms.string('et * (sqrt(0.05^2 + (0.763/sqrt(et))^2 + (2.99/et)^2))'),
    eta = cms.string('sqrt(0.02238^2 + (1.612/et)^2)'),
    phi = cms.string('sqrt(0.01396^2 + (1.5799/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )

## light jet resolutions (AK5 particle flow)
udscResolutionPF = stringResolution.clone(parametrization = 'EtEtaPhi',
                                          functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.087'),
    et  = cms.string('et * (sqrt(0.0642^2 + (0.952/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00757^2 + (1.2578/et)^2)'),
    phi = cms.string('sqrt(0.01003^2 + (1.3972/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.087<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.069^2 + (0.9303/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0071^2 + (1.2661/et)^2)'),
    phi = cms.string('sqrt(0.01^2 + (1.3886/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.0675^2 + (0.938/sqrt(et))^2 + (0.8/et)^2))'),
    eta = cms.string('sqrt(0.00795^2 + (1.2713/et)^2)'),
    phi = cms.string('sqrt(0.01017^2 + (1.4/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.0645^2 + (0.9409/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00729^2 + (1.2924/et)^2)'),
    phi = cms.string('sqrt(0.01004^2 + (1.39/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.0616^2 + (0.9614/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00689^2 + (1.3078/et)^2)'),
    phi = cms.string('sqrt(0.01024^2 + (1.4013/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.0708^2 + (0.896/sqrt(et))^2 + (1.34/et)^2))'),
    eta = cms.string('sqrt(0.00716^2 + (1.3051/et)^2)'),
    phi = cms.string('sqrt(0.00976^2 + (1.4023/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.0647^2 + (0.9395/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00783^2 + (1.2687/et)^2)'),
    phi = cms.string('sqrt(0.00997^2 + (1.3834/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.0626^2 + (0.9445/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00782^2 + (1.2664/et)^2)'),
    phi = cms.string('sqrt(0.00952^2 + (1.4145/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.0642^2 + (0.9575/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00768^2 + (1.2863/et)^2)'),
    phi = cms.string('sqrt(0.0098^2 + (1.4062/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.0625^2 + (0.9851/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0071^2 + (1.3159/et)^2)'),
    phi = cms.string('sqrt(0.01023^2 + (1.4147/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.0617^2 + (1.0112/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00865^2 + (1.2837/et)^2)'),
    phi = cms.string('sqrt(0.01041^2 + (1.4286/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.0647^2 + (1.026/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0082^2 + (1.3122/et)^2)'),
    phi = cms.string('sqrt(0.01049^2 + (1.4245/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.0636^2 + (1.0591/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00828^2 + (1.3265/et)^2)'),
    phi = cms.string('sqrt(0.01083^2 + (1.4504/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0.0661^2 + (1.0793/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00807^2 + (1.3559/et)^2)'),
    phi = cms.string('sqrt(0.01091^2 + (1.487/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0.0614^2 + (1.1195/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.01007^2 + (1.3581/et)^2)'),
    phi = cms.string('sqrt(0.01145^2 + (1.5019/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.0654^2 + (1.165/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.014^2 + (1.327/et)^2)'),
    phi = cms.string('sqrt(0.01387^2 + (1.529/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0.0575^2 + (1.205/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.01072^2 + (1.348/et)^2)'),
    phi = cms.string('sqrt(0.01462^2 + (1.58/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.566'),
    et  = cms.string('et * (sqrt(0.0469^2 + (1.19/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00992^2 + (1.395/et)^2)'),
    phi = cms.string('sqrt(0.01256^2 + (1.584/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.566<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0^2 + (1.1632/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00975^2 + (1.396/et)^2)'),
    phi = cms.string('sqrt(0.01066^2 + (1.577/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0^2 + (1.1109/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00967^2 + (1.365/et)^2)'),
    phi = cms.string('sqrt(0.01087^2 + (1.521/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(0^2 + (1.0841/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0093^2 + (1.405/et)^2)'),
    phi = cms.string('sqrt(0.01066^2 + (1.505/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(0^2 + (1.0288/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.01057^2 + (1.365/et)^2)'),
    phi = cms.string('sqrt(0.01141^2 + (1.456/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(0^2 + (0.9821/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00992^2 + (1.329/et)^2)'),
    phi = cms.string('sqrt(0.01042^2 + (1.468/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(0^2 + (0.9441/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00938^2 + (1.327/et)^2)'),
    phi = cms.string('sqrt(0.01119^2 + (1.45/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0^2 + (0.9134/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00973^2 + (1.312/et)^2)'),
    phi = cms.string('sqrt(0.01128^2 + (1.413/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(0^2 + (0.8322/sqrt(et))^2 + (2.0069/et)^2))'),
    eta = cms.string('sqrt(0.01161^2 + (1.423/et)^2)'),
    phi = cms.string('sqrt(0.01256^2 + (1.471/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.500<=abs(eta) && abs(eta)<3.000'),
    et  = cms.string('et * (sqrt(0.0526^2 + (0.774/sqrt(et))^2 + (2.39/et)^2))'),
    eta = cms.string('sqrt(0^2 + (1.4/et)^2)'),
    phi = cms.string('sqrt(0.02829^2 + (1.498/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )

## b jet resolutions (AK5 calo)
bjetResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                        functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.087'),
    et  = cms.string('et * (sqrt(0.0901^2 + (1.035/sqrt(et))^2 + (6.2/et)^2))'),
    eta = cms.string('sqrt(0.00516^2 + (1.683/et)^2)'),
    phi = cms.string('sqrt(0.0024^2 + (3.159/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.087<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.0715^2 + (1.277/sqrt(et))^2 + (4.77/et)^2))'),
    eta = cms.string('sqrt(0.00438^2 + (1.72/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.179/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.0812^2 + (1.192/sqrt(et))^2 + (5.35/et)^2))'),
    eta = cms.string('sqrt(0.00517^2 + (1.71/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.136/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.0713^2 + (1.257/sqrt(et))^2 + (4.75/et)^2))'),
    eta = cms.string('sqrt(0.00474^2 + (1.732/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.166/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.0835^2 + (1.158/sqrt(et))^2 + (5.08/et)^2))'),
    eta = cms.string('sqrt(0.0047^2 + (1.744/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.15/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.0638^2 + (1.298/sqrt(et))^2 + (4.24/et)^2))'),
    eta = cms.string('sqrt(0.00404^2 + (1.793/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.152/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.0676^2 + (1.257/sqrt(et))^2 + (4.48/et)^2))'),
    eta = cms.string('sqrt(0.00533^2 + (1.747/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.112/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.0723^2 + (1.185/sqrt(et))^2 + (5.28/et)^2))'),
    eta = cms.string('sqrt(0.00511^2 + (1.745/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.173/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.0661^2 + (1.292/sqrt(et))^2 + (4.02/et)^2))'),
    eta = cms.string('sqrt(0.00623^2 + (1.724/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.127/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.0773^2 + (1.249/sqrt(et))^2 + (4.12/et)^2))'),
    eta = cms.string('sqrt(0.00522^2 + (1.796/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.123/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.082^2 + (1.18/sqrt(et))^2 + (5.24/et)^2))'),
    eta = cms.string('sqrt(0.00564^2 + (1.772/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.125/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.0703^2 + (1.322/sqrt(et))^2 + (3.81/et)^2))'),
    eta = cms.string('sqrt(0.00337^2 + (1.832/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.143/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.0578^2 + (1.39/sqrt(et))^2 + (3.69/et)^2))'),
    eta = cms.string('sqrt(0.00323^2 + (1.85/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.175/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0.039^2 + (1.508/sqrt(et))^2 + (1.3/et)^2))'),
    eta = cms.string('sqrt(0.00309^2 + (1.916/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.182/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0.0722^2 + (1.347/sqrt(et))^2 + (4.38/et)^2))'),
    eta = cms.string('sqrt(0.00618^2 + (1.933/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.223/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.0807^2 + (1.35/sqrt(et))^2 + (4.38/et)^2))'),
    eta = cms.string('sqrt(0.00889^2 + (1.961/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.331/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0.066^2 + (1.457/sqrt(et))^2 + (3.54/et)^2))'),
    eta = cms.string('sqrt(0.00747^2 + (2.079/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.484/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.566'),
    et  = cms.string('et * (sqrt(0.0685^2 + (1.42/sqrt(et))^2 + (3.67/et)^2))'),
    eta = cms.string('sqrt(0.01005^2 + (2.045/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.583/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.566<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0^2 + (1.561/sqrt(et))^2 + (1.59/et)^2))'),
    eta = cms.string('sqrt(0.0036^2 + (2.024/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.339/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0.0736^2 + (1.264/sqrt(et))^2 + (4.34/et)^2))'),
    eta = cms.string('sqrt(0.0038^2 + (2.042/et)^2)'),
    phi = cms.string('sqrt(0^2 + (3.11/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(0.0648^2 + (1.234/sqrt(et))^2 + (4.5/et)^2))'),
    eta = cms.string('sqrt(0.0037^2 + (2.109/et)^2)'),
    phi = cms.string('sqrt(0^2 + (2.923/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(0.049^2 + (1.243/sqrt(et))^2 + (3.83/et)^2))'),
    eta = cms.string('sqrt(0.0054^2 + (1.944/et)^2)'),
    phi = cms.string('sqrt(0^2 + (2.716/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(0.0661^2 + (1.081/sqrt(et))^2 + (4.16/et)^2))'),
    eta = cms.string('sqrt(0.0033^2 + (1.871/et)^2)'),
    phi = cms.string('sqrt(0^2 + (2.548/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(0.0644^2 + (1.02/sqrt(et))^2 + (3.89/et)^2))'),
    eta = cms.string('sqrt(0^2 + (1.803/et)^2)'),
    phi = cms.string('sqrt(0^2 + (2.365/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0.0892^2 + (0.779/sqrt(et))^2 + (4.28/et)^2))'),
    eta = cms.string('sqrt(0^2 + (1.682/et)^2)'),
    phi = cms.string('sqrt(0^2 + (2.148/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(0.0498^2 + (0.912/sqrt(et))^2 + (3.53/et)^2))'),
    eta = cms.string('sqrt(0^2 + (1.732/et)^2)'),
    phi = cms.string('sqrt(0^2 + (2.019/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.500<=abs(eta) && abs(eta)<3.000'),
    et  = cms.string('et * (sqrt(0.0605^2 + (0.861/sqrt(et))^2 + (3.08/et)^2))'),
    eta = cms.string('sqrt(0^2 + (2.032/et)^2)'),
    phi = cms.string('sqrt(0^2 + (1.805/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )

## b jet resolutions (AK5 particle flow)
bjetResolutionPF = stringResolution.clone(parametrization = 'EtEtaPhi',
                                          functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.087'),
    et  = cms.string('et * (sqrt(0.0876^2 + (0.93/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00658^2 + (1.3618/et)^2)'),
    phi = cms.string('sqrt(0.00914^2 + (1.5326/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.087<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.0892^2 + (0.905/sqrt(et))^2 + (1.6/et)^2))'),
    eta = cms.string('sqrt(0.00578^2 + (1.3927/et)^2)'),
    phi = cms.string('sqrt(0.0091^2 + (1.5446/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.0856^2 + (0.946/sqrt(et))^2 + (0.2/et)^2))'),
    eta = cms.string('sqrt(0.0063^2 + (1.3873/et)^2)'),
    phi = cms.string('sqrt(0.00892^2 + (1.5446/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.0838^2 + (0.911/sqrt(et))^2 + (1.76/et)^2))'),
    eta = cms.string('sqrt(0.00587^2 + (1.4045/et)^2)'),
    phi = cms.string('sqrt(0.00889^2 + (1.5435/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.0792^2 + (0.961/sqrt(et))^2 + (0.5/et)^2))'),
    eta = cms.string('sqrt(0.00562^2 + (1.4079/et)^2)'),
    phi = cms.string('sqrt(0.00883^2 + (1.54/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.0791^2 + (0.955/sqrt(et))^2 + (0.9/et)^2))'),
    eta = cms.string('sqrt(0.00602^2 + (1.4112/et)^2)'),
    phi = cms.string('sqrt(0.00846^2 + (1.5708/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.0748^2 + (0.98/sqrt(et))^2 + (0.4/et)^2))'),
    eta = cms.string('sqrt(0.00616^2 + (1.4132/et)^2)'),
    phi = cms.string('sqrt(0.00836^2 + (1.5673/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.0753^2 + (0.969/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00664^2 + (1.3955/et)^2)'),
    phi = cms.string('sqrt(0.00826^2 + (1.588/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.0831^2 + (0.947/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00591^2 + (1.4045/et)^2)'),
    phi = cms.string('sqrt(0.00886^2 + (1.561/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.0781^2 + (0.961/sqrt(et))^2 + (1.16/et)^2))'),
    eta = cms.string('sqrt(0.00683^2 + (1.3992/et)^2)'),
    phi = cms.string('sqrt(0.00811^2 + (1.583/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.078^2 + (1.004/sqrt(et))^2 + (0.7/et)^2))'),
    eta = cms.string('sqrt(0.00695^2 + (1.425/et)^2)'),
    phi = cms.string('sqrt(0.00865^2 + (1.582/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.0787^2 + (1.025/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00618^2 + (1.452/et)^2)'),
    phi = cms.string('sqrt(0.00866^2 + (1.619/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.081^2 + (1.035/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00675^2 + (1.459/et)^2)'),
    phi = cms.string('sqrt(0.0087^2 + (1.613/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0.0853^2 + (1.048/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00738^2 + (1.489/et)^2)'),
    phi = cms.string('sqrt(0.00942^2 + (1.644/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0.0875^2 + (1.04/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00873^2 + (1.49/et)^2)'),
    phi = cms.string('sqrt(0.0094^2 + (1.68/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.0906^2 + (1.081/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.01038^2 + (1.495/et)^2)'),
    phi = cms.string('sqrt(0.01143^2 + (1.701/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0.0919^2 + (1.096/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00822^2 + (1.537/et)^2)'),
    phi = cms.string('sqrt(0.011^2 + (1.785/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.566'),
    et  = cms.string('et * (sqrt(0.0825^2 + (1.124/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00871^2 + (1.537/et)^2)'),
    phi = cms.string('sqrt(0.01065^2 + (1.786/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.566<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0.0504^2 + (1.174/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00644^2 + (1.575/et)^2)'),
    phi = cms.string('sqrt(0.00833^2 + (1.77/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0.0432^2 + (1.122/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00791^2 + (1.545/et)^2)'),
    phi = cms.string('sqrt(0.00841^2 + (1.712/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(0.0244^2 + (1.113/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00574^2 + (1.578/et)^2)'),
    phi = cms.string('sqrt(0.00697^2 + (1.702/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(0.0303^2 + (1.067/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00727^2 + (1.552/et)^2)'),
    phi = cms.string('sqrt(0.00675^2 + (1.672/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(0.0193^2 + (1.052/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00823^2 + (1.494/et)^2)'),
    phi = cms.string('sqrt(0.00676^2 + (1.609/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(0.0372^2 + (0.985/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.0075^2 + (1.484/et)^2)'),
    phi = cms.string('sqrt(0.00773^2 + (1.586/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0.0292^2 + (0.967/sqrt(et))^2 + (0/et)^2))'),
    eta = cms.string('sqrt(0.00629^2 + (1.484/et)^2)'),
    phi = cms.string('sqrt(0.00676^2 + (1.631/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(0.014^2 + (0.963/sqrt(et))^2 + (1.24/et)^2))'),
    eta = cms.string('sqrt(0^2 + (1.775/et)^2)'),
    phi = cms.string('sqrt(0.00652^2 + (1.697/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.500<=abs(eta) && abs(eta)<3.000'),
    et  = cms.string('et * (sqrt(0.0653^2 + (0.889/sqrt(et))^2 + (2.05/et)^2))'),
    eta = cms.string('sqrt(0.01595^2 + (2.003/et)^2)'),
    phi = cms.string('sqrt(0.01746^2 + (1.9/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )

## MET resolutions (calo)
metResolution  = stringResolution.clone(parametrization = 'EtEtaPhi',
                                        functions = cms.VPSet(
    cms.PSet(
    et  = cms.string('et * (sqrt(0^2 + (1.462/sqrt(et))^2 + (18.19/et)^2))'),
    eta = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0^2 + (1.237/sqrt(et))^2 + (18.702/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )

## MET resolutions (particle flow)
metResolutionPF  = stringResolution.clone(parametrization = 'EtEtaPhi',
                                          functions = cms.VPSet(
    cms.PSet(
    et  = cms.string('et * (sqrt(0.05469^2 + (0/sqrt(et))^2 + (10.549/et)^2))'),
    eta = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (0/et)^2)'),
    phi = cms.string('sqrt(0^2 + (0.164/sqrt(et))^2 + (11.068/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )
