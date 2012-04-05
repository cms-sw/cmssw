import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.stringResolutionProvider_cfi import *

print "*** Including object resolutions derived from Summer11 MC for:"
print "*** - electrons   - muons   - udscJetsPF     - bJetsPF     - pfMET"
print "*** Please make sure that you are really using resolutions that are suited for the objects in your analysis!"

udscResolutionPF = stringResolution.clone(parametrization = 'EtEtaPhi',
                                          functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.087'),
    et  = cms.string('et * (sqrt(0.06^2 + (1.0023/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.00901^2 + (1.5284/et)^2)'),
    phi  = cms.string('sqrt(0.0104^2 + (1.6004/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.087<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.0633^2 + (0.964/sqrt(et))^2 + (1.49/et)^2))'),
    eta  = cms.string('sqrt(0.00927^2 + (1.486/et)^2)'),
    phi  = cms.string('sqrt(0.01113^2 + (1.5354/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.0595^2 + (0.973/sqrt(et))^2 + (1.52/et)^2))'),
    eta  = cms.string('sqrt(0.00958^2 + (1.4794/et)^2)'),
    phi  = cms.string('sqrt(0.01093^2 + (1.5387/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.058^2 + (0.991/sqrt(et))^2 + (1.3/et)^2))'),
    eta  = cms.string('sqrt(0.00884^2 + (1.5269/et)^2)'),
    phi  = cms.string('sqrt(0.01107^2 + (1.5398/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.0597^2 + (0.96/sqrt(et))^2 + (1.82/et)^2))'),
    eta  = cms.string('sqrt(0.00915^2 + (1.5229/et)^2)'),
    phi  = cms.string('sqrt(0.01079^2 + (1.557/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.0582^2 + (0.977/sqrt(et))^2 + (1.45/et)^2))'),
    eta  = cms.string('sqrt(0.00936^2 + (1.5322/et)^2)'),
    phi  = cms.string('sqrt(0.01055^2 + (1.5636/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.0603^2 + (0.96/sqrt(et))^2 + (1.58/et)^2))'),
    eta  = cms.string('sqrt(0.00959^2 + (1.5176/et)^2)'),
    phi  = cms.string('sqrt(0.01042^2 + (1.5547/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.0535^2 + (1/sqrt(et))^2 + (1.31/et)^2))'),
    eta  = cms.string('sqrt(0.00971^2 + (1.5233/et)^2)'),
    phi  = cms.string('sqrt(0.01043^2 + (1.5674/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.0472^2 + (1.039/sqrt(et))^2 + (0.67/et)^2))'),
    eta  = cms.string('sqrt(0.00966^2 + (1.5239/et)^2)'),
    phi  = cms.string('sqrt(0.01021^2 + (1.5725/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.0561^2 + (1.016/sqrt(et))^2 + (1.31/et)^2))'),
    eta  = cms.string('sqrt(0.00969^2 + (1.5407/et)^2)'),
    phi  = cms.string('sqrt(0.00981^2 + (1.5962/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.0543^2 + (1.0701/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.00976^2 + (1.5745/et)^2)'),
    phi  = cms.string('sqrt(0.01039^2 + (1.6025/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.0544^2 + (1.071/sqrt(et))^2 + (1.2/et)^2))'),
    eta  = cms.string('sqrt(0.01025^2 + (1.5794/et)^2)'),
    phi  = cms.string('sqrt(0.01002^2 + (1.6162/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.0537^2 + (1.1222/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01038^2 + (1.5695/et)^2)'),
    phi  = cms.string('sqrt(0.01093^2 + (1.6176/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0.0509^2 + (1.1539/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01099^2 + (1.6058/et)^2)'),
    phi  = cms.string('sqrt(0.01018^2 + (1.6654/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0.0444^2 + (1.2024/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01353^2 + (1.6026/et)^2)'),
    phi  = cms.string('sqrt(0.0108^2 + (1.6873/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.0426^2 + (1.2401/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01863^2 + (1.5378/et)^2)'),
    phi  = cms.string('sqrt(0.01338^2 + (1.7234/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0.0435^2 + (1.2544/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01234^2 + (1.6169/et)^2)'),
    phi  = cms.string('sqrt(0.01361^2 + (1.7677/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.566'),
    et  = cms.string('et * (sqrt(0^2 + (1.2566/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01286^2 + (1.6066/et)^2)'),
    phi  = cms.string('sqrt(0.01145^2 + (1.7966/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.566<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0^2 + (1.1734/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01086^2 + (1.6108/et)^2)'),
    phi  = cms.string('sqrt(0.01042^2 + (1.792/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0^2 + (1.1259/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01056^2 + (1.626/et)^2)'),
    phi  = cms.string('sqrt(0.00983^2 + (1.7587/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(0^2 + (1.0982/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.00922^2 + (1.6977/et)^2)'),
    phi  = cms.string('sqrt(0.0094^2 + (1.7323/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(0^2 + (0.988/sqrt(et))^2 + (2.678/et)^2))'),
    eta  = cms.string('sqrt(0.00982^2 + (1.6827/et)^2)'),
    phi  = cms.string('sqrt(0.0085^2 + (1.7074/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(0^2 + (0.957/sqrt(et))^2 + (2.569/et)^2))'),
    eta  = cms.string('sqrt(0.01029^2 + (1.6801/et)^2)'),
    phi  = cms.string('sqrt(0.00834^2 + (1.6954/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(0^2 + (0.9455/sqrt(et))^2 + (2.48/et)^2))'),
    eta  = cms.string('sqrt(0.01114^2 + (1.6469/et)^2)'),
    phi  = cms.string('sqrt(0.0082^2 + (1.6705/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0^2 + (0.9015/sqrt(et))^2 + (2.75/et)^2))'),
    eta  = cms.string('sqrt(0.0105^2 + (1.6086/et)^2)'),
    phi  = cms.string('sqrt(0.00883^2 + (1.6729/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(0^2 + (0.9007/sqrt(et))^2 + (3.059/et)^2))'),
    eta  = cms.string('sqrt(0.01117^2 + (1.926/et)^2)'),
    phi  = cms.string('sqrt(0.01045^2 + (1.7223/et)^2)'),
    ),
    ),
                                          constraints = cms.vdouble(0)
                                          )

bjetResolutionPF = stringResolution.clone(parametrization = 'EtEtaPhi',
                                          functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.087'),
    et  = cms.string('et * (sqrt(0.0849^2 + (0.855/sqrt(et))^2 + (3.43/et)^2))'),
    eta  = cms.string('sqrt(0.00672^2 + (1.5978/et)^2)'),
    phi  = cms.string('sqrt(0.00842^2 + (1.6991/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.087<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.08^2 + (0.959/sqrt(et))^2 + (2.15/et)^2))'),
    eta  = cms.string('sqrt(0.00595^2 + (1.6273/et)^2)'),
    phi  = cms.string('sqrt(0.00802^2 + (1.7211/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.0734^2 + (0.998/sqrt(et))^2 + (1.67/et)^2))'),
    eta  = cms.string('sqrt(0.00656^2 + (1.6177/et)^2)'),
    phi  = cms.string('sqrt(0.00789^2 + (1.7235/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.0786^2 + (0.935/sqrt(et))^2 + (2.23/et)^2))'),
    eta  = cms.string('sqrt(0.00618^2 + (1.6293/et)^2)'),
    phi  = cms.string('sqrt(0.00779^2 + (1.7145/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.0721^2 + (1.001/sqrt(et))^2 + (1.42/et)^2))'),
    eta  = cms.string('sqrt(0.00623^2 + (1.6427/et)^2)'),
    phi  = cms.string('sqrt(0.00806^2 + (1.7107/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.0682^2 + (1.011/sqrt(et))^2 + (1.37/et)^2))'),
    eta  = cms.string('sqrt(0.00678^2 + (1.657/et)^2)'),
    phi  = cms.string('sqrt(0.00806^2 + (1.7229/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.0637^2 + (1.037/sqrt(et))^2 + (1.24/et)^2))'),
    eta  = cms.string('sqrt(0.00633^2 + (1.6528/et)^2)'),
    phi  = cms.string('sqrt(0.00785^2 + (1.722/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.0658^2 + (1.032/sqrt(et))^2 + (0.83/et)^2))'),
    eta  = cms.string('sqrt(0.00684^2 + (1.6606/et)^2)'),
    phi  = cms.string('sqrt(0.00777^2 + (1.7348/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0.0661^2 + (1.0633/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.00722^2 + (1.6615/et)^2)'),
    phi  = cms.string('sqrt(0.00786^2 + (1.7394/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.0649^2 + (1.0755/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.00803^2 + (1.655/et)^2)'),
    phi  = cms.string('sqrt(0.0077^2 + (1.7591/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0.0731^2 + (1.054/sqrt(et))^2 + (0.6/et)^2))'),
    eta  = cms.string('sqrt(0.00817^2 + (1.678/et)^2)'),
    phi  = cms.string('sqrt(0.00807^2 + (1.7585/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0.068^2 + (1.0925/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0085^2 + (1.6774/et)^2)'),
    phi  = cms.string('sqrt(0.00805^2 + (1.7778/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0.0662^2 + (1.1339/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.00858^2 + (1.7079/et)^2)'),
    phi  = cms.string('sqrt(0.00852^2 + (1.7953/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0.064^2 + (1.1553/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.00927^2 + (1.7597/et)^2)'),
    phi  = cms.string('sqrt(0.00856^2 + (1.8331/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0.0692^2 + (1.1655/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01203^2 + (1.7792/et)^2)'),
    phi  = cms.string('sqrt(0.00904^2 + (1.8867/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0.0756^2 + (1.1773/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01479^2 + (1.774/et)^2)'),
    phi  = cms.string('sqrt(0.0108^2 + (1.9241/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0.0761^2 + (1.1932/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01191^2 + (1.791/et)^2)'),
    phi  = cms.string('sqrt(0.011^2 + (2.006/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.566'),
    et  = cms.string('et * (sqrt(0.0631^2 + (1.2178/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01287^2 + (1.764/et)^2)'),
    phi  = cms.string('sqrt(0.01025^2 + (1.998/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.566<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0.0429^2 + (1.2103/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01092^2 + (1.789/et)^2)'),
    phi  = cms.string('sqrt(0.00904^2 + (1.99/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0^2 + (1.2206/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.00985^2 + (1.784/et)^2)'),
    phi  = cms.string('sqrt(0.0083^2 + (1.954/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(0^2 + (1.1902/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.00916^2 + (1.881/et)^2)'),
    phi  = cms.string('sqrt(0.00728^2 + (1.937/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(0^2 + (1.1441/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01144^2 + (1.811/et)^2)'),
    phi  = cms.string('sqrt(0.00664^2 + (1.9/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(0^2 + (1.1221/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.01113^2 + (1.864/et)^2)'),
    phi  = cms.string('sqrt(0.00618^2 + (1.857/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(0^2 + (1.0843/sqrt(et))^2 + (1.73/et)^2))'),
    eta  = cms.string('sqrt(0.01176^2 + (1.844/et)^2)'),
    phi  = cms.string('sqrt(0.00624^2 + (1.855/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0^2 + (1.0579/sqrt(et))^2 + (1.78/et)^2))'),
    eta  = cms.string('sqrt(0.01076^2 + (1.821/et)^2)'),
    phi  = cms.string('sqrt(0.00543^2 + (1.884/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(0^2 + (1.1037/sqrt(et))^2 + (1.62/et)^2))'),
    eta  = cms.string('sqrt(0.00883^2 + (2.189/et)^2)'),
    phi  = cms.string('sqrt(0.00836^2 + (1.959/et)^2)'),
    ),
    ),
                                          constraints = cms.vdouble(0)
                                          )

muonResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                        functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.100'),
    et  = cms.string('et * (0.00527 + 0.0001431 * et)'),
    eta  = cms.string('sqrt(0.0004308^2 + (0.00314/et)^2)'),
    phi  = cms.string('sqrt(7.22e-05^2 + (8e-05/sqrt(et))^2 + (0.00314/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.100<=abs(eta) && abs(eta)<0.200'),
    et  = cms.string('et * (0.00546 + 0.0001361 * et)'),
    eta  = cms.string('sqrt(0.0003871^2 + (0.00267/et)^2)'),
    phi  = cms.string('sqrt(7.28e-05^2 + (0.00013/sqrt(et))^2 + (0.00298/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.200<=abs(eta) && abs(eta)<0.300'),
    et  = cms.string('et * (0.00604 + 0.0001343 * et)'),
    eta  = cms.string('sqrt(0.0003385^2 + (0.00262/et)^2)'),
    phi  = cms.string('sqrt(6.62e-05^2 + (0.000293/sqrt(et))^2 + (0.00293/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.300<=abs(eta) && abs(eta)<0.400'),
    et  = cms.string('et * (0.00675 + 0.0001295 * et)'),
    eta  = cms.string('sqrt(0.000307^2 + (0.00244/et)^2)'),
    phi  = cms.string('sqrt(6.34e-05^2 + (0.000341/sqrt(et))^2 + (0.00291/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.400<=abs(eta) && abs(eta)<0.500'),
    et  = cms.string('et * (0.00694 + 0.000134 * et)'),
    eta  = cms.string('sqrt(0.0002882^2 + (0.00197/et)^2)'),
    phi  = cms.string('sqrt(6.36e-05^2 + (0.000336/sqrt(et))^2 + (0.00306/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.500<=abs(eta) && abs(eta)<0.600'),
    et  = cms.string('et * (0.00751 + 0.0001272 * et)'),
    eta  = cms.string('sqrt(0.0002852^2 + (0.0019/et)^2)'),
    phi  = cms.string('sqrt(6.57e-05^2 + (0.000314/sqrt(et))^2 + (0.00308/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.600<=abs(eta) && abs(eta)<0.700'),
    et  = cms.string('et * (0.00765 + 0.0001277 * et)'),
    eta  = cms.string('sqrt(0.0002842^2 + (0.00237/et)^2)'),
    phi  = cms.string('sqrt(6.05e-05^2 + (0.000396/sqrt(et))^2 + (0.00312/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.700<=abs(eta) && abs(eta)<0.800'),
    et  = cms.string('et * (0.00819 + 0.000125 * et)'),
    eta  = cms.string('sqrt(0.0002752^2 + (0.00294/et)^2)'),
    phi  = cms.string('sqrt(6.71e-05^2 + (0.000353/sqrt(et))^2 + (0.0032/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.800<=abs(eta) && abs(eta)<0.900'),
    et  = cms.string('et * (0.00924 + 0.0001259 * et)'),
    eta  = cms.string('sqrt(0.0002579^2 + (0.00271/et)^2)'),
    phi  = cms.string('sqrt(6.03e-05^2 + (0.000458/sqrt(et))^2 + (0.00319/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.900<=abs(eta) && abs(eta)<1.000'),
    et  = cms.string('et * (0.0108 + 0.0001502 * et)'),
    eta  = cms.string('sqrt(0.0002424^2 + (0.00278/et)^2)'),
    phi  = cms.string('sqrt(6.49e-05^2 + (0.000539/sqrt(et))^2 + (0.00323/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.000<=abs(eta) && abs(eta)<1.100'),
    et  = cms.string('et * (0.01196 + 0.0001424 * et)'),
    eta  = cms.string('sqrt(0.0002268^2 + (0.00355/et)^2)'),
    phi  = cms.string('sqrt(7.1e-05^2 + (0.0005/sqrt(et))^2 + (0.00346/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.100<=abs(eta) && abs(eta)<1.200'),
    et  = cms.string('et * (0.01364 + 0.0001263 * et)'),
    eta  = cms.string('sqrt(0.0002178^2 + (0.00345/et)^2)'),
    phi  = cms.string('sqrt(5.27e-05^2 + (0.000635/sqrt(et))^2 + (0.00365/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.200<=abs(eta) && abs(eta)<1.300'),
    et  = cms.string('et * (0.01412 + 0.0001404 * et)'),
    eta  = cms.string('sqrt(0.0002077^2 + (0.00356/et)^2)'),
    phi  = cms.string('sqrt(5.38e-05^2 + (0.000682/sqrt(et))^2 + (0.00364/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.300<=abs(eta) && abs(eta)<1.400'),
    et  = cms.string('et * (0.01473 + 0.0001449 * et)'),
    eta  = cms.string('sqrt(0.0002175^2 + (0.00354/et)^2)'),
    phi  = cms.string('sqrt(5.57e-05^2 + (0.000737/sqrt(et))^2 + (0.00339/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.400<=abs(eta) && abs(eta)<1.500'),
    et  = cms.string('et * (0.01408 + 0.0001535 * et)'),
    eta  = cms.string('sqrt(0.0002271^2 + (0.00344/et)^2)'),
    phi  = cms.string('sqrt(0^2 + (0.000909/sqrt(et))^2 + (0.00309/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.500<=abs(eta) && abs(eta)<1.600'),
    et  = cms.string('et * (0.01252 + 0.0001749 * et)'),
    eta  = cms.string('sqrt(0.0002216^2 + (0.00366/et)^2)'),
    phi  = cms.string('sqrt(0^2 + (0.000985/sqrt(et))^2 + (0.00251/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.600<=abs(eta) && abs(eta)<1.700'),
    et  = cms.string('et * (0.01281 + 0.0001934 * et)'),
    eta  = cms.string('sqrt(0.000227^2 + (0.00376/et)^2)'),
    phi  = cms.string('sqrt(5.5e-05^2 + (0.000935/sqrt(et))^2 + (0.00323/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.700<=abs(eta) && abs(eta)<1.800'),
    et  = cms.string('et * (0.01256 + 0.0002733 * et)'),
    eta  = cms.string('sqrt(0.0002608^2 + (0.00333/et)^2)'),
    phi  = cms.string('sqrt(7.6e-05^2 + (0.001002/sqrt(et))^2 + (0.00307/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.800<=abs(eta) && abs(eta)<1.900'),
    et  = cms.string('et * (0.01434 + 0.000339 * et)'),
    eta  = cms.string('sqrt(0.0002799^2 + (0.00382/et)^2)'),
    phi  = cms.string('sqrt(9.4e-05^2 + (0.001078/sqrt(et))^2 + (0.0029/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.900<=abs(eta) && abs(eta)<2.000'),
    et  = cms.string('et * (0.01478 + 0.000451 * et)'),
    eta  = cms.string('sqrt(0.0002988^2 + (0.00402/et)^2)'),
    phi  = cms.string('sqrt(8.8e-05^2 + (0.001284/sqrt(et))^2 + (0.0024/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.000<=abs(eta) && abs(eta)<2.100'),
    et  = cms.string('et * (0.01692 + 0.000581 * et)'),
    eta  = cms.string('sqrt(0.0003203^2 + (0.00434/et)^2)'),
    phi  = cms.string('sqrt(0.000148^2 + (0.00115/sqrt(et))^2 + (0.00352/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.100<=abs(eta) && abs(eta)<2.200'),
    et  = cms.string('et * (0.01935 + 0.000653 * et)'),
    eta  = cms.string('sqrt(0.0003774^2 + (0.00437/et)^2)'),
    phi  = cms.string('sqrt(0.00018^2 + (0.0011/sqrt(et))^2 + (0.0044/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.200<=abs(eta) && abs(eta)<2.300'),
    et  = cms.string('et * (0.01821 + 0.000889 * et)'),
    eta  = cms.string('sqrt(0.0004375^2 + (0.00435/et)^2)'),
    phi  = cms.string('sqrt(0.000204^2 + (0.00121/sqrt(et))^2 + (0.00451/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.300<=abs(eta) && abs(eta)<2.400'),
    et  = cms.string('et * (0.0173 + 0.001261 * et)'),
    eta  = cms.string('sqrt(0.0005159^2 + (0.00488/et)^2)'),
    phi  = cms.string('sqrt(0.000223^2 + (0.00152/sqrt(et))^2 + (0.00459/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )

elecResolution = stringResolution.clone(parametrization = 'EtEtaPhi',
                                        functions = cms.VPSet(
    cms.PSet(
    bin = cms.string('0.000<=abs(eta) && abs(eta)<0.174'),
    et  = cms.string('et * (sqrt(0.0059^2 + (0.0746/sqrt(et))^2 + (0.177/et)^2))'),
    eta  = cms.string('sqrt(0.000453^2 + (0.00379/et)^2)'),
    phi  = cms.string('sqrt(0^2 + (0.0014637/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.174<=abs(eta) && abs(eta)<0.261'),
    et  = cms.string('et * (sqrt(0.00536^2 + (0.0798/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0003882^2 + (0.00319/et)^2)'),
    phi  = cms.string('sqrt(0^2 + (0.00146341/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.261<=abs(eta) && abs(eta)<0.348'),
    et  = cms.string('et * (sqrt(0.00485^2 + (0.0838/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0003601^2 + (0.0001/et)^2)'),
    phi  = cms.string('sqrt(9.99e-05^2 + (0.001282/sqrt(et))^2 + (0.00245/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.348<=abs(eta) && abs(eta)<0.435'),
    et  = cms.string('et * (sqrt(0.0056^2 + (0.0878/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0003317^2 + (0.00176/et)^2)'),
    phi  = cms.string('sqrt(8.71e-05^2 + (0.001451/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.435<=abs(eta) && abs(eta)<0.522'),
    et  = cms.string('et * (sqrt(0.00287^2 + (0.0962/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.000313^2 + (0.00241/et)^2)'),
    phi  = cms.string('sqrt(6.66e-05^2 + (0.001485/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.522<=abs(eta) && abs(eta)<0.609'),
    et  = cms.string('et * (sqrt(0.00395^2 + (0.089/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0003226^2 + (0.00053/et)^2)'),
    phi  = cms.string('sqrt(9.9e-05^2 + (0.001377/sqrt(et))^2 + (0.00228/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.609<=abs(eta) && abs(eta)<0.696'),
    et  = cms.string('et * (sqrt(0.00364^2 + (0.0912/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0003297^2 + (0.00275/et)^2)'),
    phi  = cms.string('sqrt(0.0001094^2 + (0.00134/sqrt(et))^2 + (0.00341/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.696<=abs(eta) && abs(eta)<0.783'),
    et  = cms.string('et * (sqrt(0^2 + (0.10127/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0003302^2 + (0.0026/et)^2)'),
    phi  = cms.string('sqrt(0.0001378^2 + (0.001337/sqrt(et))^2 + (0.00272/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.783<=abs(eta) && abs(eta)<0.870'),
    et  = cms.string('et * (sqrt(0.0021^2 + (0.1076/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0003062^2 + (0.00296/et)^2)'),
    phi  = cms.string('sqrt(0.000128^2 + (0.001587/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.870<=abs(eta) && abs(eta)<0.957'),
    et  = cms.string('et * (sqrt(0^2 + (0.1096/sqrt(et))^2 + (0.068/et)^2))'),
    eta  = cms.string('sqrt(0.0002929^2 + (0.00279/et)^2)'),
    phi  = cms.string('sqrt(0.000146^2 + (0.001682/sqrt(et))^2 + (0.0002/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('0.957<=abs(eta) && abs(eta)<1.044'),
    et  = cms.string('et * (sqrt(0^2 + (0.1228/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0002839^2 + (0.00314/et)^2)'),
    phi  = cms.string('sqrt(0.0001911^2 + (0.001345/sqrt(et))^2 + (0.0048/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.044<=abs(eta) && abs(eta)<1.131'),
    et  = cms.string('et * (sqrt(0^2 + (0.1323/sqrt(et))^2 + (0.43/et)^2))'),
    eta  = cms.string('sqrt(0.0002854^2 + (0.00363/et)^2)'),
    phi  = cms.string('sqrt(0.0001752^2 + (0.00187/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.131<=abs(eta) && abs(eta)<1.218'),
    et  = cms.string('et * (sqrt(0^2 + (0.1545/sqrt(et))^2 + (0.588/et)^2))'),
    eta  = cms.string('sqrt(0.0002801^2 + (0.00364/et)^2)'),
    phi  = cms.string('sqrt(0.000207^2 + (0.00191/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.218<=abs(eta) && abs(eta)<1.305'),
    et  = cms.string('et * (sqrt(0^2 + (0.1469/sqrt(et))^2 + (0.738/et)^2))'),
    eta  = cms.string('sqrt(0.0002675^2 + (0.00364/et)^2)'),
    phi  = cms.string('sqrt(0.000249^2 + (0.001882/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.305<=abs(eta) && abs(eta)<1.392'),
    et  = cms.string('et * (sqrt(0^2 + (0.162/sqrt(et))^2 + (0.689/et)^2))'),
    eta  = cms.string('sqrt(0.0002894^2 + (0.00391/et)^2)'),
    phi  = cms.string('sqrt(0.000266^2 + (0.001999/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.392<=abs(eta) && abs(eta)<1.479'),
    et  = cms.string('et * (sqrt(0^2 + (0.1778/sqrt(et))^2 + (0.664/et)^2))'),
    eta  = cms.string('sqrt(0.0002828^2 + (0.00337/et)^2)'),
    phi  = cms.string('sqrt(0.00029^2 + (0.00204/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.479<=abs(eta) && abs(eta)<1.653'),
    et  = cms.string('et * (sqrt(0.0228^2 + (0.208/sqrt(et))^2 + (0.42/et)^2))'),
    eta  = cms.string('sqrt(0.0002823^2 + (0.00331/et)^2)'),
    phi  = cms.string('sqrt(0.000336^2 + (0.00217/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.653<=abs(eta) && abs(eta)<1.740'),
    et  = cms.string('et * (sqrt(0.0141^2 + (0.18/sqrt(et))^2 + (0.67/et)^2))'),
    eta  = cms.string('sqrt(0.0002728^2 + (0.00446/et)^2)'),
    phi  = cms.string('sqrt(0.000385^2 + (0.0021/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.740<=abs(eta) && abs(eta)<1.830'),
    et  = cms.string('et * (sqrt(0.008^2 + (0.18/sqrt(et))^2 + (0.38/et)^2))'),
    eta  = cms.string('sqrt(0.0003102^2 + (0/et)^2)'),
    phi  = cms.string('sqrt(0.000364^2 + (0.00267/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.830<=abs(eta) && abs(eta)<1.930'),
    et  = cms.string('et * (sqrt(0.0095^2 + (0.1742/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0003092^2 + (0.00393/et)^2)'),
    phi  = cms.string('sqrt(0.000323^2 + (0.003/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('1.930<=abs(eta) && abs(eta)<2.043'),
    et  = cms.string('et * (sqrt(0.0154^2 + (0.108/sqrt(et))^2 + (0.523/et)^2))'),
    eta  = cms.string('sqrt(0.0003313^2 + (0.00459/et)^2)'),
    phi  = cms.string('sqrt(0.000312^2 + (0.00319/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.043<=abs(eta) && abs(eta)<2.172'),
    et  = cms.string('et * (sqrt(0.0129^2 + (0.1387/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0003632^2 + (0.0046/et)^2)'),
    phi  = cms.string('sqrt(0.0003^2 + (0.00343/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.172<=abs(eta) && abs(eta)<2.322'),
    et  = cms.string('et * (sqrt(0.0123^2 + (0.1297/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.0004504^2 + (0.00348/et)^2)'),
    phi  = cms.string('sqrt(0.000278^2 + (0.00372/sqrt(et))^2 + (0/et)^2)'),
    ),
    cms.PSet(
    bin = cms.string('2.322<=abs(eta) && abs(eta)<2.500'),
    et  = cms.string('et * (sqrt(0.0189^2 + (0.0762/sqrt(et))^2 + (0/et)^2))'),
    eta  = cms.string('sqrt(0.000575^2 + (0.00358/et)^2)'),
    phi  = cms.string('sqrt(0.00026^2 + (0.00486/sqrt(et))^2 + (0/et)^2)'),
    ),
    ),
                                        constraints = cms.vdouble(0)
                                        )

metResolutionPF  = stringResolution.clone(parametrization = 'EtEtaPhi',
                                          functions = cms.VPSet(
    cms.PSet(
    et  = cms.string('et * (sqrt(0.0351^2 + (0/sqrt(et))^2 + (20.288/et)^2))'),
    eta  = cms.string('sqrt(0^2 + (0/sqrt(et))^2 + (0/et)^2)'),
    phi  = cms.string('sqrt(0^2 + (1.1664/sqrt(et))^2 + (3.579/et)^2)'),
    ),
    ),
                                          constraints = cms.vdouble(0)
                                          )
