rm -rf ~/public/forColin/310FullSimAN/*
mkdir ~/public/forColin/310FullSimAN/Generic
mkdir ~/public/forColin/310FullSimAN/JEC
mkdir ~/public/forColin/310FullSimAN/FastVsFull
mkdir ~/public/forColin/310FullSimAN/TauMinus
mkdir ~/public/forColin/310FullSimAN/Material
rm temp_*

#Generic
perl -pe "s/FILE1/benchmark_310_Full.root/g" plotTemplate.C >! temp_1
perl -pe "s/FILE2/benchmark_310_Full.root/g" temp_1 >! temp_2
perl -pe "s/DIR1/PFlowTaus_barrel/g" temp_2 >! temp_3
perl -pe "s/DIR2/CaloTaus_barrel/g" temp_3 >! temp_4
perl -pe "s/OUTDIR/Plots_Barrel/g" temp_4 >! plotGeneric.C
root -b plotGeneric.C
perl -pe "s/DIR1/PFlowTaus_endcap/g" temp_2 >! temp_3
perl -pe "s/DIR2/CaloTaus_endcap/g" temp_3 >! temp_4
perl -pe "s/OUTDIR/Plots_Endcap/g" temp_4 >! plotGeneric.C
root -b plotGeneric.C

cp -r Plots_* ~/public/forColin/310FullSimAN/Generic

#JEC
perl -pe "s/FILE2/benchmark_225_JEC.root/g" temp_1 >! temp_2
perl -pe "s/DIR1/PFlowTaus_barrel/g" temp_2 >! temp_3
perl -pe "s/DIR2/CaloTaus_barrel/g" temp_3 >! temp_4
perl -pe "s/OUTDIR/Plots_Barrel/g" temp_4 >! plotJEC.C
root -b plotJEC.C
perl -pe "s/DIR1/PFlowTaus_endcap/g" temp_2 >! temp_3
perl -pe "s/DIR2/CaloTaus_endcap/g" temp_3 >! temp_4
perl -pe "s/OUTDIR/Plots_Endcap/g" temp_4 >! plotJEC.C
root -b plotJEC.C
cp -r Plots_* ~/public/forColin/310FullSimAN/JEC

#FullVsFast
perl -pe "s/FILE2/benchmarkPFTau1stVersion.root/g" temp_1 >! temp_2
perl -pe "s/DIR1/PFlowTaus_barrel/g" temp_2 >! temp_3
perl -pe "s/DIR2/pfRecoTauProducerHighEfficiency_barrel/g" temp_3 >! temp_4
perl -pe "s/OUTDIR/Plots_Barrel/g" temp_4 >! plotFastVsFull.C
root -b plotFastVsFull.C
perl -pe "s/DIR1/PFlowTaus_endcap/g" temp_2 >! temp_3
perl -pe "s/DIR2/pfRecoTauProducerHighEfficiency_endcap/g" temp_3 >! temp_4
perl -pe "s/OUTDIR/Plots_Endcap/g" temp_4 >! plotFastVsFull.C
root -b plotFastVsFull.C
cp -r Plots_* ~/public/forColin/310FullSimAN/FastVsFull

#TauMinus
perl -pe "s/FILE2/benchmarkPFTau1stVersion_tauminus.root/g" temp_1 >! temp_2
perl -pe "s/DIR1/PFlowTaus_barrel/g" temp_2 >! temp_3
perl -pe "s/DIR2/CaloTaus_barrel/g" temp_3 >! temp_4
perl -pe "s/OUTDIR/Plots_Barrel/g" temp_4 >! plotTauMinus.C
root -b plotTauMinus.C
perl -pe "s/DIR1/PFlowTaus_endcap/g" temp_2 >! temp_3
perl -pe "s/DIR2/CaloTaus_endcap/g" temp_3 >! temp_4
perl -pe "s/OUTDIR/Plots_Endcap/g" temp_4 >! plotTauMinus.C
root -b plotTauMinus.C
cp -r Plots_* ~/public/forColin/310FullSimAN/TauMinus

#Material
perl -pe "s/FILE2/benchmarkPFTau1stVersion_NoMaterialEffects30KEvents.root/g" temp_1 >! temp_2
perl -pe "s/DIR1/PFlowTaus_barrel/g" temp_2 >! temp_3
perl -pe "s/DIR2/PFlowTaus_barrel/g" temp_3 >! temp_4
perl -pe "s/OUTDIR/Plots_Barrel/g" temp_4 >! plotMaterial.C
root -b plotMaterial.C
perl -pe "s/DIR1/PFlowTaus_endcap/g" temp_2 >! temp_3
perl -pe "s/DIR2/PFlowTaus_endcap/g" temp_3 >! temp_4
perl -pe "s/OUTDIR/Plots_Endcap/g" temp_4 >! plotMaterial.C
root -b plotMaterial.C
cp -r Plots_* ~/public/forColin/310FullSimAN/Material
