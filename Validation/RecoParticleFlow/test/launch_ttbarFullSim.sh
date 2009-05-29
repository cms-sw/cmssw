#!/bin/sh

castorDir=/castor/cern.ch/user/p/pjanot/CMSSW220pre1/
cmsswDir=/afs/cern.ch/user/p/pjanot/scratch0/CMSSW_2_2_0_pre1/src

for ((job=1;job<=50;job++));
  do
  echo "JOB "$job
  name="ttbar_Full_"${job}
  displayfilename="display_"${name}".root"
  aodfilename="aod_"${name}".root"
  recofilename="reco_"${name}".root"
  echo $name

  seed1=$(( ($job+1)*143123 ))
  sed -e "s/==SEED==/${seed1}/" ttbarFullSim_cfg.py > tmp_cfg

#Start to write the script
cat > job_${name}.sh << EOF
#!/bin/sh
cd $cmsswDir
eval \`scramv1 runtime -sh\`
cd -
#commande pour decoder le .cfg
cat > TEST_cfg.py << "EOF"
EOF

#Ajoute le .cfg au script
cat  tmp_cfg>> job_${name}.sh

# On poursuit le script
echo "EOF" >> job_${name}.sh
cat >> job_${name}.sh << EOF
cmsRun TEST_cfg.py >& log

#rfcp display.root $castorDir$displayfilename
rfcp aod.root $castorDir$aodfilename
#rfcp reco.root $castorDir$recofilename

EOF
chmod 755 job_${name}.sh
bsub -q cmst3 -J $name $PWD/job_${name}.sh


done

