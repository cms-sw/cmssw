#!/bin/sh

castorhome="/castor/cern.ch/user/p/pjanot/CMSSW300pre6/"
cmsswDir=/afs/cern.ch/user/p/pjanot/scratch0/CMSSW_3_0_0_pre6/src

# 150000 evenements en 8h
# for energy in 1 2 3 4 5 7 9
# 100000 evenements en 8h
# for energy in 12 15 20
# 50000 evenements en 8h
#for energy in 30 50 100 200
# 30000 evenements en 8h
#for energy in 300 500 700 1000
#for energy in 15
# Some electrons with pt = 35 GeV/c
eMin=10
eMax=100
#echo "Energy "$energy
echo "E = "$eMin" -> "$eMax

#    for pid in 211 -211 130 321 -321 2112 -2112 2212 -2212
#    for pid in -2212
for pid in 130 22
  do
  echo "PID "$pid

#	DIRNAME=$castorhome"SingleParticlePID"$pid"-E"$energy
  DIRNAME=$castorhome"SingleParticlePID"$pid
  echo "Creating directory "$DIRNAME
  rfmkdir $DIRNAME
  for ((job=0;job<=19;job++));
    do
    
    echo "JOB "$job
#	    NAME="SingleParticlePID"$pid"-E"$energy"_"$job
    NAME="SingleParticlePID"$pid"_E"$eMin"_"$eMax"_"$job

    echo "Name "$NAME
    outputfilename=$NAME".root"
    echo $outputfilename
    seed1=$(( ($job+1)*143123 ))
    
    
    sed -e "s/==OUTPUT==/$outputfilename/" -e "s/==SEED==/$seed1/" -e "s/==PID==/$pid/" -e "s/==EMIN==/$eMin/" -e "s/==EMAX==/$eMax/" SingleParticle_cfg.py > tmp_cfgfile
	    #sed -e "s/==OUTPUT==/$outputfilename/" -e "s/==SEED==/$seed1/" -e "s/==PID==/$pid/" -e "s/==ENERGY==/$energy/" template.cfg > tmp_cfgfile
    
       	#Start to write the script
    cat > job_${NAME}.sh << EOF
#!/bin/sh
cd $cmsswDir
eval \`scramv1 runtime -sh\`
cd -

#commande pour decoder le .cfg
cat > TEST_cfg.py << "EOF"
EOF

#Ajoute le .cfg au script
cat tmp_cfgfile >> job_${NAME}.sh

# On poursuit le script
echo "EOF" >> job_${NAME}.sh
cat >> job_${NAME}.sh << EOF
cmsRun TEST_cfg.py

rfcp $outputfilename $DIRNAME/$outputfilename

EOF
chmod 755 job_${NAME}.sh
bsub -q cmst3 -J $NAME -R "mem>2000" $PWD/job_${NAME}.sh

	    done
	done
