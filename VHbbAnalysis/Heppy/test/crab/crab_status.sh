# usage: crab_status.sh crab_projects_V21bis_001 2>&1 | tee crab_log.log
cd $1
printf "Execute crab command: type \"1\" for status or \"2\" for resubmit or \"3\" for purge and press return\n"
read crab_option

if [ $crab_option -eq 1 ]; then 
  echo "crab status inside "$PWD
  find . -maxdepth 1 -type d \( ! -name . \) -exec bash -c "cd '{}' && crab status -d ./" \;
elif [ $crab_option -eq 2 ]; then
  echo "crab resubmit inside "$PWD
  find . -maxdepth 1 -type d \( ! -name . \) -exec bash -c "cd '{}' && crab resubmit --maxmemory=3000 -d ./" \;
elif [ $crab_option -eq 3 ]; then
  echo "crab purge inside "$PWD
  find . -maxdepth 1 -type d \( ! -name . \) -exec bash -c "cd '{}' && crab purge -d ./" \;
fi
