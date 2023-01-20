#!/bin/bash

# Check if apptainer (or singularity) is available and that unprivileged user namespace is enabled (needed for nested containers)
# It is needed for unit tests in modules HeterogeneousCore/SonicTriton and RecoEcal/EgammaClusterProducers
# It avoids TritonService fallback server error

if type apptainer >& /dev/null; then
        echo "has apptainer"
        CONTAINER_CONFIG="/etc/apptainer/apptainer.conf"
else
        echo "missing apptainer, checking for singularity..."
        if type singularity >& /dev/null; then
                echo "has singularity"
                CONTAINER_CONFIG="/etc/singularity/singularity.conf"
        else
                echo "missing singularity also" && exit 1
        fi
fi

if [ -n "$SINGULARITY_CONTAINER" ]; then
        if grep -q "^allow setuid = no" $CONTAINER_CONFIG && unshare -U echo >/dev/null 2>&1; then
                echo "has unprivileged user namespace support"
        else
                echo "missing unprivileged user namespace support" && exit 1
        fi
fi
