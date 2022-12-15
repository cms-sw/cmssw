#!/bin/bash

# Check if singularity is available and that unprivileged user namespace is enabled (needed for singularity-in-singularity)
# It is needed for unit tests in modules HeterogeneousCore/SonicTriton and RecoEcal/EgammaClusterProducers
# It avoids TritonService fallback server error

if type singularity >& /dev/null; then
        echo "has singularity"
else
        echo "missing singularity" && exit 1
fi

if [ -n "$SINGULARITY_CONTAINER" ]; then
        if grep -q "^allow setuid = no" /etc/singularity/singularity.conf && unshare -U echo >/dev/null 2>&1; then
                echo "has unprivileged user namespace support"
        else
                echo "missing unprivileged user namespace support" && exit 1
        fi
fi
