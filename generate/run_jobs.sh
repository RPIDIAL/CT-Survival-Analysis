#!/bin/bash

NOTEBOOK_FILE="Calculate_radiomics_full.py"
ARR_BEGIN=5.0
ARR_END=50.0
SLAB_INC=5.0

ERO_BEGIN=0.0
ERO_END=5.0
ERO_INC=1.0

BATCH_SIZE=25

arr=($(seq "$ARR_BEGIN" "$SLAB_INC" "$ARR_END"))


arr_ero=($(seq "$ERO_BEGIN" "$ERO_INC" "$ERO_END"))

echo "Generating bodycomp files with slab thicknesses ${arr[@]} and erosion ${arr_ero[@]}"

count=0

for slab in "${arr[@]}"; do
  for erosion in "${arr_ero[@]}"; do

    echo "Launching slab=$slab erosion=$erosion"

    SLAB_MM="$slab" BONE_ERODE_MM="$erosion" \
    jupyter nbconvert --to notebook --execute "$NOTEBOOK_FILE" \
    >/dev/null 2>&1 &

    ((count++))

    if [ "$count" -ge "$BATCH_SIZE" ]; then
        echo "Waiting for current batch to finish..."
        wait
        count=0
    fi

  done
done

# Wait for any remaining jobs
wait

echo "All jobs finished."