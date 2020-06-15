#! /usr/bin/bash

#  _____           _    _
# |_   _|___  ___ | |_ (_) _ __    __ _
#   | | / _ \/ __|| __|| || '_ \  / _` |
#   | ||  __/\__ \| |_ | || | | || (_| |
#   |_| \___||___/ \__||_||_| |_| \__, |
#                                 |___/
#


#
# SPECIALIST NETWORKS
#

function specialists() {

    local ROOT_DIR=./results-2020_04_24/Specialist/
    local WAIT_TIME=20
    local SKIP_EXISTING=true

    # look inside of root_dir for model checkpoints
    for MODEL_PATH in $(find $ROOT_DIR -name *model.pt); do

        # skip if there are already existing test results
        local MODEL_DIR=$(dirname "${MODEL_PATH}")
        if $SKIP_EXISTING; then
            if grep -rsq "Completed testing Specialist model" $MODEL_DIR; then
                echo "Skipping $MODEL_DIR..."; continue;
            fi
        fi

        local TMUX_SESSION_NAME=$(tr '/.' '  ' <<< $(echo $MODEL_PATH | sed 's/^.\{'$(echo -n $ROOT_DIR | wc -m)'\}//'))
        local TMUX_SESSION_NAME=$(echo ${TMUX_SESSION_NAME%????????})
        if tmux new-session -s "$TMUX_SESSION_NAME" -d \; send-keys \
            "python3 test_specialist.py -s $MODEL_PATH --disconnect" Enter; then
            echo "Starting experiment \"$TMUX_SESSION_NAME\"..."
            sleep $WAIT_TIME
        fi
    done
}

#
# GATING NETWORKS
#

function gatings() {

    local ROOT_DIR=./results-2020_04_24/Gating/
    local WAIT_TIME=20
    local SKIP_EXISTING=true

    # look inside of root_dir for model checkpoints
    for MODEL_PATH in $(find $ROOT_DIR -name *model.pt); do

        # skip if there are already existing test results
        local MODEL_DIR=$(dirname "${MODEL_PATH}")
        if $SKIP_EXISTING; then
            if grep -rsq "Completed testing Gating model" $MODEL_DIR; then
                echo "Skipping $MODEL_DIR..."; continue;
            fi
        fi

        local TMUX_SESSION_NAME=$(tr '/.' '  ' <<< $(echo $MODEL_PATH | sed 's/^.\{'$(echo -n $ROOT_DIR | wc -m)'\}//'))
        local TMUX_SESSION_NAME=$(echo ${TMUX_SESSION_NAME%????????})
        if tmux new-session -s "$TMUX_SESSION_NAME" -d \; send-keys \
            "python3 test_gating.py -s $MODEL_PATH --disconnect" Enter; then
            echo "Starting experiment \"$TMUX_SESSION_NAME\"..."
            sleep $WAIT_TIME
        fi
    done
}

#
# BASELINE NETWORKS
#

function baselines() {

    local ROOT_DIR=./results-2020_04_24/Baseline/
    local WAIT_TIME=60
    local SKIP_EXISTING=true

    # look inside of root_dir for model checkpoints
    for MODEL_PATH in $(find $ROOT_DIR -name *model.pt); do

        # skip if there are already existing test results
        local MODEL_DIR=$(dirname "${MODEL_PATH}")
        if $SKIP_EXISTING; then
            if grep -rsq "Completed testing Baseline model" $MODEL_DIR; then
                echo "Skipping $MODEL_DIR..."; continue;
            fi
        fi

        local TMUX_SESSION_NAME=$(tr '/.' '  ' <<< $(echo $MODEL_PATH | sed 's/^.\{'$(echo -n $ROOT_DIR | wc -m)'\}//'))
        local TMUX_SESSION_NAME=$(echo ${TMUX_SESSION_NAME%????????})
        if tmux new-session -s "$TMUX_SESSION_NAME" -d \; send-keys \
            "python3 test_specialist.py -s $MODEL_PATH --disconnect" Enter; then
            echo "Starting experiment \"$TMUX_SESSION_NAME\"..."
            sleep $WAIT_TIME
        fi
    done
}

#
# ENSEMBLE NETWORKS
#

function ensembles_snr() {

    local ROOT_DIR=./results-2020_04_24/
    local WAIT_TIME=30
    local SKIP_EXISTING=true

    local PATH_TO_SPECIALISTS=$(find $ROOT_DIR/Specialist/gender_all/snr_n05/*x[1-9]/lr1e-03/model.pt)
    local PATH_TO_GATINGS=$(find $ROOT_DIR/Gating/snr/**x[1-9]/lr1e-03/model.pt)
    local OUTPUT_DIR=$ROOT_DIR/Ensemble/snr/

    local REGEX='\/([0-9]+)x([0-9]+)\/'

    for MODEL_GATING in $PATH_TO_GATINGS; do

        [[ $MODEL_GATING =~ $REGEX ]]
        local ARCH_GATING="g${BASH_REMATCH[1]}x${BASH_REMATCH[2]}"

        for MODEL_SPECIALIST in $PATH_TO_SPECIALISTS; do

            [[ $MODEL_SPECIALIST =~ $REGEX ]]
            local ARCH_SPECIALIST="s${BASH_REMATCH[1]}x${BASH_REMATCH[2]}"

            # skip if there are already existing test results
            local MODEL_DIR=$OUTPUT_DIR/$ARCH_GATING/$ARCH_SPECIALIST/
            if $SKIP_EXISTING; then
                if grep -rsq "Completed testing Ensemble model" $MODEL_DIR; then
                    echo "Skipping $MODEL_DIR..."; continue;
                fi
            fi

            local TMUX_SESSION_NAME="Ensemble SNR $ARCH_GATING $ARCH_SPECIALIST"
            if tmux new-session -s "$TMUX_SESSION_NAME" -d \; send-keys \
                "python3 test_ensemble_snr.py -sg $MODEL_GATING -ss $MODEL_SPECIALIST --disconnect" Enter; then
                echo "Starting experiment \"$TMUX_SESSION_NAME\"..."
                sleep $WAIT_TIME
            fi

        done
    done
}


function ensembles_gender() {

    local ROOT_DIR=./results-2020_04_24/
    local WAIT_TIME=30
    local SKIP_EXISTING=true

    local PATH_TO_SPECIALISTS=$(find $ROOT_DIR/Specialist/gender_M/snr_all/*x[1-9]/lr1e-03/model.pt)
    local PATH_TO_GATINGS=$(find $ROOT_DIR/Gating/gender/**x[1-9]/lr1e-03/model.pt)
    local OUTPUT_DIR=$ROOT_DIR/Ensemble/gender/

    local REGEX='\/([0-9]+)x([0-9]+)\/'

    for MODEL_GATING in $PATH_TO_GATINGS; do

        [[ $MODEL_GATING =~ $REGEX ]]
        local ARCH_GATING="g${BASH_REMATCH[1]}x${BASH_REMATCH[2]}"

        for MODEL_SPECIALIST in $PATH_TO_SPECIALISTS; do

            [[ $MODEL_SPECIALIST =~ $REGEX ]]
            local ARCH_SPECIALIST="s${BASH_REMATCH[1]}x${BASH_REMATCH[2]}"

            # skip if there are already existing test results
            local MODEL_DIR=$OUTPUT_DIR/$ARCH_GATING/$ARCH_SPECIALIST/
            if $SKIP_EXISTING; then
                if grep -rsq "Completed testing Ensemble model" $MODEL_DIR; then
                    echo "Skipping $MODEL_DIR..."; continue;
                fi
            fi

            local TMUX_SESSION_NAME="Ensemble GENDER $ARCH_GATING $ARCH_SPECIALIST"
            if tmux new-session -s "$TMUX_SESSION_NAME" -d \; send-keys \
                "python3 test_ensemble_gender.py -sg $MODEL_GATING -ss $MODEL_SPECIALIST --disconnect" Enter; then
                echo "Starting experiment \"$TMUX_SESSION_NAME\"..."
                sleep $WAIT_TIME
            fi

        done
    done
}

"$@"