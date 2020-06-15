#! /usr/bin/bash


declare -A FMT_SPECIALTY
FMT_SPECIALTY=( ['None']='baseline'
                ['-5']='snr_n05'
                ['0']='snr_p00'
                ['5']='snr_p05'
                ['10']='snr_p10'
                ['M']='gen_M'
                ['F']='gen_F' )

declare RESULTS_DIRECTORY=./results-2020_05_04
declare WAIT_TIME_TRAINING=120 # seconds
declare WAIT_TIME_TESTING=20 # seconds
declare SKIP_EXISTING=false

declare -a GATING_HS_SPACE=( 1 2 4 8 16 32 64 96 128 )
declare -a GATING_NL_SPACE=( 2 3 )

declare -a DENOISING_HS_SPACE=( 128 256 512 768 1024 )
declare -a DENOISING_NL_SPACE=( 2 3 )

declare -a LATENT_SPACE=( 'gender' )
declare -a LEARNING_RATE_SPACE=( 1e-3 )
declare -a MIXTURE_SNR_SPACE=( M F )


function train_denoising() {

    for LR in "${LEARNING_RATE_SPACE[@]}"; do
    for HS in "${DENOISING_HS_SPACE[@]}"; do
    for NL in "${DENOISING_NL_SPACE[@]}"; do
    for SV in "${MIXTURE_SNR_SPACE[@]}"; do

        STR_ARCH=$(printf "%04d" $HS)x$NL
        STR_LR=$(python3 -c "print('lr{:.0e}'.format($LR))")
        OUTPUT_DIRECTORY=$RESULTS_DIRECTORY/Denoising/$STR_LR/${FMT_SPECIALTY[$SV]}/$STR_ARCH/

        if $SKIP_EXISTING; then if grep -rsq "Completed training" $OUTPUT_DIRECTORY; then
            echo "Skipping $OUTPUT_DIRECTORY..."; continue; fi; fi

        if [[ "$SV" != "None" ]]; then cmd="-s $SV"; else cmd=""; fi
        TMUX_SESSION_NAME="Denoising -l $LR -z $HS -n $NL $cmd"
        if tmux new-session -s "$TMUX_SESSION_NAME" -d \; send-keys \
            "python3 train_denoising.py -l $LR -z $HS -n $NL $cmd && python3 test_denoising.py -s $OUTPUT_DIRECTORY/model.pt --disconnect" Enter; then
            echo "starting experiment \"$TMUX_SESSION_NAME\"...";
            sleep $WAIT_TIME_TRAINING;
        fi
    done; done; done; done
}


function train_gating() {

    for LR in "${LEARNING_RATE_SPACE[@]}"; do
    for HS in "${GATING_HS_SPACE[@]}"; do
    for NL in "${GATING_NL_SPACE[@]}"; do
    for LS in "${LATENT_SPACE[@]}"; do

        STR_ARCH=$(printf "%04d" $HS)x$NL
        STR_LR=$(python3 -c "print('lr{:.0e}'.format($LR))")
        OUTPUT_DIRECTORY=$RESULTS_DIRECTORY/Gating/$STR_LR/$LS/$STR_ARCH/

        if $SKIP_EXISTING; then if grep -rsq "Completed training" $OUTPUT_DIRECTORY; then
            echo "Skipping $OUTPUT_DIRECTORY..."; continue; fi; fi

        TMUX_SESSION_NAME="Gating -l $LR -z $HS -n $NL -c $LS"
        if tmux new-session -s "$TMUX_SESSION_NAME" -d \; send-keys \
            "python3 train_gating.py -l $LR -z $HS -n $NL -c $LS && python3 test_gating.py -s $OUTPUT_DIRECTORY/model.pt --disconnect" Enter; then
            echo "starting experiment \"$TMUX_SESSION_NAME\"...";
            sleep $WAIT_TIME_TRAINING;
        fi
    done; done; done; done
}

function test_denoising() {

    # look inside of root_dir for model checkpoints
    for MODEL_PATH in $(find ./results-2020_04_24/Baseline/ -name *model.pt); do

        local MODEL_DIR=$(dirname "${MODEL_PATH}")
        if $SKIP_EXISTING; then if grep -rsq "Completed testing" $MODEL_DIR; then
            echo "Skipping $MODEL_DIR..."; continue; fi; fi

        local TMUX_SESSION_NAME=$(tr '/.' '  ' <<< $(echo $MODEL_PATH | sed 's/^.\{'$(echo -n $ROOT_DIR | wc -m)'\}//'))
        local TMUX_SESSION_NAME=$(echo ${TMUX_SESSION_NAME%????????})
        if tmux new-session -s "$TMUX_SESSION_NAME" -d \; send-keys \
            "python3 test_denoising.py -s $MODEL_PATH --disconnect" Enter; then
            echo "Starting experiment \"$TMUX_SESSION_NAME\"..."
            sleep $WAIT_TIME_TESTING;
        fi
    done
}

function test_gating() {

    # look inside of root_dir for model checkpoints
    for MODEL_PATH in $(find $RESULTS_DIRECTORY/Gating/ -name *model.pt); do

        local MODEL_DIR=$(dirname "${MODEL_PATH}")
        if $SKIP_EXISTING; then if grep -rsq "Completed testing" $MODEL_DIR; then
            echo "Skipping $MODEL_DIR..."; continue; fi; fi

        local TMUX_SESSION_NAME=$(tr '/.' '  ' <<< $(echo $MODEL_PATH | sed 's/^.\{'$(echo -n $ROOT_DIR | wc -m)'\}//'))
        local TMUX_SESSION_NAME=$(echo ${TMUX_SESSION_NAME%????????})
        if tmux new-session -s "$TMUX_SESSION_NAME" -d \; send-keys \
            "python3 test_gating.py -s $MODEL_PATH --disconnect" Enter; then
            echo "Starting experiment \"$TMUX_SESSION_NAME\"..."
            sleep $WAIT_TIME_TESTING;
        fi
    done
}

function clear() {
    find $RESULTS_DIRECTORY/* -maxdepth 0 -exec rm -vr '{}' \;
}

"$@"