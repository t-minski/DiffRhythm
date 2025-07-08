cd "$(dirname "$0")"
cd ../

export PYTHONPATH=$PYTHONPATH:$PWD
export CUDA_VISIBLE_DEVICES=0

if [[ "$OSTYPE" =~ ^darwin ]]; then
    export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib
fi

# Default paths
LRC_PATH="infer/example/eg_cn_full.lrc"
WAV_PATH="infer/example/eg_cn.wav"

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --lrc)
        LRC_PATH="$2"
        shift # past argument
        shift # past value
        ;;
        --wav)
        WAV_PATH="$2"
        shift
        shift
        ;;
        *)    # unknown option
        shift
        ;;
    esac
done


python3 infer/infer.py \
    --lrc-path "$LRC_PATH" \
    --ref-audio-path "$WAV_PATH" \
    --audio-length 285 \
    --repo_id ASLP-lab/DiffRhythm-full \
    --output-dir infer/example/output \
    --chunked