#!/bin/bash

exec python /run/break_background_wav_to_1_sec.py &
exec python /run/processing_and_training.py