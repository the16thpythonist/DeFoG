#!/bin/bash
# Submit the ZINC conditional training experiment to HAICORE (1 GPU, 1 hour)
aslurmx -cn haicore_1gpu -o time=01:00:00 cmd python experiments/conditional_training__zinc_det.py
