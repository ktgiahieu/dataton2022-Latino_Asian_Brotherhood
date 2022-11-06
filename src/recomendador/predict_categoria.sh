#!/usr/bin/env bash

cd src/deliverable_categorizacion
python3 3-Infer_participacion_classifier.py
python3 5-Infer_categoria_classifier copy.py
python3 6-Combine_participacion_categoria.py
