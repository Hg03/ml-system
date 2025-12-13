full:
    @echo "Full Pipeline (Feature + Training) run started.."
    @python app/training_endpoint.python

feature:
    @echo "Feature Pipeline Only started.."
    @python app/training_endpoint.py pipeline.stage=feature

train:
    @echo "Training Pipeline Only started.."
    @python app/training_endpoint.py pipeline.stage=train

infer:
    @echo "Inference Pipeline started.."
    @fastapi dev app/inference_endpoint.py