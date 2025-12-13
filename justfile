full:
    @echo "Full Pipeline (Feature + Training) run started.."
    @python app/training_endpoint.py

full_offline:
    @echo "Full Pipeline (Feature + Training) run started.."
    @python app/training_endpoint.py pipeline.type=offline

full_offline:
    @echo "Full Pipeline (Feature + Training) run started.."
    @python app/training_endpoint.python pipeline.type=offline

feature:
    @echo "Feature Pipeline Only started.."
    @python app/training_endpoint.py pipeline.stage=feature

feature_offline:
    @echo "Feature Pipeline Only started.."
    @python app/training_endpoint.py pipeline.stage=feature pipeline.type=offline

train:
    @echo "Training Pipeline Only started.."
    @python app/training_endpoint.py pipeline.stage=train

train_offline:
    @echo "Training Pipeline Only started.."
    @python app/training_endpoint.py pipeline.stage=train pipeline.type=offline

infer:
    @echo "Inference Pipeline started.."
    @fastapi dev app/inference_endpoint.py