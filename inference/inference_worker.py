from kafka import KafkaConsumer, KafkaProducer
import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import time


# ==========================================
# Kafka Configuration
# ==========================================

BOOTSTRAP_SERVERS = "localhost:9092"

INPUT_TOPIC = "processed_features"
OUTPUT_TOPIC = "anomaly_scores"


# ==========================================
# Autoencoder Architecture
# ==========================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# ==========================================
# Load Models
# ==========================================

print("Loading models...")

if_model = joblib.load(
    "models/saved_models/isolation_forest.pkl"
)

metadata = joblib.load(
    "models/saved_models/ae_metadata.pkl"
)

threshold = float(
    joblib.load(
        "models/saved_models/ae_threshold.pkl"
    )
)

input_dim = metadata["input_dim"]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")

ae_model = Autoencoder(input_dim).to(device)

ae_model.load_state_dict(
    torch.load(
        "models/saved_models/autoencoder.pth",
        map_location=device
    )
)

ae_model.eval()

print("Models loaded successfully")


# ==========================================
# Kafka Consumer
# ==========================================

consumer = KafkaConsumer(
    INPUT_TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda x: json.loads(
        x.decode("utf-8")
    ),
    auto_offset_reset="latest",
    group_id="inference-worker-v2"
)


# ==========================================
# Kafka Producer
# ==========================================

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode(
        "utf-8"
    )
)

processed = 0

print("Inference Worker Started...")


# ==========================================
# Inference Loop
# ==========================================

try:

    for msg in consumer:

        features = msg.value["features"]

        X = np.array(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        # ------------------------------
        # Isolation Forest Score
        # ------------------------------

        if_score = float(
            -if_model.score_samples(X)[0]
        )

        # ------------------------------
        # Autoencoder Score
        # ------------------------------

        tensor_x = torch.tensor(
            X,
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            reconstructed = ae_model(tensor_x)

        reconstruction_error = float(
            torch.mean(
                (tensor_x - reconstructed) ** 2
            ).cpu().item()
        )

        ae_score = min(
            reconstruction_error / threshold,
            1.0
        )

        ae_anomaly = int(
            reconstruction_error > threshold
        )

        # ------------------------------
        # Output Message
        # ------------------------------

        result = {
            # Original 59 processed features
            "features": features,

            # Model outputs
            "if_score": if_score,
            "ae_error": reconstruction_error,
            "ae_score": ae_score,
            "ae_anomaly": ae_anomaly,
            "threshold": threshold,

            "timestamp": time.time()
        }

        producer.send(
            OUTPUT_TOPIC,
            value=result
        )

        processed += 1

        if processed % 1000 == 0:
            print(
                f"Scored {processed} records"
            )
            producer.flush()

except KeyboardInterrupt:

    print("\nStopping inference worker...")

finally:

    producer.flush()
    producer.close()
    consumer.close()

    print("Inference worker stopped.")