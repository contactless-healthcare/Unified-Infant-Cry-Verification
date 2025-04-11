from speechbrain.pretrained import SpeakerRecognition
import dill

# Load model
encoder = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="spkrec-ecapa-voxceleb"
)

# Save with dill
with open("ecapa_tdnn.dill", "wb") as f:
    dill.dump(encoder, f)