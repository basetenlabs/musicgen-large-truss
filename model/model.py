import base64
import random

import torchaudio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen


class Model:
    def load(self):
        self.model = MusicGen.get_pretrained("small", device="cuda")

    def predict(self, request):
        with torchaudio.no_grad():
            try:
                prompts = request.pop("prompts")
                duration = request.pop("duration")
                self.model.set_generation_params(duration=duration)
                wav = self.model.generate(prompts)
                filename = hex(random.randint(0, 1000000))[2:]
                audio_write(
                    f"{filename}",
                    wav.cpu(),
                    self.model.sample_rate,
                    strategy="loudness",
                    loudness_compressor=True,
                )
                # Read file and encode it into base64 format
                with open(f"{filename}.wav", "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                    return {"data": data}

            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}
