import base64
import os
import random
import tempfile

from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen


def ls_dir(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            # get file size
            size = os.path.getsize(file_path)
            print(file_path, size)


class Model:
    def load(self):
        self.model = MusicGen.get_pretrained("small", device="cuda")

    def predict(self, request):
        try:
            prompts = request.pop("prompts")
            duration = request.pop("duration") or 8
            self.model.set_generation_params(duration=duration)
            wav = self.model.generate(prompts)
            output_files = []
            for idx, one_wav in enumerate(wav):
                with tempfile.NamedTemporaryFile() as tmpfile:
                    print(f"Writing {tmpfile.name}")
                    audio_write(
                        tmpfile.name,
                        one_wav.cpu(),
                        self.model.sample_rate,
                        strategy="loudness",
                    )
                    ls_dir("/tmp")
                    with open(tmpfile.name + ".wav", "rb") as f:
                        output_files.append(base64.b64encode(f.read()).decode("utf-8"))
                return {"data": output_files}

        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}
