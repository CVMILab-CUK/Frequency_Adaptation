import cv2
import numpy as np

from .frequency import high_pass_numpy, to_model_range


class Resize(object):
    """Resize both the ground-truth image and the conditioning source."""

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        h, w = self.shape[0], self.shape[1]
        data["filtered_image"] = cv2.resize(data["filtered_image"], dsize=(h, w),
                                            interpolation=cv2.INTER_LINEAR)
        data["gt"] = cv2.resize(data["gt"], dsize=(h, w), interpolation=cv2.INTER_LINEAR)
        return data


class SourceResolution(object):
    """Round-trip the image through a lower resolution before anything else.

    Reproduces the earlier setup, where 256px FFHQ images were resized up to
    512 because `makeDatasets` was called without `img_size`. That matters:
    an upscaled image carries its structure at lower absolute frequency, and
    the frozen VAE was measured to preserve the high-pass condition far better
    for such images (r=0.3: 0.798 vs 0.577 correlation through encode-decode; measured by
    scripts/vae_condition_survival.py).
    """

    def __init__(self, size):
        self.size = int(size)

    def __call__(self, data):
        for k in ("gt", "filtered_image"):
            h, w = data[k].shape[:2]
            small = cv2.resize(data[k], (self.size, self.size), interpolation=cv2.INTER_AREA)
            data[k] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return data


class Normalization(object):
    """Map the ground-truth image from [0, 1] to [-1, 1] for the VAE.

    Only `gt` is touched here; `filtered_image` is produced downstream by
    `Frequency_Filtering`, which applies the identical mapping itself.
    """

    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data["gt"] = (data["gt"] - self.mean) / self.std
        return data


class Frequency_Filtering(object):
    """Build the structural conditioning signal with an FFT high-pass filter.

    Args:
        frequency_img: 'gray' collapses colour before filtering (structure only,
            no chroma leakage); 'color' filters each channel independently.
        frequency_rate: 'random' samples a cutoff uniformly from `rate_range`
            per call -- this is the randomised-bandwidth training that lets one
            model cover the whole structure-density spectrum. A float pins the
            cutoff, which is what every evaluation uses.
        mode: 'high_pass' (structure) or 'low_pass' (blur).
        rate_range: the (min, max) cutoff sampled when frequency_rate=='random'.
        out_range: 'model' emits [-1, 1] (VAE-ready, matches `gt`); 'unit'
            emits [0, 1] (for visualisation only).

    The emitted `filtered_image` is always 3-channel so it can go straight
    through the frozen VAE encoder.
    """

    def __init__(self,
                 frequency_img="gray",
                 frequency_rate="random",
                 mode="high_pass",
                 rate_range=(0.0, 1.0),
                 rate_sampler="uniform",
                 out_range="model",
                 rng=None):
        assert frequency_img in ("gray", "color")
        assert out_range in ("model", "unit")
        assert rate_sampler in ("uniform", "sqrt", "log")
        if frequency_rate != "random":
            assert 0.0 <= float(frequency_rate) <= 1.0, "frequency_rate must be in [0, 1]"
        self.frequency_img = frequency_img
        self.frequency_rate = frequency_rate
        self.mode = mode
        self.rate_range = tuple(rate_range)
        self.rate_sampler = rate_sampler
        self.out_range = out_range
        self.rng = rng

    def sample_rate(self):
        """Draw a cutoff for this sample.

        Measured on FFHQ-512: log10(retained spectral energy) falls almost
        linearly in r over [0.05, 0.7] (-0.32, -0.33, -0.22, -0.37 dex per
        step), so `uniform` sampling in r is already close to uniform in
        information content -- which is why it is the default. The exception is
        r < 0.05, where energy drops ~10x inside a very narrow band; `sqrt`
        concentrates samples there so that regime is not undertrained. `log`
        goes further and is spaced geometrically. The three are an ablation
        axis, not a free parameter to tune on the test set.
        """
        if self.frequency_rate != "random":
            return float(self.frequency_rate)

        r = self.rng if self.rng is not None else np.random
        lo, hi = self.rate_range
        u = float(r.uniform(0.0, 1.0))
        if self.rate_sampler == "uniform":
            return lo + u * (hi - lo)
        if self.rate_sampler == "sqrt":
            return lo + (u ** 2) * (hi - lo)
        lo_c = max(lo, 1e-3)
        return float(np.exp(np.log(lo_c) + u * (np.log(hi) - np.log(lo_c))))

    def __call__(self, data):
        image = data.get("filtered_image", data.get("image"))
        if image is None:
            return data

        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        if self.frequency_img == "gray" and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]

        cutoff = self.sample_rate()
        filtered = high_pass_numpy(image, cutoff, mode=self.mode)  # [0, 1]

        if filtered.shape[2] == 1:
            filtered = np.repeat(filtered, 3, axis=2)

        if self.out_range == "model":
            filtered = to_model_range(filtered)

        data["filtered_image"] = filtered.astype(np.float32)
        # Recorded so the trainer/evaluator can log or condition on the actual
        # cutoff that produced this sample.
        data["cutoff"] = np.float32(cutoff)
        return data
