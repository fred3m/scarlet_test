import os
import time

import numpy as np
import scarlet
from scarlet_extensions.initialization import initAllSources
from .measure import measure_blend


def deblend(blend_id: str, max_iter: int, e_rel: float, data_path: str, filters: str):
    # Load the sample images
    filename = os.path.join(data_path, "{}.npz".format(blend_id))
    data = np.load(filename)
    images = data["images"]
    mask = data["footprint"]
    weights = 1 / data["variance"] * ~mask
    centers = data["centers"]
    psfs = scarlet.PSF(data["psfs"])

    # Initialize the model, frame, observation, and sources
    t0 = time.time()
    from functools import partial
    model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=.8), shape=(None, 11, 11))

    model_frame = scarlet.Frame(
        images.shape,
        psfs=model_psf,
        channels=filters)

    observation = scarlet.Observation(
        images,
        psfs=psfs,
        weights=weights,
        channels=filters).match(model_frame)

    sources, skipped = initAllSources(model_frame, centers, observation, maxComponents=2, edgeDistance=None)

    # Fit the blend
    t1 = time.time()
    blend = scarlet.Blend(sources, observation)
    blend.fit(max_iter, e_rel=e_rel)
    t2 = time.time()

    if hasattr(observation, "log_norm"):
        log_norm = observation.log_norm
    else:
        _weights = observation.weights
        _images = observation.images
        log_sigma = np.zeros(_weights.shape, dtype=_weights.dtype)
        cuts = _weights > 0
        log_sigma[cuts] = np.log(1/weights[cuts])
        log_norm = np.prod(_images.shape)/2 * np.log(2*np.pi)+np.sum(log_sigma)/2

    measurements = {
        'init time': (t1 - t0) * 1000,
        'runtime': (t2 - t1) * 1000 / len(sources),
        'iterations': len(blend.loss),
        'logL': blend.loss[-1] - log_norm,
        'init logL': blend.loss[0] - log_norm,
    }

    for k in skipped:
        sources.insert(k, None)

    source_measurements = measure_blend(data, sources, filters)
    for measurement in source_measurements:
        measurement.update(measurements)

    return source_measurements, images, observation, sources
