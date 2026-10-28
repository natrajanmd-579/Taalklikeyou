try:
    import pyximport
    pyximport.install()
    from .blend import blend_images_cy as blend_images_cy  # noqa: E402
except Exception:
    import numpy as np

    def blend_images_cy(
        mask_warped: np.ndarray,
        frame_warped: np.ndarray,
        frame_rgb: np.ndarray,
        result: np.ndarray,
    ):
        mask = mask_warped[:, :, None]
        blended = mask * frame_warped + (1.0 - mask) * frame_rgb
        np.clip(blended, 0, 255, out=blended)
        result[:] = blended.astype(np.uint8)

